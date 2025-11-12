from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from asgiref.sync import sync_to_async
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from memory.models import Slides, SlidesRevision, _strip_ctrl

from .utils import (
    DB_PRIMARY,
    EditorJS,
    get_tool_context,
    normalize_editorjs,
    minimal_editorjs,
    _ensure_single_top_header,
    _now_ms,
    _llm,
)

log = logging.getLogger(__name__)

# Require 3–9 sections (i.e., 4–10 total slides counting the title slide)
MIN_SECTION_SLIDES = 3
MAX_SECTION_SLIDES = 9
MAX_BULLETS_PER_SECTION = 8

# --------------------------------------------------------------------------------------
# Summary helpers (multiline bullets everywhere)
# --------------------------------------------------------------------------------------
def _bulletize(lines: List[str]) -> str:
    clean = []
    for x in lines:
        s = _strip_ctrl(str(x or "")).strip()
        if s:
            clean.append(s)
    return "\n".join(f"- {s}" for s in clean)

# ======================================================================================
# Input Schemas
# ======================================================================================
class SlidesGenerateInput(BaseModel):
    prompt: Optional[str] = Field(None, description="Topic / natural language request for the deck. Creates a new deck.")
    title: Optional[str] = Field(None, description="Deck title (optional; inferred if omitted).")
    summary: Optional[str] = Field(None, description="Short abstract (optional).")
    editorjs: Optional[EditorJS | List[Dict[str, Any]]] = Field(None, description="Ignored; always generates fresh content.")
    context: Optional[str] = Field(None, description="Additional context for the deck (optional).")
    max_sections: int = Field(6, ge=3, le=16)

    @field_validator("editorjs")
    @classmethod
    def _validate_editorjs(cls, v):
        if v is None:
            return v
        ej = normalize_editorjs(v)
        if not ej:
            raise ValueError("editorjs must contain non-empty blocks")
        return ej


class SlidesRevertInput(BaseModel):
    version: Optional[int] = Field(None, description="Version to revert to. If omitted, previous version is used.")


class SlidesListVersionsInput(BaseModel):
    limit: int = Field(10, ge=1, le=50, description="How many versions to list")


class SlidesDiffInput(BaseModel):
    compare_to_version: Optional[int] = Field(None, description="If omitted, compare latest vs previous")


class FetchSlidesInput(BaseModel):
    pass


class SectionSpec(BaseModel):
    header: str = Field(..., description="Section header")
    bullets: List[str] = Field(default_factory=list, description="Bullet points (0..8)")


class SlidesAddSectionsInput(BaseModel):
    sections: List[SectionSpec] = Field(default_factory=list, description="Sections to add, in order")
    after: Optional[str] = Field(None, description="Insert AFTER first section whose header contains this substring; append if not found.")
    count: int = Field(1, ge=1, le=20, description="If a single SectionSpec is provided, duplicate it this many times.")


class SlidesRemoveSectionsInput(BaseModel):
    header: Optional[str] = Field(None, description="Match by substring (case-insensitive). If omitted, remove from end.")
    count: int = Field(1, ge=1, le=20)
    all_matches: bool = Field(False, description="If true with a header, remove all matches up to `count`.")


class SlidesEditInput(BaseModel):
    replace_editorjs: Optional[EditorJS | List[Dict[str, Any]]] = None
    new_title: Optional[str] = None
    new_summary: Optional[str] = None

    class SectionEdit(BaseModel):
        index: Optional[int] = None
        header_icontains: Optional[str] = None
        new_header: Optional[str] = None
        new_bullets: Optional[List[str]] = None
        append_bullets: Optional[List[str]] = None
        remove_bullets_indices: Optional[List[int]] = None

    section_edits: List[SectionEdit] = Field(default_factory=list)

# ======================================================================================
# Helpers (EditorJS)
# ======================================================================================
def _ej_blocks(ej: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list((ej or {}).get("blocks") or [])

def _is_header(block: Dict[str, Any], level: int) -> bool:
    if (block.get("type") or "").lower() != "header":
        return False
    try:
        lvl = int((block.get("data") or {}).get("level", 2))
    except Exception:
        return False
    return lvl == level

def _is_h3_section_header(block: Dict[str, Any]) -> bool:
    return _is_header(block, 3) and bool(_strip_ctrl((block.get("data") or {}).get("text") or "").strip())

def _h_text(block: Dict[str, Any]) -> str:
    return _strip_ctrl(((block.get("data") or {}).get("text") or "")).strip()

def _count_sections(blocks: List[Dict[str, Any]]) -> int:
    return sum(1 for b in blocks if _is_h3_section_header(b))

def _find_section_indices(blocks: List[Dict[str, Any]]) -> List[int]:
    return [i for i, b in enumerate(blocks) if _is_h3_section_header(b)]

def _find_section_by_icontains(blocks: List[Dict[str, Any]], needle: str) -> Optional[int]:
    n = _strip_ctrl(needle or "").lower()
    if not n:
        return None
    for i in _find_section_indices(blocks):
        if n in _h_text(blocks[i]).lower():
            return i
    return None

def _end_of_section(blocks: List[Dict[str, Any]], start_idx: int) -> int:
    j = start_idx
    if j + 1 < len(blocks) and (blocks[j + 1].get("type") or "").lower() in {"list", "paragraph"}:
        j += 1
    return j

def _bullet_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return _strip_ctrl(x).strip()
    if isinstance(x, dict):
        for k in ("content", "text", "title", "value"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return _strip_ctrl(v).strip()
        if isinstance(x.get("items"), list):
            joined = ", ".join(_bullet_text(i) for i in x["items"])
            return _strip_ctrl(joined).strip()
    return _strip_ctrl(f"{x}").strip()

def _mk_section_blocks(header: str, bullets: List[Any]) -> List[Dict[str, Any]]:
    header = _strip_ctrl(header or "Section")
    items = [_bullet_text(b) for b in (bullets or [])]
    items = [b for b in items if b][:MAX_BULLETS_PER_SECTION]
    if not items:
        items = ["Key point 1", "Key point 2", "Key point 3"]
    return [
        {"type": "header", "data": {"text": header, "level": 3}},
        {"type": "list", "data": {"style": "unordered", "items": items}},
    ]

def _ensure_editorjs(ej_like: EditorJS | List[Dict[str, Any]] | Dict[str, Any]) -> Dict[str, Any]:
    ej = normalize_editorjs(ej_like) if isinstance(ej_like, (dict, list)) else {}
    blocks = _ensure_single_top_header(_ej_blocks(ej), None)

    cleaned: List[Dict[str, Any]] = []
    for b in blocks:
        t = (b.get("type") or "").lower()
        d = dict(b.get("data") or {})
        if t == "list":
            items = d.get("items") or []
            if not isinstance(items, list):
                items = []
            d["items"] = [x for x in (_bullet_text(i) for i in items) if x]
            d["style"] = "unordered"
        elif t == "header":
            try:
                lvl = int(d.get("level", 2))
            except Exception:
                lvl = 2
            d["level"] = 2 if cleaned == [] else (lvl if lvl in (1, 2, 3) else 3)
            d["text"] = _strip_ctrl(d.get("text", ""))
        elif t == "paragraph":
            d["text"] = _strip_ctrl(d.get("text", ""))
        cleaned.append({"type": t, "data": d})

    return {"time": _now_ms(), "version": "2.x", "blocks": cleaned}

def _clamp_section_bounds(blocks: List[Dict[str, Any]]) -> Tuple[bool, str]:
    n = _count_sections(blocks)
    if n < MIN_SECTION_SLIDES:
        return False, f"Deck must have at least {MIN_SECTION_SLIDES} sections (has {n})."
    if n > MAX_SECTION_SLIDES:
        return False, f"Deck cannot exceed {MAX_SECTION_SLIDES} sections (has {n})."
    return True, ""

def _editorjs_stats(ej: Dict[str, Any]) -> Dict[str, Any]:
    h2 = []
    h3 = []
    paragraphs = 0
    list_items = 0
    blocks = _ej_blocks(ej)
    for b in blocks:
        t = (b.get("type") or "").lower()
        d = b.get("data") or {}
        if t == "header":
            try:
                lvl = int(d.get("level", 2) or 2)
            except Exception:
                lvl = 2
            txt = _strip_ctrl(d.get("text") or "")
            if lvl <= 2:
                h2.append(txt)
            else:
                h3.append(txt)
        elif t == "paragraph":
            paragraphs += 1
        elif t == "list":
            items = d.get("items") or []
            list_items += len([_bullet_text(x) for x in items if _bullet_text(x)])
    return {"h2": h2, "h3": h3, "paragraphs": paragraphs, "list_items": list_items, "blocks": len(blocks)}

def _summarize_editorjs_diff(old_ej: Dict[str, Any], new_ej: Dict[str, Any]) -> List[str]:
    o = _editorjs_stats(old_ej or {})
    n = _editorjs_stats(new_ej or {})
    lines: List[str] = []
    if (o["h2"][:1] or [""])[0] != (n["h2"][:1] or [""])[0]:
        lines.append(f"Title changed: “{(o['h2'][:1] or [''])[0]}” → “{(n['h2'][:1] or [''])[0]}”.")
    added_h3 = [h for h in n["h3"] if h and h not in o["h3"]]
    removed_h3 = [h for h in o["h3"] if h and h not in n["h3"]]
    if added_h3:
        lines.append(f"Added {len(added_h3)} section(s): " + ", ".join(added_h3[:3]) + ("…" if len(added_h3) > 3 else ""))
    if removed_h3:
        lines.append(f"Removed {len(removed_h3)} section(s): " + ", ".join(removed_h3[:3]) + ("…" if len(removed_h3) > 3 else ""))
    di = n["list_items"] - o["list_items"]
    if di:
        lines.append(("+" if di > 0 else "") + f"{di} bullet(s) total.")
    db = n["blocks"] - o["blocks"]
    if db:
        lines.append(("+" if db > 0 else "") + f"{db} blocks.")
    if not lines:
        lines.append("Minor text edits; structure unchanged.")
    return lines

async def _llm_summarize_change(*, title: str, before_note: str, after_note: str, model: Optional[str] = None) -> str:
    llm = _llm(model=model, temperature=0.2)
    sys = "Summarize in 2–4 short bullets what changed and what the result is. Be concrete."
    user = f"Title: {title}\n\nBEFORE:\n{before_note.strip() or '(none)'}\n\nAFTER:\n{after_note.strip() or '(none)'}"
    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user[:4000])])
        return (getattr(msg, "content", "") or "").strip()[:800]
    except Exception:
        return f"Updated “{title}”. (Summary unavailable.)"

# ======================================================================================
# Slides sync wrappers (ORM in threads)
# ======================================================================================
@sync_to_async(thread_sensitive=True)
def _slides_fetch_latest_sync(session_id: int):
    return (
        Slides.objects.using(DB_PRIMARY)
        .filter(session_id=session_id)
        .order_by("-updated_at", "-id")
        .first()
    )

@sync_to_async(thread_sensitive=True)
def _slides_upsert_sync(session_id: int, *, title: str, summary: str, editorjs: EditorJS, updated_by: str):
    from django.db import transaction
    with transaction.atomic(using=DB_PRIMARY):
        s = (
            Slides.objects.using(DB_PRIMARY)
            .filter(session_id=session_id)
            .select_for_update()
            .first()
        )
        before_note = f"title={getattr(s, 'title', '')!r}, v={getattr(s, 'version', 0)}" if s else "(none)"
        if not s:
            s = Slides.objects.using(DB_PRIMARY).create(session_id=session_id)
        ej = _ensure_editorjs(editorjs)
        s.rotate_and_update(title=title or "Untitled Deck", summary=summary or "", editorjs=ej, updated_by=updated_by)
        s.refresh_from_db(using=DB_PRIMARY)
        return s, before_note

@sync_to_async(thread_sensitive=True)
def _slides_revert_sync(session_id: int, version: Optional[int]):
    from django.db import transaction
    with transaction.atomic(using=DB_PRIMARY):
        s = (
            Slides.objects.using(DB_PRIMARY)
            .select_for_update()
            .filter(session_id=session_id)
            .first()
        )
        if not s:
            raise ValueError("No slide deck to revert.")
        before_note = f"title={s.title!r}, v={s.version}"
        if version is None:
            prev = (
                SlidesRevision.objects.using(DB_PRIMARY)
                .filter(session_id=session_id, version__lt=s.version)
                .order_by("-version")
                .first()
            )
            if not prev:
                raise ValueError("No previous version to revert to.")
            target_version = prev.version
        else:
            target_version = int(version)
        s.revert_to_version(target_version=target_version, updated_by="tool:slides_revert")
        s.refresh_from_db(using=DB_PRIMARY)
        return s, before_note, target_version

@sync_to_async(thread_sensitive=True)
def _slides_list_revisions_sync(session_id: int, limit: int = 10):
    return list(
        SlidesRevision.objects.using(DB_PRIMARY)
        .filter(session_id=session_id)
        .order_by("-version")[: max(1, min(50, limit))]
        .values("version", "title", "created_at", "updated_by")
    )

@sync_to_async(thread_sensitive=True)
def _slides_get_revision_sync(session_id: int, version: Optional[int] = None):
    if version is None:
        s = Slides.objects.using(DB_PRIMARY).filter(session_id=session_id).first()
        if not s:
            return None, None
        prev = (
            SlidesRevision.objects.using(DB_PRIMARY)
            .filter(session_id=session_id, version__lt=s.version)
            .order_by("-version")
            .first()
        )
        return s, prev
    else:
        s = SlidesRevision.objects.using(DB_PRIMARY).filter(session_id=session_id, version=version).first()
        return s, None

# ======================================================================================
# Formatting + Return Contract (ONLY: status, summary, data.slides)
# ======================================================================================
def _return_ok(slides: Optional[Slides], summary: str) -> Dict[str, Any]:
    if not slides:
        return {"status": "ok", "summary": summary, "data": {"slides": {}}}
    ej = _ensure_editorjs(slides.editorjs or {})
    return {
        "status": "ok",
        "summary": summary,
        "data": {
            "slides": {
                "version": slides.version,
                "title": slides.title or "",
                "summary": slides.summary or "",
                "editorjs": ej,
                "updated_at": slides.updated_at.isoformat(),
            }
        },
    }

def _return_failed(slides: Optional[Slides], summary: str) -> Dict[str, Any]:
    if not slides:
        return {"status": "failed", "summary": summary, "data": {"slides": {}}}
    ej = _ensure_editorjs(slides.editorjs or {})
    return {
        "status": "failed",
        "summary": summary,
        "data": {
            "slides": {
                "version": slides.version,
                "title": slides.title or "",
                "summary": slides.summary or "",
                "editorjs": ej,
                "updated_at": slides.updated_at.isoformat(),
            }
        },
    }

# ======================================================================================
# Outline generation
# ======================================================================================
def _fallback_sections_from_text(context: str, need_n: int) -> List[Dict[str, Any]]:
    import re
    text = (_strip_ctrl(context or "")).strip()
    if not text:
        seeds = [
            ("Overview", ["Goal of this topic", "Why it matters", "What we'll cover"]),
            ("Key Ideas", ["Idea #1", "Idea #2", "Idea #3"]),
            ("Details", ["How it works", "Constraints", "Tradeoffs"]),
            ("Examples", ["Case A", "Case B", "Case C"]),
            ("Summary", ["Takeaways", "Next steps", "Q&A"]),
        ]
    else:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        seeds = []
        for i, p in enumerate(paras[:need_n]):
            parts = re.split(r"(?:[.!?]\s+|,\s+|\n)", p)
            bullets = [b.strip() for b in parts if b and len(b.strip()) > 3][:6]
            if not bullets:
                bullets = [p[:80] + ("…" if len(p) > 80 else "")]
            seeds.append((f"Section {i+1}", bullets))
        default_pad = [("Summary", ["Top takeaways", "Recommendations", "Next steps"])]
        while len(seeds) < need_n:
            seeds += default_pad
    return [{"header": h, "bullets": bs[:MAX_BULLETS_PER_SECTION]} for (h, bs) in seeds[:need_n]]

async def _llm_outline_to_editorjs(
    *,
    prompt: str,
    context: str,
    max_sections: int = 6,
    model: Optional[str] = None,
) -> EditorJS:
    sec_target_max = min(MAX_SECTION_SLIDES, max_sections)
    sec_target = max(MIN_SECTION_SLIDES, sec_target_max)

    llm = _llm(model=model, temperature=0.3)
    sys = (
        "Return STRICT JSON for a slide outline.\n"
        'Schema: {"title":"string","summary":"1-2 sentences","sections":[{"header":"string","bullets":["point",...]}]}'
    )
    prefix = ("Context:\n" + (context or "").strip() + "\n\n") if (context or "").strip() else ""
    user = prefix + ("Request:\n" + (prompt or "")[:4000])

    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user)])
        raw = (getattr(msg, "content", "") or "").strip()
        data = json.loads(raw if raw.startswith("{") else raw[raw.find("{"):raw.rfind("}") + 1])
        title = _strip_ctrl((data.get("title") or "Untitled Deck")[:140])
        summary = _strip_ctrl((data.get("summary") or "")[:300])
        sections = data.get("sections") or []

        if len(sections) < MIN_SECTION_SLIDES:
            sections = sections + _fallback_sections_from_text(context, MIN_SECTION_SLIDES - len(sections))
        if len(sections) > sec_target:
            sections = sections[:sec_target]

        blocks: List[Dict[str, Any]] = [{"type": "header", "data": {"text": title, "level": 2}}]
        if summary:
            blocks.append({"type": "paragraph", "data": {"text": summary}})

        for s in sections:
            hdr = _strip_ctrl((s.get("header") or "").strip()) or "Section"
            bullets_raw = s.get("bullets") or []
            bullets = [_bullet_text(x) for x in bullets_raw]
            bullets = [b for b in bullets if b][:MAX_BULLETS_PER_SECTION]
            blocks.append({"type": "header", "data": {"text": hdr, "level": 3}})
            blocks.append({"type": "list", "data": {"style": "unordered", "items": bullets}})

        ej = _ensure_editorjs({"time": _now_ms(), "version": "2.x", "blocks": blocks})
        return ej or minimal_editorjs(title, summary)
    except Exception:
        title = (prompt or "Untitled Deck").strip()[:140] or "Untitled Deck"
        sections = _fallback_sections_from_text(context, MIN_SECTION_SLIDES)
        blocks = [{"type": "header", "data": {"text": _strip_ctrl(title), "level": 2}}]
        for s in sections:
            blocks.extend(_mk_section_blocks(_strip_ctrl(s["header"]), s["bullets"][:MAX_BULLETS_PER_SECTION]))
        return {"time": _now_ms(), "version": "2.x", "blocks": blocks}

# ======================================================================================
# Local EditorJS edit helpers
# ======================================================================================
def _insert_sections(blocks: List[Dict[str, Any]], sections: List[SectionSpec], after: Optional[str], count: int) -> List[Dict[str, Any]]:
    to_add: List[Dict[str, Any]] = []
    if len(sections) == 1 and count > 1:
        base = sections[0]
        for i in range(count):
            suffix = f" ({i+1})" if i > 0 else ""
            to_add.extend(_mk_section_blocks(base.header + suffix, base.bullets))
    else:
        for s in sections:
            to_add.extend(_mk_section_blocks(s.header, s.bullets))

    if not to_add:
        return blocks

    insert_at = None
    if after:
        idx = _find_section_by_icontains(blocks, after)
        if idx is not None:
            insert_at = _end_of_section(blocks, idx) + 1

    if insert_at is None:
        secs = _find_section_indices(blocks)
        insert_at = _end_of_section(blocks, secs[-1]) + 1 if secs else len(blocks)

    return blocks[:insert_at] + to_add + blocks[insert_at:]

def _remove_sections(blocks: List[Dict[str, Any]], header: Optional[str], count: int, all_matches: bool) -> List[Dict[str, Any]]:
    indices = _find_section_indices(blocks)
    if not indices:
        return blocks

    targets: List[int] = []
    if header:
        h = _strip_ctrl(header).lower()
        for i in indices:
            if h in _h_text(blocks[i]).lower():
                targets.append(i)
                if not all_matches and len(targets) >= count:
                    break
        if all_matches:
            targets = targets[:count]
    else:
        targets = list(reversed(indices))[:count]

    if not targets:
        return blocks

    targets = sorted(set(targets))
    new_blocks = blocks[:]
    for idx in reversed(targets):
        remaining_sections = _count_sections(new_blocks) - 1
        if remaining_sections < MIN_SECTION_SLIDES:
            continue
        end = _end_of_section(new_blocks, idx)
        del new_blocks[idx : end + 1]

    return new_blocks

def _find_list_after_section(blocks: List[Dict[str, Any]], sec_idx: int) -> Optional[int]:
    j = sec_idx + 1
    if j < len(blocks) and (blocks[j].get("type") or "").lower() == "list":
        return j
    return None

def _apply_section_edit(blocks: List[Dict[str, Any]], edit: SlidesEditInput.SectionEdit) -> List[Dict[str, Any]]:
    indices = _find_section_indices(blocks)
    if not indices:
        return blocks

    target_idx: Optional[int] = None
    if edit.index is not None:
        if 0 <= edit.index < len(indices):
            target_idx = indices[edit.index]
    elif edit.header_icontains:
        target_idx = _find_section_by_icontains(blocks, edit.header_icontains)

    if target_idx is None:
        return blocks

    if edit.new_header is not None:
        blocks[target_idx] = {"type": "header", "data": {"text": _strip_ctrl(edit.new_header), "level": 3}}

    need_bullet_change = (edit.new_bullets is not None) or (edit.append_bullets is not None) or (edit.remove_bullets_indices is not None)
    list_idx = _find_list_after_section(blocks, target_idx)
    if need_bullet_change and list_idx is None:
        list_idx = target_idx + 1
        blocks.insert(list_idx, {"type": "list", "data": {"style": "unordered", "items": []}})

    if list_idx is not None:
        items = list((blocks[list_idx].get("data") or {}).get("items") or [])
        if edit.new_bullets is not None:
            items = [_bullet_text(x) for x in (edit.new_bullets or []) if _bullet_text(x)]
        if edit.append_bullets is not None:
            items += [_bullet_text(x) for x in (edit.append_bullets or []) if _bullet_text(x)]
        if edit.remove_bullets_indices:
            for i in sorted(set(edit.remove_bullets_indices), reverse=True):
                if 0 <= i < len(items):
                    del items[i]
        items = [x for x in items if x][:MAX_BULLETS_PER_SECTION]
        blocks[list_idx] = {"type": "list", "data": {"style": "unordered", "items": items}}

    return blocks

# ======================================================================================
# Tools (STRICT RETURN: only status, summary, data.slides) — with MULTILINE bullet summaries
# ======================================================================================
@tool("slides_fetch", args_schema=FetchSlidesInput, description="Fetch the latest slides.")
async def slides_fetch() -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _return_failed(None, _bulletize(["Missing session context."]))

    s = await _slides_fetch_latest_sync(int(session_id))
    if not s:
        return _return_ok(None, _bulletize(["No slides found.", "Share a topic to create a deck."]))
    return _return_ok(s, _bulletize([
        f"Fetched latest v{s.version}.",
        f"Updated: {s.updated_at.isoformat()}",
        f"Title: {s.title or 'Untitled Deck'}",
    ]))


@tool("slides_generate_or_update", args_schema=SlidesGenerateInput, description="Generate a new deck (title + 3..9 sections). Rotates version.")
async def slides_generate_or_update(
    prompt: Optional[str] = None,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    editorjs: Optional[EditorJS] = None,
    context: Optional[str] = None,
    max_sections: int = 6,
) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _return_failed(None, _bulletize(["Missing session context."]))

    seed = _strip_ctrl((title or prompt or "Untitled Deck").strip())
    summary_clean = _strip_ctrl((summary or "").strip())
    context_clean = _strip_ctrl((context or "").strip())

    ej = await _llm_outline_to_editorjs(prompt=seed, context=context_clean, max_sections=max_sections)
    ej = _ensure_editorjs(ej)
    ok, msg = _clamp_section_bounds(_ej_blocks(ej))
    if not ok:
        return _return_failed(None, _bulletize(["Invalid deck.", msg]))

    s, _ = await _slides_upsert_sync(
        int(session_id),
        title=ej["blocks"][0]["data"]["text"] if ej["blocks"] else seed,
        summary=summary_clean,
        editorjs=ej,
        updated_by="tool:slides_generate_or_update",
    )
    stats = _editorjs_stats(ej)
    return _return_ok(
        s,
        _bulletize([
            f"Created/updated deck v{s.version}.",
            f"Title: {s.title or 'Untitled Deck'}",
            f"Sections: {len(stats.get('h3', []))}",
        ]),
    )

@tool("slides_list_versions", args_schema=SlidesListVersionsInput, description="List version count (no payload, only latest slides returned).")
async def slides_list_versions(limit: int = 10) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _return_failed(None, _bulletize(["Missing session context."]))

    s = await _slides_fetch_latest_sync(int(session_id))
    versions = await _slides_list_revisions_sync(int(session_id), limit=limit)
    if not s:
        return _return_ok(None, _bulletize([f"No slides found.", f"{len(versions)} revision(s) exist for this session."]))
    return _return_ok(s, _bulletize([f"Found {len(versions)} revision(s).", f"Latest: v{s.version}."]))

@tool("slides_revert", args_schema=SlidesRevertInput, description="Revert to a previous version (or previous if omitted).")
async def slides_revert(version: Optional[int] = None) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _return_failed(None, _bulletize(["Missing session context."]))

    current = await _slides_fetch_latest_sync(int(session_id))
    if not current:
        return _return_failed(None, _bulletize(["No slides found to revert."]))

    try:
        s, _before, target_version = await _slides_revert_sync(int(session_id), version)
    except Exception as e:
        return _return_failed(current, _bulletize([f"Revert failed: {e}"]))

    return _return_ok(s, _bulletize([f"Reverted to v{target_version}.", f"Current version: v{s.version}."]))

@tool("slides_diff", args_schema=SlidesDiffInput, description="Compare latest to a previous version (or previous by default).")
async def slides_diff(compare_to_version: Optional[int] = None) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _return_failed(None, _bulletize(["Missing session context."]))

    current = await _slides_fetch_latest_sync(int(session_id))
    if not current:
        return _return_ok(None, _bulletize(["No slides found.", "Share a topic to create a deck."]))

    if compare_to_version is None:
        _, prev = await _slides_get_revision_sync(int(session_id), None)
        base = prev
    else:
        base, _ = await _slides_get_revision_sync(int(session_id), int(compare_to_version))

    ej_current = _ensure_editorjs(current.editorjs or {})
    ej_base = _ensure_editorjs((getattr(base, "editorjs", None) or {}) or {})
    lines = _summarize_editorjs_diff(ej_base, ej_current)
    base_v = getattr(base, "version", current.version - 1 if current.version else None)

    summary = _bulletize([f"Compared v{current.version} to v{base_v if base_v is not None else 'N/A'}"] + lines)
    return _return_ok(current, summary)

@tool("slides_add_sections", args_schema=SlidesAddSectionsInput, description="Add sections (H3 + bullets).")
async def slides_add_sections(sections: List[SectionSpec], after: Optional[str] = None, count: int = 1) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _return_failed(None, _bulletize(["Missing session context."]))

    s = await _slides_fetch_latest_sync(int(session_id))
    if not s:
        return _return_ok(None, _bulletize(["No slides found.", "Share a topic to create a deck."]))

    before_ej = _ensure_editorjs(s.editorjs or {})
    blocks = _ej_blocks(before_ej)
    before_n = _count_sections(blocks)

    new_blocks = _insert_sections(blocks, sections, after, count)
    ok, msg = _clamp_section_bounds(new_blocks)
    if not ok:
        return _return_failed(s, _bulletize(["Not applied.", msg]))

    after_n = _count_sections(new_blocks)
    added_n = max(0, after_n - before_n)

    new_ej = {"time": _now_ms(), "version": "2.x", "blocks": new_blocks}
    s2, _ = await _slides_upsert_sync(
        int(session_id),
        title=_h_text(new_blocks[0]) if new_blocks else s.title,
        summary=s.summary,
        editorjs=new_ej,
        updated_by="tool:slides_add_sections",
    )
    return _return_ok(s2, _bulletize([f"Added {added_n} section(s).", f"Now {after_n} section(s) total."]))

@tool("slides_remove_sections", args_schema=SlidesRemoveSectionsInput, description="Remove sections by match or from end.")
async def slides_remove_sections(header: Optional[str] = None, count: int = 1, all_matches: bool = False) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _return_failed(None, _bulletize(["Missing session context."]))

    s = await _slides_fetch_latest_sync(int(session_id))
    if not s:
        return _return_ok(None, _bulletize(["No slides found.", "Share a topic to create a deck."]))

    before_ej = _ensure_editorjs(s.editorjs or {})
    blocks = _ej_blocks(before_ej)
    before_n = _count_sections(blocks)

    new_blocks = _remove_sections(blocks, header, count, all_matches)
    if new_blocks == blocks:
        return _return_ok(s, _bulletize(["No sections removed."]))

    ok, msg = _clamp_section_bounds(new_blocks)
    if not ok:
        return _return_failed(s, _bulletize(["Not applied.", msg]))

    after_n = _count_sections(new_blocks)
    removed_n = max(0, before_n - after_n)

    new_ej = {"time": _now_ms(), "version": "2.x", "blocks": new_blocks}
    s2, _ = await _slides_upsert_sync(
        int(session_id),
        title=_h_text(new_blocks[0]) if new_blocks else s.title,
        summary=s.summary,
        editorjs=new_ej,
        updated_by="tool:slides_remove_sections",
    )
    return _return_ok(s2, _bulletize([f"Removed {removed_n} section(s).", f"Now {after_n} section(s) total."]))

@tool("slides_edit", args_schema=SlidesEditInput, description="Replace whole EditorJS or apply targeted section edits.")
async def slides_edit(
    replace_editorjs: Optional[EditorJS | List[Dict[str, Any]]] = None,
    new_title: Optional[str] = None,
    new_summary: Optional[str] = None,
    section_edits: List[SlidesEditInput.SectionEdit] = [],
) -> Dict[str, Any]:

    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _return_failed(None, _bulletize(["Missing session context."]))

    s = await _slides_fetch_latest_sync(int(session_id))
    if not s:
        return _return_ok(None, _bulletize(["No slides found.", "Share a topic to create a deck."]))

    before_ej = _ensure_editorjs(s.editorjs or {})
    blocks = _ej_blocks(before_ej)

    if replace_editorjs is not None:
        candidate = _ensure_editorjs(replace_editorjs)
        ok, msg = _clamp_section_bounds(_ej_blocks(candidate))
        if not ok:
            return _return_failed(s, _bulletize(["Not applied.", msg]))
        new_ej = candidate
        new_title_final = _h_text(_ej_blocks(candidate)[0]) if _ej_blocks(candidate) else (new_title or s.title)
        new_summary_final = new_summary if new_summary is not None else s.summary
    else:
        if new_title is not None and blocks:
            blocks[0] = {"type": "header", "data": {"text": _strip_ctrl(new_title), "level": 2}}
        if new_summary is not None:
            if len(blocks) >= 2 and (blocks[1].get("type") or "").lower() == "paragraph":
                blocks[1] = {"type": "paragraph", "data": {"text": _strip_ctrl(new_summary)}}
            else:
                blocks = blocks[:1] + [{"type": "paragraph", "data": {"text": _strip_ctrl(new_summary)}}] + blocks[1:]

        for ed in section_edits or []:
            blocks = _apply_section_edit(blocks, ed)

        ok, msg = _clamp_section_bounds(blocks)
        if not ok:
            return _return_failed(s, _bulletize(["Not applied.", msg]))
        new_ej = {"time": _now_ms(), "version": "2.x", "blocks": blocks}
        new_title_final = _h_text(blocks[0]) if blocks else (new_title or s.title)
        new_summary_final = new_summary if new_summary is not None else s.summary

    new_ej = _ensure_editorjs(new_ej)
    diff_lines = _summarize_editorjs_diff(before_ej, new_ej)

    s2, _ = await _slides_upsert_sync(
        int(session_id),
        title=new_title_final or s.title,
        summary=new_summary_final or "",
        editorjs=new_ej,
        updated_by="tool:slides_edit",
    )
    return _return_ok(s2, _bulletize(["Edited slides."] + diff_lines))
