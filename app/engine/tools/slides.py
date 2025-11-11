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
    DEFAULT_TEXT_MODEL,
    MAX_SOURCES_PER_ANALYSIS,
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


# ======================================================================================
# Input Schemas
# ======================================================================================
class SlidesGenerateInput(BaseModel):
    prompt: Optional[str] = Field(None, description="Topic / natural language request for the deck. Creates a new deck.")
    title: Optional[str] = Field(None, description="Deck title (optional; inferred if omitted).")
    summary: Optional[str] = Field(None, description="Short abstract (optional).")
    editorjs: Optional[EditorJS | List[Dict[str, Any]]] = Field(
        None, description="(Ignored) Always generates fresh content."
    )
    context: Optional[str] = Field(None, description="Additional context for the deck (optional).")
    # Upper bound for sections; final total slides = title + 3..9 sections.
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
    # Insert new sections AFTER the first section whose header contains this substring (case-insensitive).
    # If not found or omitted, append at the end.
    after: Optional[str] = Field(None)
    # If you pass a single SectionSpec and count > 1, it will duplicate that spec (header gets suffixes).
    count: int = Field(1, ge=1, le=20, description="How many to add (used with one SectionSpec)")


class SlidesRemoveSectionsInput(BaseModel):
    # Remove by substring match (case-insensitive); if omitted, removes from the end.
    header: Optional[str] = None
    # Remove at most this many sections (will stop early if the floor would be violated).
    count: int = Field(1, ge=1, le=20)
    # If true with a header, will remove all matches up to `count`.
    all_matches: bool = False


class SlidesEditInput(BaseModel):
    # Optional direct replacement of the entire EditorJS document (must obey section bounds).
    replace_editorjs: Optional[EditorJS | List[Dict[str, Any]]] = None

    # Simple top-level edits
    new_title: Optional[str] = None
    new_summary: Optional[str] = None

    # Targeted section edits (apply in order)
    class SectionEdit(BaseModel):
        # Identify a section: either by zero-based index or by substring of header (case-insensitive)
        index: Optional[int] = None
        header_icontains: Optional[str] = None

        # Edits
        new_header: Optional[str] = None
        # Replace all bullets with this list
        new_bullets: Optional[List[str]] = None
        # Append bullets
        append_bullets: Optional[List[str]] = None
        # Remove bullets by index positions
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


def _is_h2_title(block: Dict[str, Any]) -> bool:
    return _is_header(block, 2)


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
    """
    Heuristic grouping: H3 header + (optional) immediate paragraph/list after it.
    """
    j = start_idx
    if j + 1 < len(blocks) and (blocks[j + 1].get("type") or "").lower() in {"list", "paragraph"}:
        j += 1
    return j


def _mk_section_blocks(header: str, bullets: List[str]) -> List[Dict[str, Any]]:
    header = _strip_ctrl(header or "Section")
    items = [_strip_ctrl(x) for x in (bullets or []) if str(x).strip()][:MAX_BULLETS_PER_SECTION]
    if not items:
        items = ["Key point 1", "Key point 2", "Key point 3"]
    return [
        {"type": "header", "data": {"text": header, "level": 3}},
        {"type": "list", "data": {"style": "unordered", "items": items}},
    ]


def _ensure_editorjs(ej_like: EditorJS | List[Dict[str, Any]] | Dict[str, Any]) -> Dict[str, Any]:
    """
    Force strict EditorJS format {time,version,blocks}, and ensure exactly one H2 top header.
    """
    ej = normalize_editorjs(ej_like) if isinstance(ej_like, (dict, list)) else {}
    blocks = _ensure_single_top_header(_ej_blocks(ej), None)
    return {"time": _now_ms(), "version": "2.x", "blocks": blocks}


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
            lvl = int(d.get("level", 2) or 2)
            txt = _strip_ctrl(d.get("text") or "")
            if lvl <= 2:
                h2.append(txt)
            else:
                h3.append(txt)
        elif t == "paragraph":
            paragraphs += 1
        elif t == "list":
            items = d.get("items") or []
            list_items += len([x for x in items if str(x).strip()])
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
        # Ensure strict EditorJS and single H2
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
# Formatting
# ======================================================================================
async def _format_slides_response(s: Slides) -> Dict[str, Any]:
    # Ensure we return strict EditorJS again (defensive)
    ej = _ensure_editorjs(s.editorjs or {})
    return {
        "version": s.version,
        "title": s.title or "",
        "summary": s.summary or "",
        "editorjs": ej,
        "updated_at": s.updated_at.isoformat(),
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
            bullets = [_strip_ctrl(str(x)) for x in (s.get("bullets") or []) if str(x).strip()][:MAX_BULLETS_PER_SECTION]
            blocks.append({"type": "header", "data": {"text": hdr, "level": 3}})
            blocks.append({"type": "list", "data": {"style": "unordered", "items": bullets}})

        ej = _ensure_editorjs({"time": _now_ms(), "version": "2.x", "blocks": blocks})
        return ej or minimal_editorjs(title, summary)
    except Exception:
        # deterministic fallback (title + 3 sections)
        title = (prompt or "Untitled Deck").strip()[:140] or "Untitled Deck"
        sections = _fallback_sections_from_text(context, MIN_SECTION_SLIDES)
        blocks = [{"type": "header", "data": {"text": _strip_ctrl(title), "level": 2}}]
        for s in sections:
            blocks.append({"type": "header", "data": {"text": _strip_ctrl(s["header"]), "level": 3}})
            blocks.append({"type": "list", "data": {"style": "unordered", "items": [_strip_ctrl(x) for x in s["bullets"][:MAX_BULLETS_PER_SECTION]]}})
        return {"time": _now_ms(), "version": "2.x", "blocks": blocks}


# ======================================================================================
# Tools
# ======================================================================================
def _standard_return(status: str, summary: str, slides: Optional[Slides]) -> Dict[str, Any]:
    data = {}
    if slides:
        data = {"slides": {"version": slides.version, "title": slides.title or "", "summary": slides.summary or "", "editorjs": _ensure_editorjs(slides.editorjs or {}), "updated_at": slides.updated_at.isoformat()}}
    return {"status": status, "summary": summary, "data": data}


@tool(
    "slides_generate_or_update",
    args_schema=SlidesGenerateInput,
    description="Create a brand-new deck (title + 3..9 sections). Rotates previous into SlidesRevision. Returns {status, summary, data.slides}."
)
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
        return _standard_return("failed", "Missing session context.", None)

    prompt_clean = _strip_ctrl((prompt or "").strip())
    title_clean = _strip_ctrl((title or "").strip())
    summary_clean = _strip_ctrl((summary or "").strip())
    context_clean = _strip_ctrl((context or "").strip())

    seed = title_clean or prompt_clean or "Untitled Deck"
    ej = await _llm_outline_to_editorjs(prompt=seed, context=context_clean, max_sections=max_sections)

    # Keep H2 stable and strict EJ
    if ej and ej.get("blocks"):
        ej = _ensure_editorjs({"time": _now_ms(), "version": "2.x", "blocks": _ensure_single_top_header(ej["blocks"], title_clean or None)})
    else:
        ej = minimal_editorjs(title_clean or (prompt_clean[:80] if prompt_clean else "Untitled Deck"), summary_clean or "")

    # Bounds check
    ok, reason = _clamp_section_bounds(_ej_blocks(ej))
    if not ok:
        return _standard_return("failed", f"Cannot generate deck: {reason}", None)

    try:
        s, before_note = await _slides_upsert_sync(
            session_id,
            title=(title_clean or (_ej_blocks(ej)[0]["data"]["text"] if _ej_blocks(ej) else "Untitled Deck")),
            summary=summary_clean or "",
            editorjs=ej,
            updated_by="tool:slides_generate_or_update:new_deck",
        )
        latest, prev = await _slides_get_revision_sync(session_id, None)
        changes: List[str] = []
        if prev:
            try:
                changes = _summarize_editorjs_diff(prev.editorjs or {}, latest.editorjs or {})
            except Exception:
                changes = []
        after_note = f"title={s.title!r}, v={s.version}"
        llm_summary = await _llm_summarize_change(
            title=s.title or "Deck",
            before_note=(f"title={getattr(prev, 'title', '')!r}, v={getattr(prev, 'version', 0)}") if prev else "(none)",
            after_note=after_note,
        )
        if prev and changes:
            summary_out = f"{llm_summary}\n- " + "\n- ".join(changes[:6])
        elif prev:
            summary_out = f"{llm_summary}"
        else:
            summary_out = f"Created new deck v{s.version} titled “{s.title or 'Untitled Deck'}”."
        return _standard_return("ok", summary_out, s)
    except Exception as e:
        log.exception("[slides_generate_or_update] persist error")
        return _standard_return("failed", f"Failed to save slides: {e}", None)


@tool(
    "slides_fetch_latest",
    args_schema=FetchSlidesInput,
    description="Fetch the latest slides. Returns {status, summary, data.slides}."
)
async def slides_fetch_latest() -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _standard_return("failed", "Missing session context.", None)
    try:
        s = await _slides_fetch_latest_sync(session_id)
    except Exception as e:
        log.exception("[slides_fetch_latest] db error")
        return _standard_return("failed", f"DB error: {e}", None)
    if not s:
        return _standard_return("not_found", "No slide deck exists yet.", None)
    return _standard_return("ok", f"Fetched deck v{s.version} titled “{s.title or 'Untitled Deck'}”.", s)


@tool(
    "slides_list_versions",
    args_schema=SlidesListVersionsInput,
    description="List recent slide versions; also returns latest deck. Returns {status, summary, data:{slides,versions}}."
)
async def slides_list_versions(limit: int = 10) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "summary": "Missing session context.", "data": {}}
    rows = await _slides_list_revisions_sync(session_id, limit=limit)
    latest = await _slides_fetch_latest_sync(session_id)
    if not rows:
        return {"status": "not_found", "summary": "No versions found.", "data": {"slides": (await _format_slides_response(latest)) if latest else None, "versions": []}}
    titles = ", ".join([f"v{r['version']}:{r['title'] or 'Untitled'}" for r in rows[:5]])
    return {"status": "ok", "summary": f"Found {len(rows)} version(s): {titles}.", "data": {"slides": await _format_slides_response(latest) if latest else None, "versions": rows}}


@tool(
    "slides_diff_latest",
    args_schema=SlidesDiffInput,
    description="Summarize differences between latest slides and a previous version (or previous by default). Returns {status, summary, data:{slides,from_version,to_version,changes}}."
)
async def slides_diff_latest(compare_to_version: Optional[int] = None) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "summary": "Missing session context.", "data": {}}
    try:
        latest = await _slides_fetch_latest_sync(session_id)
        if not latest:
            return {"status": "not_found", "summary": "No slide deck exists yet.", "data": {}}
        if compare_to_version is None:
            _, prev = await _slides_get_revision_sync(session_id, None)
        else:
            prev, _ = await _slides_get_revision_sync(session_id, compare_to_version)
        if not prev:
            return {"status": "not_found", "summary": "No previous version to compare against.", "data": {"slides": await _format_slides_response(latest)}}
        changes = _summarize_editorjs_diff(prev.editorjs or {}, latest.editorjs or {})
        summary_out = f"Compared v{latest.version} to v{prev.version}.\n- " + "\n- ".join(changes[:6])
        return {"status": "ok", "summary": summary_out, "data": {"slides": await _format_slides_response(latest), "from_version": prev.version, "to_version": latest.version, "changes": changes}}
    except Exception as e:
        log.exception("[slides_diff_latest] error")
        return {"status": "failed", "summary": f"Failed to diff slides: {e}", "data": {}}


@tool(
    "slides_revert",
    args_schema=SlidesRevertInput,
    description="Revert slides to a specific version (or previous if not provided). Returns {status, summary, data.slides}."
)
async def slides_revert(version: Optional[int] = None) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _standard_return("failed", "Missing session context.", None)
    try:
        s, before_note, target_version = await _slides_revert_sync(session_id, version)
        after_note = f"title={s.title!r}, v={s.version}"
        summary_out = await _llm_summarize_change(
            title=s.title or "Deck",
            before_note=before_note,
            after_note=f"Reverted to v{target_version}; now {after_note}",
        )
        return _standard_return("ok", summary_out, s)
    except ValueError as ve:
        return _standard_return("failed", str(ve), None)
    except Exception as e:
        log.exception("[slides_revert] db error")
        return _standard_return("failed", f"Failed to revert: {e}", None)


# === Add sections =====================================================================
@tool(
    "slides_add_sections",
    args_schema=SlidesAddSectionsInput,
    description="Add 1..N sections (H3 + bullets). Enforces 3..9 sections. Returns {status, summary, data.slides}."
)
async def slides_add_sections(sections: List[SectionSpec] = [], after: Optional[str] = None, count: int = 1) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _standard_return("failed", "Missing session context.", None)

    s = await _slides_fetch_latest_sync(session_id)
    if not s:
        return _standard_return("not_found", "No slide deck exists yet.", None)

    old_ej = _ensure_editorjs(s.editorjs or {})
    blocks = _ej_blocks(old_ej)
    current = _count_sections(blocks)
    if current >= MAX_SECTION_SLIDES:
        msg = (
            f"Section limit reached ({MAX_SECTION_SLIDES}). "
            "I can remove a section and then add a new one—would you like me to remove one? "
            "Specify a header substring or index."
        )
        return _standard_return("blocked", msg, s)

    insert_at = len(blocks)
    if after:
        idx = _find_section_by_icontains(blocks, after)
        if idx is not None:
            insert_at = _end_of_section(blocks, idx) + 1

    to_add: List[SectionSpec] = list(sections or [])
    if not to_add and count > 0:
        # Nothing to add; default to a single blank section spec repeated
        to_add = [SectionSpec(header="New Section", bullets=[])]
    if len(to_add) == 1 and count > 1:
        base = to_add[0]
        for i in range(2, count + 1):
            to_add.append(SectionSpec(header=f"{base.header} {i}", bullets=base.bullets))

    remaining = MAX_SECTION_SLIDES - current
    if len(to_add) > remaining:
        # Partial add up to the remaining capacity
        kept = to_add[:remaining]
        dropped = len(to_add) - len(kept)
        to_add = kept
        partial_note = f"Added {len(kept)} section(s); {dropped} not added due to the {MAX_SECTION_SLIDES}-section cap."
    else:
        partial_note = ""

    for spec in to_add:
        new = _mk_section_blocks(spec.header, spec.bullets)
        blocks[insert_at:insert_at] = new
        insert_at += len(new)

    # Final bounds check
    ok, reason = _clamp_section_bounds(blocks)
    if not ok:
        return _standard_return("failed", f"Cannot add sections: {reason}", s)

    updated_ej = _ensure_editorjs({"time": _now_ms(), "version": "2.x", "blocks": _ensure_single_top_header(blocks, s.title or None)})
    s2, before_note = await _slides_upsert_sync(
        session_id,
        title=s.title or "Untitled Deck",
        summary=s.summary or "",
        editorjs=updated_ej,
        updated_by="tool:slides_add_sections",
    )

    changes = _summarize_editorjs_diff(old_ej or {}, s2.editorjs or {})
    llm_note = await _llm_summarize_change(
        title=s2.title or "Deck",
        before_note=before_note,
        after_note=f"Added {len(to_add)} section(s); now title={s2.title!r}, v={s2.version}",
    )
    summary_out = (llm_note + (("\n" + partial_note) if partial_note else "") + ("\n- " + "\n- ".join(changes[:6]) if changes else "")).strip()
    return _standard_return("ok", summary_out, s2)


# === Remove sections ==================================================================
@tool(
    "slides_remove_sections",
    args_schema=SlidesRemoveSectionsInput,
    description="Remove up to N sections by header substring (or from the end). Enforces floor of 3 sections. Returns {status, summary, data.slides}."
)
async def slides_remove_sections(header: Optional[str] = None, count: int = 1, all_matches: bool = False) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _standard_return("failed", "Missing session context.", None)

    s = await _slides_fetch_latest_sync(session_id)
    if not s:
        return _standard_return("not_found", "No slide deck exists yet.", None)

    old_ej = _ensure_editorjs(s.editorjs or {})
    blocks = _ej_blocks(old_ej)
    section_idxs = _find_section_indices(blocks)
    if not section_idxs:
        return _standard_return("not_found", "No sections to remove.", s)

    removed_titles: List[str] = []
    removed = 0

    def _remove_at(blks: List[Dict[str, Any]], idx: int) -> List[Dict[str, Any]]:
        end = _end_of_section(blks, idx)
        removed_titles.append(_h_text(blks[idx]))
        return blks[:idx] + blks[end + 1:]

    while removed < count:
        current_n = _count_sections(blocks)
        if current_n <= MIN_SECTION_SLIDES:
            break  # floor reached
        if header:
            idx = _find_section_by_icontains(blocks, header)
            if idx is None:
                break
            blocks = _remove_at(blocks, idx)
            removed += 1
            if not all_matches:
                break
        else:
            idxs = _find_section_indices(blocks)
            if not idxs:
                break
            blocks = _remove_at(blocks, idxs[-1])
            removed += 1

    if removed == 0:
        msg = "No matching section found to remove." if header else f"Cannot remove: deck is at the minimum of {MIN_SECTION_SLIDES} sections."
        return _standard_return("blocked", msg, s)

    ok, reason = _clamp_section_bounds(blocks)
    if not ok:
        return _standard_return("failed", f"Cannot remove sections: {reason}", s)

    updated_ej = _ensure_editorjs({"time": _now_ms(), "version": "2.x", "blocks": _ensure_single_top_header(blocks, s.title or None)})
    s2, before_note = await _slides_upsert_sync(
        session_id,
        title=s.title or "Untitled Deck",
        summary=s.summary or "",
        editorjs=updated_ej,
        updated_by="tool:slides_remove_sections",
    )

    changes = _summarize_editorjs_diff(old_ej or {}, s2.editorjs or {})
    removed_label = ", ".join([f"“{t}”" for t in removed_titles[:3]]) + ("…" if len(removed_titles) > 3 else "")
    llm_note = await _llm_summarize_change(
        title=s2.title or "Deck",
        before_note=before_note,
        after_note=f"Removed {removed} section(s) {removed_label}; now title={s2.title!r}, v={s2.version}",
    )
    summary_out = (llm_note + ("\n- " + "\n- ".join(changes[:6]) if changes else "")).strip()
    return _standard_return("ok", summary_out, s2)


# === Edit deck (title/summary/sections/whole blocks) ==================================
@tool(
    "slides_edit",
    args_schema=SlidesEditInput,
    description=(
        "Edit title/summary and/or specific sections (rename/replace bullets/append/remove bullets), "
        "or replace the entire editorjs document. Enforces 3..9 sections. Returns {status, summary, data.slides}."
    ),
)
async def slides_edit(
    replace_editorjs: Optional[EditorJS | List[Dict[str, Any]]] = None,
    new_title: Optional[str] = None,
    new_summary: Optional[str] = None,
    section_edits: List[SlidesEditInput.SectionEdit] = [],
) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return _standard_return("failed", "Missing session context.", None)

    s = await _slides_fetch_latest_sync(session_id)
    if not s:
        return _standard_return("not_found", "No slide deck exists yet.", None)

    if replace_editorjs is not None:
        ej = _ensure_editorjs(replace_editorjs)
        blocks = _ej_blocks(ej)
    else:
        ej = _ensure_editorjs(s.editorjs or {})
        blocks = _ej_blocks(ej)

    # Title/summary edits
    title_out = _strip_ctrl(new_title) if (new_title is not None) else (s.title or "")
    summary_out = _strip_ctrl(new_summary) if (new_summary is not None) else (s.summary or "")

    # Ensure first block is H2 with desired title
    blocks = _ensure_single_top_header(blocks, title_out or None)

    # Targeted section edits
    for edit in section_edits or []:
        # Locate target
        idx = None
        if edit.index is not None:
            sec_idxs = _find_section_indices(blocks)
            if 0 <= edit.index < len(sec_idxs):
                idx = sec_idxs[edit.index]
        elif edit.header_icontains:
            idx = _find_section_by_icontains(blocks, edit.header_icontains)

        if idx is None:
            # skip silently (could also summarize that a target wasn't found)
            continue

        # Rename header
        if edit.new_header is not None:
            if _is_h3_section_header(blocks[idx]):
                blocks[idx] = {"type": "header", "data": {"text": _strip_ctrl(edit.new_header), "level": 3}}

        # Operate on the block after header if it's a list
        end = _end_of_section(blocks, idx)
        if idx + 1 <= end and (blocks[idx + 1].get("type") or "").lower() == "list":
            items = [(x if isinstance(x, str) else str(x)) for x in (blocks[idx + 1].get("data", {}).get("items") or [])]
        else:
            items = []

        # Replace bullets
        if edit.new_bullets is not None:
            items = [_strip_ctrl(x) for x in edit.new_bullets][:MAX_BULLETS_PER_SECTION]

        # Append bullets
        if edit.append_bullets:
            for b in edit.append_bullets:
                if len(items) >= MAX_BULLETS_PER_SECTION:
                    break
                if str(b).strip():
                    items.append(_strip_ctrl(b))

        # Remove bullets by indices
        if edit.remove_bullets_indices:
            keep = [x for i, x in enumerate(items) if i not in set(edit.remove_bullets_indices)]
            items = keep[:MAX_BULLETS_PER_SECTION]

        # Write back bullets (ensure there's a list block)
        if idx + 1 <= end and (blocks[idx + 1].get("type") or "").lower() == "list":
            blocks[idx + 1] = {"type": "list", "data": {"style": "unordered", "items": items[:MAX_BULLETS_PER_SECTION]}}
        else:
            # insert a new list right after header
            blocks = blocks[:idx + 1] + [{"type": "list", "data": {"style": "unordered", "items": items[:MAX_BULLETS_PER_SECTION]}}] + blocks[idx + 1:]

    # Bounds check (sections count must be within 3..9)
    ok, reason = _clamp_section_bounds(blocks)
    if not ok:
        return _standard_return("failed", f"Edit would violate bounds: {reason}", s)

    updated_ej = _ensure_editorjs({"time": _now_ms(), "version": "2.x", "blocks": blocks})
    s2, before_note = await _slides_upsert_sync(
        session_id,
        title=title_out or "Untitled Deck",
        summary=summary_out or "",
        editorjs=updated_ej,
        updated_by="tool:slides_edit",
    )

    changes = _summarize_editorjs_diff(ej or {}, s2.editorjs or {})
    after_note = f"title={s2.title!r}, v={s2.version}"
    llm_note = await _llm_summarize_change(title=s2.title or "Deck", before_note=before_note, after_note=after_note)
    summary_out = (llm_note + ("\n- " + "\n- ".join(changes[:6]) if changes else "")).strip()
    return _standard_return("ok", summary_out, s2)
