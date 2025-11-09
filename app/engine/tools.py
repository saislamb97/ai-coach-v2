from __future__ import annotations

import json
import logging
import os
import re
import time
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

from asgiref.sync import sync_to_async
from django.core.files.storage import default_storage
from django.db import models
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from memory.models import (
    Slides,
    SlidesRevision,
    Knowledge,
    Session,
    _strip_ctrl,
)

# ---- token truncation (engine/tokens.py) ------------------------------------
try:
    from engine.tokens import truncate_string_to_token_limit
except Exception:  # fallback
    def truncate_string_to_token_limit(content: str, max_token_limit: int | None = None) -> str:
        if not content:
            return ""
        limit = max_token_limit or 10000
        b = content.encode("utf-8", errors="ignore")
        return b[:limit].decode("utf-8", errors="ignore")

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Config & Context
# --------------------------------------------------------------------------------------
DB_PRIMARY = os.getenv("DB_PRIMARY_ALIAS", "default")
DEFAULT_TEXT_MODEL = os.getenv("RAG_TEXT_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_SECS = 25

MAX_EDITORJS_BLOCKS = 120
MAX_EXTRACTED_CHARS_PER_DOC = int(os.getenv("DOCS_MAX_CHARS", "16000"))
TOKENS_PER_DOC_DEFAULT = int(os.getenv("DOCS_TOKENS_PER_DOC", "8000"))
MAX_SOURCES_PER_ANALYSIS = 5

# Slides: require 4–6 total slides (title + 3..5 sections)
MIN_SECTION_SLIDES = 3
MAX_SECTION_SLIDES = 5

_TOOL_CTX: ContextVar[Dict[str, Any]] = ContextVar("TOOL_CTX", default={})

def set_tool_context(ctx: Dict[str, Any]) -> None:
    _TOOL_CTX.set(dict(ctx or {}))

def get_tool_context() -> Dict[str, Any]:
    return dict(_TOOL_CTX.get() or {})

def _now_ms() -> int:
    return int(time.time() * 1000)

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _slug(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s

def _tokenize_query(q: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]{2,}", (q or "").lower()) if t]

def _name_similarity(q: str, a: Knowledge) -> float:
    import difflib
    qn = re.sub(r"[^a-z0-9]+", "", (q or "").lower())
    cand = re.sub(r"[^a-z0-9]+", "", ((a.title or "") + " " + (a.file_name or "")).lower())
    return difflib.SequenceMatcher(a=qn, b=cand).ratio()

def _search_term_hits(q: str, a: Knowledge) -> int:
    toks = set(_tokenize_query(q))
    try:
        terms = set((a.search_terms or []))
    except Exception:
        terms = set()
    return len(toks & terms)

def _rank_assets(assets: List[Knowledge], query: str) -> List[Knowledge]:
    return sorted(
        assets,
        key=lambda k: (_name_similarity(query, k) + 0.15 * _search_term_hits(query, k)),
        reverse=True,
    )

def _llm(model: Optional[str] = None, temperature: float = 0.2, timeout: int = LLM_TIMEOUT_SECS) -> ChatOpenAI:
    return ChatOpenAI(model=(model or DEFAULT_TEXT_MODEL), temperature=temperature, timeout=timeout)

# -----------------------------
# Minimal Editor.js helpers
# -----------------------------
EditorJS = Dict[str, Any]
ALLOWED_BLOCK_TYPES = {"header", "paragraph", "list"}

def _coerce_blocks(blocks_any: Any) -> List[Dict[str, Any]]:
    if not isinstance(blocks_any, list):
        return []
    out: List[Dict[str, Any]] = []
    for b in blocks_any:
        if not isinstance(b, dict):
            continue
        t = (b.get("type") or "").lower()
        d = b.get("data") if isinstance(b.get("data"), dict) else None
        if t not in ALLOWED_BLOCK_TYPES or d is None:
            continue
        if t == "header":
            try:
                level = int(d.get("level", 2))
            except Exception:
                level = 2
            if level not in (1, 2, 3):  # normalize to 2/3
                level = 2
            d = {**d, "level": level, "text": _strip_ctrl(d.get("text", ""))}
        elif t == "paragraph":
            d = {**d, "text": _strip_ctrl(d.get("text", ""))}
        elif t == "list":
            items = d.get("items")
            if not isinstance(items, list):
                items = []
            d = {**d, "style": "unordered", "items": [_strip_ctrl(str(x)) for x in items if str(x).strip()]}
        out.append({"type": t, "data": d})
        if len(out) >= MAX_EDITORJS_BLOCKS:
            break
    return out

def normalize_editorjs(obj: Any) -> Optional[EditorJS]:
    if isinstance(obj, list):
        blocks = _coerce_blocks(obj)
        return {"time": _now_ms(), "blocks": blocks, "version": "2.x"} if blocks else None
    if isinstance(obj, dict) and "blocks" in obj:
        blocks = _coerce_blocks(obj.get("blocks"))
        if not blocks:
            return None
        time_val = obj.get("time")
        ver = obj.get("version") or "2.x"
        try:
            time_val = int(time_val) if time_val is not None else _now_ms()
        except Exception:
            time_val = _now_ms()
        return {"time": time_val, "blocks": blocks, "version": str(ver)}
    return None

def minimal_editorjs(title: str, summary: str = "") -> EditorJS:
    title = _strip_ctrl((title or "Untitled Deck").strip()[:140])
    summary = _strip_ctrl((summary or "").strip()[:300])
    blocks: List[Dict[str, Any]] = [{"type": "header", "data": {"text": title, "level": 2}}]
    if summary:
        blocks.append({"type": "paragraph", "data": {"text": summary}})
    return {"time": _now_ms(), "version": "2.x", "blocks": blocks}

def _ensure_single_top_header(blocks: List[Dict[str, Any]], title: Optional[str]) -> List[Dict[str, Any]]:
    if not title:
        return blocks
    title = _strip_ctrl(title)
    out: List[Dict[str, Any]] = []
    put_title = False
    for b in blocks:
        if (b.get("type") or "").lower() == "header" and not put_title:
            out.append({"type": "header", "data": {"text": title, "level": 2}})
            put_title = True
        else:
            out.append(b)
    if not put_title:
        out = [{"type": "header", "data": {"text": title, "level": 2}}] + out
    return out

# --------------------------------------------------------------------------------------
# Async-safe DB helpers
# --------------------------------------------------------------------------------------
@sync_to_async(thread_sensitive=True)
def _infer_user_from_session(session_id: int) -> Optional[int]:
    sess = Session.objects.using(DB_PRIMARY).filter(id=session_id).only("user_id").first()
    return getattr(sess, "user_id", None) if sess else None

async def _ensure_agent_user_from_ctx(ctx: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    agent_id = ctx.get("agent_id")
    user_id = ctx.get("user_id") or ctx.get("owner_user_id") or ctx.get("user")
    if not user_id:
        sess_id = ctx.get("session_id")
        if sess_id:
            try:
                user_id = await _infer_user_from_session(int(sess_id))
            except Exception:
                user_id = None
    return (int(agent_id) if agent_id else None, int(user_id) if user_id else None)

@sync_to_async(thread_sensitive=True)
def _db_list_assets(*, agent_id: int, user_id: int, query: str = "", limit: int = 20) -> List[Knowledge]:
    qs = (
        Knowledge.objects.using(DB_PRIMARY)
        .filter(agent_id=agent_id, user_id=user_id)
        .order_by("-created_at")
    )
    if query:
        q_text = query.strip()
        slug = _slug(q_text)
        toks = _tokenize_query(q_text)
        q_obj = (
            models.Q(file_name__icontains=q_text)
            | models.Q(title__icontains=q_text)
            | models.Q(normalized_name__icontains=slug)
            | models.Q(excerpt__icontains=q_text)
        )
        for t in toks:
            q_obj |= models.Q(search_terms__contains=[t])
        qs = qs.filter(q_obj)
    return list(qs[: max(1, min(200, limit))])

# Slides sync wrappers (ORM in threads)
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
        s.rotate_and_update(title=title or "Untitled Deck", summary=summary or "", editorjs=editorjs, updated_by=updated_by)
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
        s = (
            Slides.objects.using(DB_PRIMARY)
            .filter(session_id=session_id)
            .first()
        )
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
        s = (
            SlidesRevision.objects.using(DB_PRIMARY)
            .filter(session_id=session_id, version=version)
            .first()
        )
        return s, None

# --------------------------------------------------------------------------------------
# Extraction helpers for Documents
# --------------------------------------------------------------------------------------
def _truncate_text_by_tokens(text: str, tokens: int = TOKENS_PER_DOC_DEFAULT) -> str:
    try:
        return truncate_string_to_token_limit(text, max_token_limit=tokens)
    except Exception:
        b = text.encode("utf-8", errors="ignore")
        return b[:tokens].decode("utf-8", errors="ignore")

def _extract_asset_content(
    asset: Knowledge,
    limit_chars: int = MAX_EXTRACTED_CHARS_PER_DOC,
    token_cap: int = TOKENS_PER_DOC_DEFAULT,
) -> str:
    from tempfile import NamedTemporaryFile
    try:
        from memory.extract import extract_rich
    except Exception:
        from memory.extract import extract_rich  # noqa

    storage_path = asset.file.name
    blob: bytes = b""
    try:
        with default_storage.open(storage_path, "rb") as f:
            blob = f.read()
    except Exception:
        return ""

    ext = os.path.splitext(asset.file_name or storage_path)[1].lower() or ".bin"
    tmp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(blob)
            tmp_path = tmp.name

        try:
            result = extract_rich(tmp_path)
            text = _strip_ctrl(result.text or "")
        except Exception:
            try:
                text = _strip_ctrl(blob.decode("utf-8", errors="replace"))
            except Exception:
                text = _strip_ctrl(blob.decode("latin-1", errors="replace"))

        if limit_chars and len(text) > limit_chars:
            text = text[:limit_chars]
        return _truncate_text_by_tokens(text, tokens=token_cap)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

# --------------------------------------------------------------------------------------
# Slide outline helpers (ensure 4–6 slides total)
# --------------------------------------------------------------------------------------
def _fallback_sections_from_text(context: str, need_n: int) -> List[Dict[str, Any]]:
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
    return [{"header": h, "bullets": bs[:6]} for (h, bs) in seeds[:need_n]]

async def _llm_summarize_change(*, title: str, before_note: str, after_note: str, model: Optional[str] = None) -> str:
    llm = _llm(model=model, temperature=0.2)
    sys = "Summarize in 2–4 short bullets what changed and what the result is. Be concrete."
    user = f"Title: {title}\n\nBEFORE:\n{before_note.strip() or '(none)'}\n\nAFTER:\n{after_note.strip() or '(none)'}"
    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user[:4000])])
        return (getattr(msg, "content", "") or "").strip()[:800]
    except Exception:
        return f"Updated “{title}”. (Summary unavailable.)"

def _editorjs_stats(ej: Dict[str, Any]) -> Dict[str, Any]:
    blocks = (ej or {}).get("blocks") or []
    h2 = []
    h3 = []
    paragraphs = 0
    list_items = 0
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
    return {
        "h2": h2, "h3": h3,
        "paragraphs": paragraphs,
        "list_items": list_items,
        "blocks": len(blocks),
    }

def _summarize_editorjs_diff(old_ej: Dict[str, Any], new_ej: Dict[str, Any]) -> List[str]:
    o = _editorjs_stats(old_ej or {})
    n = _editorjs_stats(new_ej or {})
    lines: List[str] = []
    if (o["h2"][:1] or [""])[0] != (n["h2"][:1] or [""])[0]:
        lines.append(f"Title changed: “{(o['h2'][:1] or [''])[0]}” → “{(n['h2'][:1] or [''])[0]}”.")
    # headers
    added_h3 = [h for h in n["h3"] if h and h not in o["h3"]]
    removed_h3 = [h for h in o["h3"] if h and h not in n["h3"]]
    if added_h3:
        lines.append(f"Added {len(added_h3)} section(s): " + ", ".join(added_h3[:3]) + ("…" if len(added_h3) > 3 else ""))
    if removed_h3:
        lines.append(f"Removed {len(removed_h3)} section(s): " + ", ".join(removed_h3[:3]) + ("…" if len(removed_h3) > 3 else ""))
    # bullets & blocks
    di = n["list_items"] - o["list_items"]
    if di:
        lines.append(("+" if di > 0 else "") + f"{di} bullet(s) total.")
    db = n["blocks"] - o["blocks"]
    if db:
        lines.append(("+" if db > 0 else "") + f"{db} blocks.")
    if not lines:
        lines.append("Minor text edits; structure unchanged.")
    return lines

async def _llm_outline_to_editorjs(
    *,
    prompt: str,
    context: str,
    max_sections: int = 6,
    model: Optional[str] = None,
) -> EditorJS:
    # clamp to 4–6 total slides => sections = 3..5
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
            bullets = [_strip_ctrl(str(x)) for x in (s.get("bullets") or []) if str(x).strip()] or ["Key point 1", "Key point 2", "Key point 3"]
            blocks.append({"type": "header", "data": {"text": hdr, "level": 3}})
            blocks.append({"type": "list", "data": {"style": "unordered", "items": bullets[:8]}})

        ej = normalize_editorjs({"time": _now_ms(), "version": "2.x", "blocks": blocks})
        return ej or minimal_editorjs(title, summary)
    except Exception:
        # deterministic fallback (title + 3 sections)
        title = (prompt or "Untitled Deck").strip()[:140] or "Untitled Deck"
        sections = _fallback_sections_from_text(context, MIN_SECTION_SLIDES)
        blocks = [{"type": "header", "data": {"text": _strip_ctrl(title), "level": 2}}]
        for s in sections:
            blocks.append({"type": "header", "data": {"text": _strip_ctrl(s["header"]), "level": 3}})
            blocks.append({"type": "list", "data": {"style": "unordered", "items": [_strip_ctrl(x) for x in s["bullets"][:6]]}})
        return {"time": _now_ms(), "version": "2.x", "blocks": blocks}

async def _llm_analyze_sources(*, question: str, sources: List[Dict[str, Any]], model: Optional[str] = None) -> Dict[str, Any]:
    llm = _llm(model=model or DEFAULT_TEXT_MODEL, temperature=0.2)
    sys = (
        "You are a precise analyst. Answer ONLY using the provided sources.\n"
        "Return STRICT JSON: {\"answer\":\"...\",\"bullets\":[\"...\"],\"citations\":[{\"title\":\"...\"}]}"
    )
    src_lines: List[str] = []
    for i, s in enumerate(sources, 1):
        title = _strip_ctrl(s.get("title") or s.get("file_name") or f"Source {i}")
        snippet = _strip_ctrl((s.get("content") or "")[:4000])
        src_lines.append(f"[{i}] {title}\n{snippet}\n")
    user = "SOURCES:\n\n" + "\n".join(src_lines) + f"\nQUESTION:\n{_strip_ctrl(question.strip())}\n"
    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user[:12000])])
        raw = (getattr(msg, "content", "") or "").strip()
        data = json.loads(raw if raw.startswith("{") else raw[raw.find("{"):raw.rfind("}") + 1])
        data["answer"] = _strip_ctrl((data.get("answer") or "").strip())
        data["bullets"] = [_strip_ctrl(b) for b in (data.get("bullets") or []) if isinstance(b, str) and b.strip()]
        cits = data.get("citations") or []
        data["citations"] = [{"title": _strip_ctrl((c or {}).get("title") or "")} for c in cits if (c or {}).get("title")]
        return data
    except Exception:
        return {"answer": "I don't have enough information in the provided sources.", "bullets": [], "citations": []}

# --------------------------------------------------------------------------------------
# Input Schemas
# --------------------------------------------------------------------------------------
class SlidesGenerateInput(BaseModel):
    prompt: Optional[str] = Field(None, description="Natural language request for the deck")
    title: Optional[str] = Field(None, description="Deck title (optional)")
    summary: Optional[str] = Field(None, description="Short abstract (optional)")
    editorjs: Optional[EditorJS | List[Dict[str, Any]]] = Field(None, description="Editor.js doc or blocks[]")
    context: Optional[str] = Field(None, description="Brief topic context (optional)")
    max_sections: int = Field(6, ge=4, le=16, description="Max sections (title + 3..5 slides enforced)")

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

class DocumentsListInput(BaseModel):
    query: Optional[str] = Field(None, description="Filter by name/title (fuzzy).")
    limit: int = Field(20, ge=1, le=200, description="Max number of documents to list")

class DocumentsFetchInput(BaseModel):
    query: str = Field(..., description="Title/file-name/keywords to fetch the closest document(s)")
    limit: int = Field(1, ge=1, le=5, description="How many docs to return")
    include_content: bool = Field(True, description="Include truncated extracted content")
    sample_chars_per_doc: int = Field(MAX_EXTRACTED_CHARS_PER_DOC, ge=1000, le=100000, description="Character cap per doc")
    tokens_per_doc: int = Field(TOKENS_PER_DOC_DEFAULT, ge=2000, le=14000, description="Token cap per doc")

class DocumentsAnalyzeInput(BaseModel):
    question: str = Field(..., description="Your question about the documents")
    search_query: Optional[str] = Field(None, description="Name/title filter; leave empty to analyze recent files")
    limit_files: int = Field(3, ge=1, le=MAX_SOURCES_PER_ANALYSIS, description="How many files to analyze")
    sample_chars_per_doc: int = Field(MAX_EXTRACTED_CHARS_PER_DOC, ge=2000, le=100000, description="Content char cap per doc")
    tokens_per_doc: int = Field(TOKENS_PER_DOC_DEFAULT, ge=2000, le=14000, description="Token cap per doc")
    make_slides: bool = Field(False, description="If true, also generate slides grounded on these sources")
    slides_max_sections: int = Field(6, ge=4, le=16, description="Max sections when creating slides")

class DocumentsSlidesInput(BaseModel):
    query: str = Field(..., description="Title/file-name/keywords to find doc(s) to turn into slides")
    limit_files: int = Field(3, ge=1, le=MAX_SOURCES_PER_ANALYSIS, description="Max docs to pull into slides")
    sample_chars_per_doc: int = Field(MAX_EXTRACTED_CHARS_PER_DOC, ge=2000, le=100000, description="Content char cap per doc")
    tokens_per_doc: int = Field(TOKENS_PER_DOC_DEFAULT, ge=2000, le=14000, description="Token cap per doc")
    slides_max_sections: int = Field(6, ge=4, le=16, description="Max sections in the deck")
    title_override: Optional[str] = Field(None, description="Optional deck title to force")

class EmotionInput(BaseModel):
    text: str = Field(..., description="Text to classify into a single dominant emotion.")

# --------------------------------------------------------------------------------------
# Slides format helper
# --------------------------------------------------------------------------------------
async def _format_slides_response(s: Slides) -> Dict[str, Any]:
    return {
        "version": s.version,
        "title": s.title or "",
        "summary": s.summary or "",
        "editorjs": s.editorjs or {},
        "updated_at": s.updated_at.isoformat(),
    }

# --------------------------------------------------------------------------------------
# Slides Tools
# --------------------------------------------------------------------------------------
@tool(
    "slides_generate_or_update",
    args_schema=SlidesGenerateInput,
    description="Create or update slides. Returns a summary and {version,title,summary,editorjs,updated_at}.",
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
        return {"status": "failed", "summary": "Missing session context.", "data": {}}

    prompt_clean = _strip_ctrl((prompt or "").strip())
    title_clean = _strip_ctrl((title or "").strip())
    summary_clean = _strip_ctrl((summary or "").strip())
    context_clean = _strip_ctrl((context or "").strip())
    ej = normalize_editorjs(editorjs) if editorjs is not None else None

    # Build outline if none provided
    if not (title_clean or summary_clean or (ej and ej.get("blocks"))):
        ej = await _llm_outline_to_editorjs(
            prompt=prompt_clean or "Untitled Deck",
            context=context_clean,
            max_sections=max_sections,
        )
        first_header = next((b for b in ej["blocks"] if b["type"] == "header"), None)
        if first_header:
            title_clean = title_clean or first_header["data"].get("text", "Untitled Deck")

    # Normalize and guarantee single top header
    if ej and ej.get("blocks"):
        ej = normalize_editorjs(
            {"time": _now_ms(), "version": "2.x", "blocks": _ensure_single_top_header(ej["blocks"], title_clean)}
        ) or ej
    if not ej:
        ej = minimal_editorjs(title_clean or (prompt_clean[:80] if prompt_clean else "Untitled Deck"), summary_clean or "")

    try:
        s, before_note = await _slides_upsert_sync(
            session_id, title=title_clean or "Untitled Deck", summary=summary_clean or "", editorjs=ej, updated_by="tool:slides"
        )
        data = {"slides": await _format_slides_response(s)}
        after_note = f"title={s.title!r}, v={s.version}"
        summary_out = await _llm_summarize_change(title=s.title or "Untitled Deck", before_note=before_note, after_note=after_note)
        return {"status": "ok", "summary": summary_out, "data": data}
    except Exception as e:
        log.exception("[slides_generate_or_update] persist error")
        return {"status": "failed", "summary": f"Failed to save slides: {e}", "data": {}}

@tool(
    "slides_fetch_latest",
    args_schema=FetchSlidesInput,
    description="Fetch the latest slides for this session. Returns {version,title,summary,editorjs,updated_at}.",
)
async def slides_fetch_latest() -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "summary": "Missing session context.", "data": {}}

    try:
        s = await _slides_fetch_latest_sync(session_id)
    except Exception as e:
        log.exception("[slides_fetch_latest] db error")
        return {"status": "failed", "summary": f"DB error: {e}", "data": {}}

    if not s:
        return {"status": "not_found", "summary": "No slide deck exists yet.", "data": {}}

    data = {"slides": await _format_slides_response(s)}
    summary_out = f"Fetched deck v{s.version} titled “{s.title or 'Untitled Deck'}”."
    return {"status": "ok", "summary": summary_out, "data": data}

@tool(
    "slides_list_versions",
    args_schema=SlidesListVersionsInput,
    description="List recent slide versions with title and timestamps.",
)
async def slides_list_versions(limit: int = 10) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "summary": "Missing session context.", "data": {}}
    rows = await _slides_list_revisions_sync(session_id, limit=limit)
    if not rows:
        return {"status": "not_found", "summary": "No versions found.", "data": {}}
    titles = ", ".join([f"v{r['version']}:{r['title'] or 'Untitled'}" for r in rows[:5]])
    return {"status": "ok", "summary": f"Found {len(rows)} version(s): {titles}.", "data": {"versions": rows}}

@tool(
    "slides_diff_latest",
    args_schema=SlidesDiffInput,
    description="Summarize differences between the latest slides and a previous version (or previous by default). Returns diff summary and the latest deck.",
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
            prev = SlidesRevision.objects.using(DB_PRIMARY).filter(session_id=session_id, version=compare_to_version).first()
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
    description="Revert slides to a specific version (or previous if not provided). Returns the current deck.",
)
async def slides_revert(version: Optional[int] = None) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "summary": "Missing session context.", "data": {}}
    try:
        s, before_note, target_version = await _slides_revert_sync(session_id, version)
        data = {"slides": await _format_slides_response(s)}
        after_note = f"title={s.title!r}, v={s.version}"
        summary_out = await _llm_summarize_change(
            title=s.title or "Deck",
            before_note=before_note,
            after_note=f"Reverted to v{target_version}; now {after_note}",
        )
        return {"status": "ok", "summary": summary_out, "data": data}
    except ValueError as ve:
        return {"status": "failed", "summary": str(ve), "data": {}}
    except Exception as e:
        log.exception("[slides_revert] db error")
        return {"status": "failed", "summary": f"Failed to revert: {e}", "data": {}}

# --------------------------------------------------------------------------------------
# Documents Tools  (renamed from knowledge_*)
# --------------------------------------------------------------------------------------
@tool(
    "documents_list",
    args_schema=DocumentsListInput,
    description="List documents for this agent/user. Returns title, file_name, updated_at, and meta.",
)
async def documents_list(query: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    ctx = get_tool_context()
    agent_id, user_id = await _ensure_agent_user_from_ctx(ctx)
    if not agent_id or not user_id:
        return {"status": "failed", "summary": "Missing agent or user context.", "data": {}}

    items = await _db_list_assets(agent_id=agent_id, user_id=user_id, query=(query or "").strip(), limit=limit)
    ranked = _rank_assets(items, query) if query else items

    assets = [
        {
            "title": (a.title or a.file_name),
            "file_name": a.file_name,
            "updated_at": a.updated_at.isoformat(),
            "meta": a.meta or {},
        }
        for a in ranked[:limit]
    ]
    if not assets:
        return {"status": "not_found", "summary": "No documents found.", "data": {"assets": []}}

    names = ", ".join(x["title"] for x in assets[:5])
    summary_out = f"Found {len(assets)} document(s): {names}."
    return {"status": "ok", "summary": summary_out, "data": {"assets": assets}}

@tool(
    "documents_fetch",
    args_schema=DocumentsFetchInput,
    description="Fetch the closest document(s) by fuzzy title/file/keywords and (optionally) include truncated extracted content.",
)
async def documents_fetch(
    query: str,
    limit: int = 1,
    include_content: bool = True,
    sample_chars_per_doc: int = MAX_EXTRACTED_CHARS_PER_DOC,
    tokens_per_doc: int = TOKENS_PER_DOC_DEFAULT,
) -> Dict[str, Any]:
    ctx = get_tool_context()
    agent_id, user_id = await _ensure_agent_user_from_ctx(ctx)
    if not agent_id or not user_id:
        return {"status": "failed", "summary": "Missing agent or user context.", "data": {}}

    items = await _db_list_assets(agent_id=agent_id, user_id=user_id, query=(query or "").strip(), limit=200)
    if not items:
        return {"status": "not_found", "summary": f"No documents found for “{query}”.", "data": {}}

    ranked = _rank_assets(items, query)[:limit]
    assets = []
    for a in ranked:
        payload = {
            "title": (a.title or a.file_name),
            "file_name": a.file_name,
            "updated_at": a.updated_at.isoformat(),
            "meta": a.meta or {},
        }
        if include_content:
            payload["content"] = _extract_asset_content(
                a, limit_chars=sample_chars_per_doc, token_cap=tokens_per_doc
            )
        assets.append(payload)

    names = ", ".join(x["title"] for x in assets)
    summary_out = f"Fetched {len(assets)} document(s): {names}."
    return {"status": "ok", "summary": summary_out, "data": {"assets": assets}}

@tool(
    "documents_analyze",
    args_schema=DocumentsAnalyzeInput,
    description=("Answer a question using uploaded documents. Returns analysis + sources. "
                 "Set make_slides=true to also generate slides grounded on these sources."),
)
async def documents_analyze(
    question: str,
    search_query: Optional[str] = None,
    limit_files: int = 3,
    sample_chars_per_doc: int = MAX_EXTRACTED_CHARS_PER_DOC,
    tokens_per_doc: int = TOKENS_PER_DOC_DEFAULT,
    make_slides: bool = False,
    slides_max_sections: int = 6,
) -> Dict[str, Any]:
    ctx = get_tool_context()
    agent_id, user_id = await _ensure_agent_user_from_ctx(ctx)
    session_id = ctx.get("session_id")
    if not agent_id or not user_id:
        return {"status": "failed", "summary": "Missing agent or user context.", "data": {}}

    # If no filter provided, analyze recent files (do NOT use the freeform question as a filter).
    derived_query = (search_query or "").strip()
    all_assets = await _db_list_assets(agent_id=agent_id, user_id=user_id, query=derived_query, limit=200)
    if not all_assets:
        return {"status": "not_found", "summary": "No matching documents found.", "data": {}}

    ranked = _rank_assets(all_assets, derived_query)[:limit_files] if derived_query else all_assets[:limit_files]

    sources: List[Dict[str, Any]] = []
    for a in ranked:
        content = _extract_asset_content(a, limit_chars=sample_chars_per_doc, token_cap=tokens_per_doc)
        sources.append(
            {
                "title": (a.title or a.file_name),
                "file_name": a.file_name,
                "updated_at": a.updated_at.isoformat(),
                "meta": a.meta or {},
                "content": content,
            }
        )

    analysis = await _llm_analyze_sources(question=_strip_ctrl(question), sources=sources)
    titles = ", ".join([s["title"] for s in sources])
    bullets = analysis.get("bullets") or []
    first_line = f"Analyzed {len(sources)} document(s): {titles}."
    summary_out = (first_line + (("\n- " + "\n- ".join(bullets[:5])) if bullets else ""))[:1200]

    out_data: Dict[str, Any] = {"analysis": analysis, "sources": sources}

    if make_slides:
        if not session_id:
            return {"status": "failed", "summary": "Missing session for slide creation.", "data": {}}
        slide_prompt = f"Create concise slides summarizing key insights for the question: {question}"
        context_text = "\n\n".join(_strip_ctrl(s["content"][:2000]) for s in sources if s.get("content"))[:8000]
        ej = await _llm_outline_to_editorjs(
            prompt=slide_prompt, context=context_text, max_sections=slides_max_sections
        )
        try:
            # best default: first source title
            deck_title = (sources[0]["title"] if sources else "Document Summary")[:140]
            s, before_note = await _slides_upsert_sync(
                session_id, title=deck_title, summary="", editorjs=ej, updated_by="tool:documents_analyze"
            )
            slides_data = await _format_slides_response(s)
            after_note = f"title={s.title!r}, v={s.version}"
            summary_slides = await _llm_summarize_change(title=s.title or "Deck", before_note=before_note, after_note=after_note)
            summary_out = (summary_out + "\n" + summary_slides).strip()
            out_data["slides"] = slides_data
        except Exception as e:
            log.exception("[documents_analyze] slide generation failed")
            summary_out = (summary_out + f"\nFailed to create slides: {e}").strip()

    return {"status": "ok", "summary": summary_out, "data": out_data}

@tool(
    "documents_generate_slides",
    args_schema=DocumentsSlidesInput,
    description="Generate/update slides from document(s) matching a fuzzy query. Returns the new deck and the sources used.",
)
async def documents_generate_slides(
    query: str,
    limit_files: int = 3,
    sample_chars_per_doc: int = MAX_EXTRACTED_CHARS_PER_DOC,
    tokens_per_doc: int = TOKENS_PER_DOC_DEFAULT,
    slides_max_sections: int = 6,
    title_override: Optional[str] = None,
) -> Dict[str, Any]:
    ctx = get_tool_context()
    agent_id, user_id = await _ensure_agent_user_from_ctx(ctx)
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "summary": "Missing session context.", "data": {}}
    if not agent_id or not user_id:
        return {"status": "failed", "summary": "Missing agent or user context.", "data": {}}

    items = await _db_list_assets(agent_id=agent_id, user_id=user_id, query=(query or "").strip(), limit=200)
    if not items:
        return {"status": "not_found", "summary": f"No documents found for “{query}”.", "data": {}}

    ranked = _rank_assets(items, query)[:limit_files]

    sources: List[Dict[str, Any]] = []
    for a in ranked:
        content = _extract_asset_content(a, limit_chars=sample_chars_per_doc, token_cap=tokens_per_doc)
        sources.append(
            {
                "title": (a.title or a.file_name),
                "file_name": a.file_name,
                "updated_at": a.updated_at.isoformat(),
                "meta": a.meta or {},
                "content": content,
            }
        )

    context_text = "\n\n".join(_strip_ctrl(s["content"][:2000]) for s in sources if s.get("content"))[:8000]
    prompt = f"Create concise slides summarizing the key ideas from the selected document(s) about: {query}"
    ej = await _llm_outline_to_editorjs(prompt=prompt, context=context_text, max_sections=slides_max_sections)

    try:
        deck_title = (title_override or (sources[0]["title"] if sources else "Slides"))[:140]
        s, before_note = await _slides_upsert_sync(
            session_id, title=deck_title, summary="", editorjs=ej, updated_by="tool:documents_generate_slides"
        )
        slides_data = await _format_slides_response(s)
        after_note = f"title={s.title!r}, v={s.version}"
        summary_slides = await _llm_summarize_change(title=s.title or "Deck", before_note=before_note, after_note=after_note)
        names = ", ".join([src["title"] for src in sources])
        summary_out = f"Built slides v{s.version} from {len(sources)} doc(s): {names}.\n{summary_slides}"
        return {"status": "ok", "summary": summary_out, "data": {"slides": slides_data, "sources": sources}}
    except Exception as e:
        log.exception("[documents_generate_slides] persist error")
        return {"status": "failed", "summary": f"Failed to generate slides: {e}", "data": {}}

# --------------------------------------------------------------------------------------
# Emotion Tool
# --------------------------------------------------------------------------------------
@tool(
    "emotion_detect",
    args_schema=EmotionInput,
    description="Classify a SINGLE dominant emotion (Joy, Anger, Sadness, Surprise) with intensity 1..3.",
)
async def emotion_detect(text: str) -> Dict[str, Any]:
    llm = _llm(model=DEFAULT_TEXT_MODEL, temperature=0.0, timeout=8)
    sys = (
        "Return a SINGLE dominant emotion.\n"
        "Allowed names: Joy, Anger, Sadness, Surprise. Intensity: integer 1..3.\n"
        'Return STRICT JSON ONLY like: {"name":"Joy","intensity":2}'
    )
    user = _strip_ctrl((text or "").strip()[:8000])
    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user)])
        raw = (getattr(msg, "content", "") or "").strip()
        data = json.loads(raw if raw.startswith("{") else raw[raw.find("{"):raw.rfind("}") + 1])
        name = str(data.get("name", "Joy")).title()
        inten = int(data.get("intensity", 1))
        if name not in ("Joy", "Anger", "Sadness", "Surprise"):
            name = "Joy"
        if inten < 1 or inten > 3:
            inten = 1
        return {"status": "ok", "summary": f"Detected {name} ({inten}/3).", "data": {"name": name, "intensity": inten}}
    except Exception:
        return {"status": "ok", "summary": "Defaulted to Joy (1/3).", "data": {"name": "Joy", "intensity": 1}}

# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
AGENT_TOOLS: Dict[str, Any] = {
    # Slides
    "slides_generate_or_update": slides_generate_or_update,
    "slides_fetch_latest": slides_fetch_latest,
    "slides_list_versions": slides_list_versions,
    "slides_diff_latest": slides_diff_latest,
    "slides_revert": slides_revert,
    # Documents
    "documents_list": documents_list,
    "documents_fetch": documents_fetch,
    "documents_analyze": documents_analyze,
    "documents_generate_slides": documents_generate_slides,
    # Emotion
    "emotion_detect": emotion_detect,
}

TOOLS_SCHEMA = [convert_to_openai_tool(t) for t in AGENT_TOOLS.values()]
