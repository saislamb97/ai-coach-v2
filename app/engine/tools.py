from __future__ import annotations

import json
import logging
import os
import re
import time
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

import wikipedia
from asgiref.sync import sync_to_async
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from memory.models import Slides  # Django ORM

log = logging.getLogger(__name__)
DB_PRIMARY = os.getenv("DB_PRIMARY_ALIAS", "default")

# -----------------------------------------------------------------------------
# Shared tool context
# -----------------------------------------------------------------------------
_TOOL_CTX: ContextVar[Dict[str, Any]] = ContextVar("TOOL_CTX", default={})

def set_tool_context(ctx: Dict[str, Any]) -> None:
    _TOOL_CTX.set(dict(ctx or {}))

def get_tool_context() -> Dict[str, Any]:
    return dict(_TOOL_CTX.get() or {})

# =============================================================================
# Metadata
# =============================================================================
AGENT_TOOL_DESCRIPTIONS: Dict[str, str] = {
    "search_wikipedia": "Find a topic and return a brief 1–2 sentence summary from Wikipedia.",
    "fetch_latest_slides": "Read-only: fetch the latest slide snapshot for the current session (current + previous + version).",
    "generate_or_update_slides": "Create/update a session presentation in Editor.js format. Skips empty input.",
    "emotion_analyze": "Return a single dominant emotion with intensity 1..3 for a given text.",
}

def default_agent_tools() -> List[str]:
    return ["search_wikipedia", "fetch_latest_slides", "generate_or_update_slides", "emotion_analyze"]

# =============================================================================
# Editor.js helpers (pure)
# =============================================================================
EditorJS = Dict[str, Any]
ALLOWED_BLOCK_TYPES = {"header", "paragraph", "list"}
MAX_BLOCKS = 120

def _elapsed_ms(t0: float) -> int:
    import time as _t
    return int((_t.perf_counter() - t0) * 1000)

def _now_ms() -> int:
    return int(time.time() * 1000)

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
            level = int(d.get("level", 2))
            if level not in (2, 3):
                level = 2
            d = {**d, "level": level}
        if t == "list":
            items = d.get("items")
            if not isinstance(items, list):
                items = []
            d = {**d, "items": [str(x) for x in items if isinstance(x, (str, int, float))]}
        out.append({"type": t, "data": d})
        if len(out) >= MAX_BLOCKS:
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

def minimal_editorjs(title: str, summary: str) -> EditorJS:
    title = (title or "").strip()[:120]
    summary = (summary or "").strip()[:200]
    blocks: List[Dict[str, Any]] = []
    if title:
        blocks.append({"type": "header", "data": {"text": title, "level": 2}})
    if summary:
        blocks.append({"type": "paragraph", "data": {"text": summary}})
    return {"time": _now_ms(), "version": "2.x", "blocks": blocks}

def _ensure_single_top_header(blocks: List[Dict[str, Any]], title: Optional[str]) -> List[Dict[str, Any]]:
    if title:
        seen_header = False
        out: List[Dict[str, Any]] = []
        for b in blocks:
            if b.get("type") == "header":
                if not seen_header:
                    out.append({"type": "header", "data": {"text": title, "level": 2}})
                    seen_header = True
            else:
                out.append(b)
        if not seen_header:
            out = [{"type": "header", "data": {"text": title, "level": 2}}] + out
        return out
    first = True
    out: List[Dict[str, Any]] = []
    for b in blocks:
        if b.get("type") == "header":
            if first:
                out.append({"type": "header", "data": {"text": (b.get('data', {}).get('text') or '').strip(), "level": 2}})
                first = False
        else:
            out.append(b)
    return out

def _compact_paragraphs(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    carry: Optional[str] = None
    for b in blocks:
        if b.get("type") == "paragraph":
            text = (b.get("data", {}).get("text") or "").strip()
            if not text:
                continue
            if carry is None:
                carry = text
            else:
                if len(carry) + len(text) <= 320:
                    carry = f"{carry} {text}"
                else:
                    out.append({"type": "paragraph", "data": {"text": carry}})
                    carry = text
        else:
            if carry is not None:
                out.append({"type": "paragraph", "data": {"text": carry}})
                carry = None
            out.append(b)
    if carry is not None:
        out.append({"type": "paragraph", "data": {"text": carry}})
    return out

def _headers(ej: Dict[str, Any]) -> List[str]:
    out = []
    for b in (ej or {}).get("blocks", []):
        if (b.get("type") or "").lower() == "header":
            t = (b.get("data", {}).get("text") or "").strip()
            if t:
                out.append(t)
    return out

def _diff_simple_headers(cur_ej: Dict[str, Any], prev_ej: Dict[str, Any]) -> Dict[str, List[str]]:
    cur = _headers(cur_ej or {})
    prev = _headers(prev_ej or {})
    added = [h for h in cur if h not in prev]
    removed = [h for h in prev if h not in cur]
    unchanged = [h for h in cur if h in prev]
    return {"added": added, "removed": removed, "unchanged": unchanged}

# =============================================================================
# Schemas
# =============================================================================
class WikipediaInput(BaseModel):
    query: str = Field(..., description="Topic to summarize in 1–2 sentences")

class SlideInput(BaseModel):
    title: Optional[str] = Field(None, description="Deck title (optional)")
    summary: Optional[str] = Field(None, description="Short abstract (optional)")
    editorjs: Optional[EditorJS | List[Dict[str, Any]]] = Field(
        None, description="Editor.js document or blocks[]; optional"
    )
    ai_enrich: bool = Field(True, description="If true, LLM expands with outline/sections")
    max_sections: int = Field(6, ge=1, le=12, description="Max sections when AI enrichment is enabled.")

    @field_validator("editorjs")
    @classmethod
    def _validate_editorjs(cls, v):
        if v is None:
            return v
        ej = normalize_editorjs(v)
        if not ej:
            raise ValueError("editorjs must contain non-empty blocks in Editor.js format")
        return ej

class EmotionInput(BaseModel):
    text: str = Field(..., description="Full response text to classify (one dominant emotion).")

class FetchSlidesInput(BaseModel):
    """No arguments; relies on tool context for session_id."""
    pass

# =============================================================================
# Internal LLM helpers
# =============================================================================
def _extract_json_maybe(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    if not raw:
        return None
    fence = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
    m = fence.match(raw)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
    except Exception:
        pass
    return None

async def _ai_expand_editorjs(
    *,
    title: Optional[str],
    summary: str,
    base_ej: EditorJS,
    max_sections: int,
    model: Optional[str] = None,
) -> Optional[EditorJS]:
    model_name = (model or "gpt-4o-mini").strip()
    llm = ChatOpenAI(model=model_name, temperature=0.2, timeout=20)

    sys = (
        "Generate slide content as Editor.js JSON ONLY.\n"
        'Schema: {"time":INT,"version":"2.x","blocks":[...]}\n'
        "Use 'header' (level 3) for SECTION titles (not the deck title).\n"
        "Do NOT output a duplicate top-level title header; assume the deck title is provided separately.\n"
        "Use 'paragraph' and optional 'list' (unordered, items:[...]).\n"
        f"Keep <= {max_sections} concise, non-redundant sections."
    )
    user = (
        (f"# Deck Title\n{title}\n\n" if title else "")
        + f"# Summary\n{summary}\n\n"
        + f"# Sample Blocks (first 2 shown)\n{json.dumps(base_ej.get('blocks', [])[:2], ensure_ascii=False)}\n"
        + "\nReturn ONLY valid JSON."
    )

    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user)])
        raw = (getattr(msg, "content", "") or "").strip()
        data = _extract_json_maybe(raw)
        ej = normalize_editorjs(data) if data is not None else None
        return ej
    except Exception as e:
        log.warning("[slide:ai_enrich] failed to expand deck: %s", e)
        return None

# =============================================================================
# ORM read helper (wrapped for async contexts)
# =============================================================================
@sync_to_async
def _db_fetch_latest_slides_snapshot(session_id: int) -> Optional[Dict[str, Any]]:
    s = (
        Slides.objects.using(DB_PRIMARY)
        .filter(session_id=session_id)
        .order_by("-updated_at")
        .first()
    )
    if not s:
        return None
    s.refresh_from_db(using=DB_PRIMARY)
    snap = {
        "title": s.title,
        "summary": s.summary,
        "editorjs": s.editorjs,
        "previous": {
            "title": s.previous_title,
            "summary": s.previous_summary,
            "editorjs": s.previous_editorjs,
        },
        "version": s.version,
        "thread_id": s.session.thread_id,
        "updated_by": s.updated_by,
        "updated_at": s.updated_at.isoformat(),
        "diff_headers": _diff_simple_headers(s.editorjs or {}, s.previous_editorjs or {}),
    }
    return snap

# =============================================================================
# Tools
# =============================================================================
@tool(
    "search_wikipedia",
    args_schema=WikipediaInput,
    description="Return a brief 1–2 sentence summary for a topic from Wikipedia.",
)
async def search_wikipedia(query: str) -> Dict[str, Any]:
    """Return a brief 1–2 sentence summary for a topic from Wikipedia."""
    import asyncio
    wikipedia.set_lang("en")

    async def _search_and_summary(q: str) -> Tuple[List[str], Optional[str]]:
        def _blocking() -> Tuple[List[str], Optional[str]]:
            hits = wikipedia.search(q, results=5)
            if not hits:
                return [], None
            title = next((h for h in hits if h.lower() == q.lower() or h.lower().startswith(q.lower())), hits[0])
            try:
                summary = wikipedia.summary(title, sentences=2)
            except Exception:
                summary = None
            return hits, summary
        return await asyncio.to_thread(_blocking)

    t0 = time.perf_counter()
    q = (query or "").strip()
    if not q:
        return {"status": "failed", "error": "empty_query", "meta": {}}

    try:
        hits, summary = await _search_and_summary(q)
        if not hits:
            return {"status": "failed", "error": "no_results", "meta": {"query": q}}
        if not summary:
            return {"status": "failed", "error": "not_found", "meta": {"query": q}}
        ms = _elapsed_ms(t0)
        ctx = get_tool_context()
        return {"status": "succeeded", "data": {"query": q, "title": hits[0], "summary": summary, **ctx}, "meta": {"elapsed_ms": ms, "suggestions": hits}}
    except wikipedia.exceptions.DisambiguationError as e:
        return {"status": "failed", "error": "ambiguous", "meta": {"suggestions": e.options[:8], "query": q}}
    except wikipedia.exceptions.PageError:
        return {"status": "failed", "error": "not_found", "meta": {"query": q}}
    except Exception as e:
        log.exception("[tool:wikipedia] unexpected")
        return {"status": "failed", "error": f"error:{e}", "meta": {"query": q}}

@tool(
    "fetch_latest_slides",
    args_schema=FetchSlidesInput,
    description="Read-only: fetch the latest Slides row for this session from the primary DB.",
)
async def fetch_latest_slides() -> Dict[str, Any]:
    """Read-only fetch of the latest slides snapshot for the current session (never writes)."""
    t0 = time.perf_counter()
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "error": "missing_session_context"}

    try:
        snap = await _db_fetch_latest_slides_snapshot(int(session_id))
    except Exception as e:
        log.exception("[tool:fetch_latest_slides] DB read failed")
        return {"status": "failed", "error": f"db_error:{e}"}

    if not snap:
        return {"status": "not_found", "meta": {"elapsed_ms": _elapsed_ms(t0), **ctx}}

    return {"status": "ok", "slides": snap, "meta": {"elapsed_ms": _elapsed_ms(t0), **ctx}}

@tool(
    "generate_or_update_slides",
    args_schema=SlideInput,
    description="Create or update slides for this session. Skips persistence if input is empty.",
)
async def generate_or_update_slides(
    title: Optional[str] = None,
    summary: Optional[str] = None,
    editorjs: EditorJS | None = None,
    ai_enrich: bool = True,
    max_sections: int = 6,
) -> Dict[str, Any]:
    """Prepare a new slides payload; caller decides whether to persist via rotate/update."""
    def meaningful(t: Any, s: Any, ej: Any) -> bool:
        def _empty_str(v: Any) -> bool:
            if not isinstance(v, str):
                return True
            vv = v.strip().lower()
            return vv == "" or vv in {"string", "null", "none"}
        has_t = isinstance(t, str) and not _empty_str(t)
        has_s = isinstance(s, str) and not _empty_str(s)
        has_b = isinstance(ej, dict) and bool(ej.get("blocks"))
        return has_t or has_s or has_b

    ej = normalize_editorjs(editorjs) if editorjs is not None else None
    title_clean = (title or "").strip()
    summary_clean = (summary or "").strip()

    if not meaningful(title_clean, summary_clean, ej):
        return {"no_write": True, **get_tool_context()}

    if ej is None:
        ej = minimal_editorjs(title_clean, summary_clean)

    deck_title: Optional[str] = title_clean or None
    if deck_title is None:
        for b in ej.get("blocks", []):
            if b.get("type") == "header":
                t = (b.get("data", {}).get("text") or "").strip()
                if t:
                    deck_title = t
                    break

    deck_summary = summary_clean

    if ai_enrich:
        enriched = await _ai_expand_editorjs(title=deck_title, summary=deck_summary, base_ej=ej, max_sections=max_sections)
        if enriched and (enriched.get("blocks") or []):
            blocks = list(enriched.get("blocks") or [])
            blocks = _ensure_single_top_header(blocks, deck_title)
            blocks = _compact_paragraphs(blocks)
            ej = normalize_editorjs({"time": int(time.time() * 1000), "version": "2.x", "blocks": blocks}) or enriched

    if not deck_title:
        for b in ej.get("blocks", []):
            if b.get("type") == "header":
                deck_title = (b.get("data", {}).get("text") or "").strip() or deck_title
                if deck_title:
                    break

    return {"title": deck_title or "", "summary": deck_summary, "editorjs": ej, **get_tool_context()}

@tool(
    "emotion_analyze",
    args_schema=EmotionInput,
    description="Classify a single dominant emotion (Joy/Anger/Sadness/Surprise) with intensity 1..3.",
)
async def emotion_analyze(text: str) -> Dict[str, Any]:
    """Return a single dominant emotion and intensity 1..3 for the given text."""
    model_name = "gpt-4o-mini"
    llm = ChatOpenAI(model=model_name, temperature=0.0, timeout=8)
    sys = (
        "Return a SINGLE dominant emotion for the input.\n"
        "Allowed names: Joy, Anger, Sadness, Surprise. Intensity: integer 1..3.\n"
        'Return STRICT JSON ONLY like: {"name":"Joy","intensity":2}'
    )
    user = (text or "").strip()[:8000]
    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user)])
        raw = (getattr(msg, "content", "") or "").strip()
        data = json.loads(raw)
        name = str(data.get("name", "Joy")).title()
        inten = int(data.get("intensity", 1))
        if name not in ("Joy", "Anger", "Sadness", "Surprise"):
            name = "Joy"
        if inten < 1 or inten > 3:
            inten = 1
        return {"name": name, "intensity": inten}
    except Exception:
        return {"name": "Joy", "intensity": 1}

# =============================================================================
# Registry & OpenAI tool schemas
# =============================================================================
AGENT_TOOLS: Dict[str, Any] = {
    "search_wikipedia": search_wikipedia,
    "fetch_latest_slides": fetch_latest_slides,
    "generate_or_update_slides": generate_or_update_slides,
    "emotion_analyze": emotion_analyze,
}

TOOLS_SCHEMA = [convert_to_openai_tool(t) for t in AGENT_TOOLS.values()]
