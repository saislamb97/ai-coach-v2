from __future__ import annotations

import time
import json
import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, field_validator
import wikipedia
from wikipedia.exceptions import PageError, DisambiguationError

log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Shared tool context (auto-injected by nodes): session/bot/agent IDs
# -------------------------------------------------------------------
_TOOL_CTX: ContextVar[Dict[str, Any]] = ContextVar("TOOL_CTX", default={})

def set_tool_context(ctx: Dict[str, Any]) -> None:
    _TOOL_CTX.set(dict(ctx or {}))

def get_tool_context() -> Dict[str, Any]:
    return dict(_TOOL_CTX.get() or {})

# =========================================================
# Public metadata (used by preambles / admin UI)
# =========================================================
AGENT_TOOL_DESCRIPTIONS: Dict[str, str] = {
    "search_wikipedia": "Find a topic and return a brief 1–2 sentence summary from Wikipedia.",
    "generate_or_update_slides": "Create/update a session presentation in Editor.js format. Can enrich with AI.",
}

AGENT_TOOL_CHOICES: List[Tuple[str, str]] = sorted(
    [
        ("search_wikipedia", "Wikipedia Search"),
        ("generate_or_update_slides", "Generate/Update Slides"),
    ],
    key=lambda x: x[1].lower(),
)

def default_agent_tools() -> List[str]:
    return ["search_wikipedia", "generate_or_update_slides"]

# =========================================================
# Small utils
# =========================================================
def _elapsed_ms(t0: float) -> int:
    import time as _t
    return int((_t.perf_counter() - t0) * 1000)

def _wrap_ok(data: Any, **meta) -> Dict[str, Any]:
    # kept for wikipedia only
    return {"status": "succeeded", "data": data, "meta": meta or {}}

def _wrap_err(msg: str, **meta) -> Dict[str, Any]:
    return {"status": "failed", "error": msg, "meta": meta or {}}

def _preview(v: Any, n: int = 160) -> str:
    try:
        s = str(v)
    except Exception:
        s = repr(v)
    return s if len(s) <= n else f"{s[: n//2]} … {s[-(n//2):]}"

# =========================================================
# Editor.js helpers (pure)
# =========================================================
EditorJS = Dict[str, Any]

def _coerce_blocks(blocks_any: Any) -> List[Dict[str, Any]]:
    if not isinstance(blocks_any, list):
        return []
    out: List[Dict[str, Any]] = []
    for b in blocks_any:
        if not isinstance(b, dict):
            continue
        bid = b.get("id") or b.get("_id") or ""
        btype = b.get("type")
        bdata = b.get("data")
        if not isinstance(btype, str) or not isinstance(bdata, dict):
            continue
        out.append({"id": str(bid) if bid else None, "type": btype, "data": bdata})
    return out

def _now_ms() -> int:
    return int(time.time() * 1000)

def normalize_editorjs(obj: Any) -> Optional[EditorJS]:
    """
    Accepts: full Editor.js dict, {"blocks":[...]}, or bare list of blocks.
    Returns normalized dict or None if invalid/empty.
    """
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

def _first_text_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    for b in blocks:
        txt = (b.get("data", {}).get("text") or "").strip()
        if txt:
            return txt
    return ""

def _infer_title_summary(ej: EditorJS, title_in: Optional[str], summary_in: Optional[str]) -> tuple[str, str]:
    """
    Title: first header.text else any text.
    Summary: first paragraph.text else next text block(s) joined (<=160 chars).
    """
    blocks = ej.get("blocks") or []
    title = (title_in or "").strip()
    summary = (summary_in or "").strip()

    if not title:
        for b in blocks:
            if (b.get("type") or "").lower() == "header":
                t = (b.get("data", {}).get("text") or "").strip()
                if t:
                    title = t
                    break
        if not title:
            title = _first_text_from_blocks(blocks)[:80] or "Presentation"

    if not summary:
        for b in blocks:
            if (b.get("type") or "").lower() == "paragraph":
                p = (b.get("data", {}).get("text") or "").strip()
                if p:
                    summary = p
                    break
        if not summary:
            parts: List[str] = []
            for b in blocks:
                t = (b.get("data", {}).get("text") or "").strip()
                if t:
                    parts.append(t)
                if len(" ".join(parts)) >= 160:
                    break
            summary = " ".join(parts)[:160]

    return title or "Presentation", summary or ""

def minimal_editorjs(title: str, summary: str) -> EditorJS:
    title = (title or "Presentation").strip()[:120]
    summary = (summary or "").strip()[:200]
    return {
        "time": _now_ms(),
        "version": "2.x",
        "blocks": [
            {"type": "header", "data": {"text": title, "level": 2}},
            {"type": "paragraph", "data": {"text": summary or " "}},
        ],
    }

# =========================================================
# Pydantic schemas
# =========================================================
class WikipediaInput(BaseModel):
    query: str = Field(..., description="Topic to summarize in 1–2 sentences")

class SlideInput(BaseModel):
    title: Optional[str] = Field(None, description="Deck title (optional; inferred if blank)")
    summary: Optional[str] = Field(None, description="Short abstract (optional; inferred if blank)")
    editorjs: Optional[EditorJS | List[Dict[str, Any]]] = Field(
        None, description="Editor.js document or blocks[]; optional for creation"
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

# =========================================================
# Internal LLM helpers for slide enrichment
# =========================================================
async def _ai_expand_editorjs(
    *,
    title: str,
    summary: str,
    base_ej: EditorJS,
    max_sections: int,
    model: str | None = None,
) -> Optional[EditorJS]:
    """
    Return Editor.js JSON with section headers and short text bullets.
    """
    model_name = (model or "gpt-4o-mini").strip()
    llm = ChatOpenAI(model=model_name, temperature=0.2, timeout=20)

    sys = (
        "Generate slide content as Editor.js JSON ONLY.\n"
        'Schema: {"time":INT,"version":"2.x","blocks":[...]}\n'
        "Use 'header' (level 2/3), 'paragraph', optional 'list' (unordered, items:[...]).\n"
        f"Keep <= {max_sections} concise, non-redundant sections."
    )
    user = (
        f"# Title\n{title}\n\n"
        f"# Summary\n{summary}\n\n"
        f"# Base Blocks (first 2 shown)\n"
        f"{json.dumps(base_ej.get('blocks', [])[:2], ensure_ascii=False)}\n"
        "\nReturn ONLY valid JSON."
    )

    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user)])
        raw = (getattr(msg, "content", "") or "").strip()
        data = json.loads(raw)
        ej = normalize_editorjs(data)
        return ej
    except Exception as e:
        log.warning("[slide:ai_enrich] failed to expand deck: %s", e)
        return None

# =========================================================
# Tools
# =========================================================
@tool("search_wikipedia", args_schema=WikipediaInput)
async def search_wikipedia(query: str) -> Dict[str, Any]:
    """Return a brief 1–2 sentence summary for a topic from Wikipedia."""
    t0 = time.perf_counter()
    q = (query or "").strip()
    log.info("[tool:wikipedia] query=%s", _preview(q))
    if not q:
        return _wrap_err("empty_query")

    try:
        hits = wikipedia.search(q, results=5)
        if not hits:
            return _wrap_err("no_results", query=q)
        title = next((h for h in hits if h.lower() == q.lower() or h.lower().startswith(q.lower())), hits[0])
        summary = wikipedia.summary(title, sentences=2)
        ms = _elapsed_ms(t0)
        ctx = get_tool_context()
        return _wrap_ok(
            {"query": q, "title": title, "summary": summary, **ctx},
            elapsed_ms=ms,
            suggestions=hits,
        )
    except DisambiguationError as e:
        return _wrap_err("ambiguous", suggestions=e.options[:8], query=q)
    except PageError:
        return _wrap_err("not_found", query=q)
    except Exception as e:
        log.exception("[tool:wikipedia] unexpected")
        return _wrap_err(f"error:{e}", query=q)

@tool("generate_or_update_slides", args_schema=SlideInput)
async def generate_or_update_slides(
    title: Optional[str] = None,
    summary: Optional[str] = None,
    editorjs: EditorJS | None = None,
    ai_enrich: bool = True,
    max_sections: int = 6,
) -> Dict[str, Any]:
    """
    Create/update the session's Editor.js deck synchronously (no status/job_id).
    ALWAYS use this when the user asks for a presentation/deck/slides/PPT/Keynote.
    Returns: {"title":..., "summary":..., "editorjs":...}
    """
    ctx = get_tool_context()

    ej = normalize_editorjs(editorjs) if editorjs is not None else None
    if not ej:
        base_title = (title or "").strip() or "Presentation"
        base_summary = (summary or "").strip()
        ej = minimal_editorjs(base_title, base_summary)

    try:
        t, s = _infer_title_summary(ej, title, summary)

        if ai_enrich:
            enriched = await _ai_expand_editorjs(title=t, summary=s, base_ej=ej, max_sections=max_sections)
            if enriched and (enriched.get("blocks") or []):
                keep = ej.get("blocks")[:1] if ej.get("blocks") else []
                ej = normalize_editorjs({"time": _now_ms(), "version": "2.x", "blocks": keep + enriched.get("blocks")}) or ej
                t, s = _infer_title_summary(ej, t, s)

        payload = {"title": t, "summary": s, "editorjs": ej, **ctx}
        return payload
    except Exception as e:
        log.exception("[tool:generate_or_update_slides] failed")
        # even on failure, return a minimal doc so UI persists something sensible
        base_title = (title or "Presentation")
        base_summary = (summary or "")
        return {"title": base_title, "summary": base_summary, "editorjs": minimal_editorjs(base_title, base_summary), **ctx}

# =========================================================
# Registry & OpenAI tool schemas
# =========================================================
AGENT_TOOLS: Dict[str, Any] = {
    "search_wikipedia": search_wikipedia,
    "generate_or_update_slides": generate_or_update_slides,
}

# IMPORTANT: plain list (no trailing comma!)
TOOLS_SCHEMA = [convert_to_openai_tool(t) for t in AGENT_TOOLS.values()]
