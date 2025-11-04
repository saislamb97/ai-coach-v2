from __future__ import annotations

import json
import logging
import time
import re
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import wikipedia
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Shared tool context (auto-injected by nodes): session/bot/agent IDs
# -----------------------------------------------------------------------------
_TOOL_CTX: ContextVar[Dict[str, Any]] = ContextVar("TOOL_CTX", default={})


def set_tool_context(ctx: Dict[str, Any]) -> None:
    """Inject session/bot/agent metadata so tools can stamp responses."""
    _TOOL_CTX.set(dict(ctx or {}))


def get_tool_context() -> Dict[str, Any]:
    """Return the current tool context (safe copy)."""
    return dict(_TOOL_CTX.get() or {})


# =============================================================================
# Public metadata (used by preambles / admin UI)
# =============================================================================
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


# =============================================================================
# Small utils
# =============================================================================
def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _wrap_ok(data: Any, **meta) -> Dict[str, Any]:
    # kept for wikipedia only (slides returns raw payload for callers)
    return {"status": "succeeded", "data": data, "meta": meta or {}}


def _wrap_err(msg: str, **meta) -> Dict[str, Any]:
    return {"status": "failed", "error": msg, "meta": meta or {}}


def _preview(v: Any, n: int = 160) -> str:
    try:
        s = str(v)
    except Exception:
        s = repr(v)
    return s if len(s) <= n else f"{s[: n//2]} … {s[-(n//2):]}"


# =============================================================================
# Editor.js helpers (pure, side-effect-free)
# =============================================================================
EditorJS = Dict[str, Any]

ALLOWED_BLOCK_TYPES = {"header", "paragraph", "list"}  # keep it tight for now
MAX_BLOCKS = 120  # guardrail so we don't persist giant decks


def _coerce_blocks(blocks_any: Any) -> List[Dict[str, Any]]:
    if not isinstance(blocks_any, list):
        return []
    out: List[Dict[str, Any]] = []
    for b in blocks_any:
        if not isinstance(b, dict):
            continue
        btype = (b.get("type") or "").strip()
        bdata = b.get("data")
        if not isinstance(btype, str) or not isinstance(bdata, dict):
            continue
        t = btype.lower()
        if t not in ALLOWED_BLOCK_TYPES:
            continue
        # normalize header level (2 or 3 only)
        if t == "header":
            level = int(bdata.get("level", 2))
            if level not in (2, 3):
                level = 2
            bdata = {**bdata, "level": level}
        # normalize list items (ensure list of strings)
        if t == "list":
            items = bdata.get("items")
            if not isinstance(items, list):
                items = []
            bdata = {**bdata, "items": [str(x) for x in items if isinstance(x, (str, int, float))]}
        out.append({"type": t, "data": bdata})
        if len(out) >= MAX_BLOCKS:
            break
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
            if b.get("type") == "header":
                t = (b.get("data", {}).get("text") or "").strip()
                if t:
                    title = t
                    break
        if not title:
            title = _first_text_from_blocks(blocks)[:80] or "Presentation"

    if not summary:
        for b in blocks:
            if b.get("type") == "paragraph":
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


def _ensure_single_top_header(blocks: List[Dict[str, Any]], title: str) -> List[Dict[str, Any]]:
    """
    Guarantee exactly one top header. If none exists, insert one.
    If one exists and differs from title, keep AI's header (no duplicates).
    Remove extra header blocks beyond the first.
    """
    first_header_idx = next((i for i, b in enumerate(blocks) if b.get("type") == "header"), -1)
    if first_header_idx == -1:
        # add our header up front
        blocks = [{"type": "header", "data": {"text": title, "level": 2}}] + blocks
        first_header_idx = 0

    # remove all subsequent headers
    filtered: List[Dict[str, Any]] = []
    seen_header = False
    for i, b in enumerate(blocks):
        if b.get("type") != "header":
            filtered.append(b)
            continue
        if not seen_header:
            # keep the first header but normalize level
            h = {"type": "header", "data": {"text": (b.get("data", {}).get("text") or title).strip() or title, "level": 2}}
            filtered.append(h)
            seen_header = True
        # else: skip additional header blocks entirely
    return filtered


def _compact_paragraphs(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive small paragraphs into one to reduce clutter."""
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
                # merge small paragraphs
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


# =============================================================================
# Pydantic schemas
# =============================================================================
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


# =============================================================================
# Internal LLM helpers for slide enrichment
# =============================================================================
def _extract_json_maybe(raw: str) -> Optional[dict]:
    """
    Be resilient to code-fences or accidental prose: try best to parse JSON object.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    # strip triple backticks if present
    fence = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
    m = fence.match(raw)
    if m:
        raw = m.group(1).strip()
    # direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass
    # find first {...} region heuristically (simple, good-enough)
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
        "Use 'header' (level 3) for SECTION titles (not the deck title).\n"
        "Do NOT output a duplicate top-level title header; assume the deck already has one.\n"
        "Use 'paragraph' and optional 'list' (unordered, items:[...]).\n"
        f"Keep <= {max_sections} concise, non-redundant sections."
    )
    user = (
        f"# Deck Title\n{title}\n\n"
        f"# Summary\n{summary}\n\n"
        f"# Sample Blocks (first 2 shown)\n"
        f"{json.dumps(base_ej.get('blocks', [])[:2], ensure_ascii=False)}\n"
        "\nReturn ONLY valid JSON."
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
# Tools
# =============================================================================
@tool("search_wikipedia", args_schema=WikipediaInput)
async def search_wikipedia(query: str) -> Dict[str, Any]:
    """
    Return a brief 1–2 sentence summary for a topic from Wikipedia.
    Non-blocking: calls wikipedia APIs in a worker thread.
    """
    import asyncio

    wikipedia.set_lang("en")

    async def _search_and_summary(q: str) -> Tuple[List[str], Optional[str]]:
        def _blocking() -> Tuple[List[str], Optional[str]]:
            hits = wikipedia.search(q, results=5)
            if not hits:
                return [], None
            # prefer exact/startswith match
            title = next((h for h in hits if h.lower() == q.lower() or h.lower().startswith(q.lower())), hits[0])
            try:
                summary = wikipedia.summary(title, sentences=2)
            except Exception:
                summary = None
            return hits, summary

        return await asyncio.to_thread(_blocking)

    t0 = time.perf_counter()
    q = (query or "").strip()
    log.info("[tool:wikipedia] query=%s", _preview(q))
    if not q:
        return _wrap_err("empty_query")

    try:
        hits, summary = await _search_and_summary(q)
        if not hits:
            return _wrap_err("no_results", query=q)
        if not summary:
            return _wrap_err("not_found", query=q)
        ms = _elapsed_ms(t0)
        ctx = get_tool_context()
        return _wrap_ok(
            {"query": q, "title": hits[0], "summary": summary, **ctx},
            elapsed_ms=ms,
            suggestions=hits,
        )
    except wikipedia.exceptions.DisambiguationError as e:
        return _wrap_err("ambiguous", suggestions=e.options[:8], query=q)
    except wikipedia.exceptions.PageError:
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

    Returns: {"title":..., "summary":..., "editorjs":..., <tool_ctx>}
    """
    ctx = get_tool_context()

    # Normalize/seed a base deck
    ej = normalize_editorjs(editorjs) if editorjs is not None else None
    if not ej:
        base_title = (title or "").strip() or "Presentation"
        base_summary = (summary or "").strip()
        ej = minimal_editorjs(base_title, base_summary)

    try:
        # Derive canonical title/summary from existing content
        t, s = _infer_title_summary(ej, title, summary)

        # Optional AI expand
        if ai_enrich:
            enriched = await _ai_expand_editorjs(title=t, summary=s, base_ej=ej, max_sections=max_sections)
            if enriched and (enriched.get("blocks") or []):
                blocks = list(enriched.get("blocks") or [])
                # Ensure exactly one top-level header (fixes duplicate headers)
                blocks = _ensure_single_top_header(blocks, t)
                # Compact consecutive short paragraphs
                blocks = _compact_paragraphs(blocks)
                ej = normalize_editorjs({"time": _now_ms(), "version": "2.x", "blocks": blocks}) or enriched
                t, s = _infer_title_summary(ej, t, s)

        payload = {"title": t, "summary": s, "editorjs": ej, **ctx}
        return payload
    except Exception as e:
        log.exception("[tool:generate_or_update_slides] failed")
        # even on failure, return a minimal doc so UI persists something sensible
        base_title = (title or "Presentation")
        base_summary = (summary or "")
        return {"title": base_title, "summary": base_summary, "editorjs": minimal_editorjs(base_title, base_summary), **ctx}


# =============================================================================
# Registry & OpenAI tool schemas
# =============================================================================
AGENT_TOOLS: Dict[str, Any] = {
    "search_wikipedia": search_wikipedia,
    "generate_or_update_slides": generate_or_update_slides,
}

# IMPORTANT: plain list (no trailing comma!)
TOOLS_SCHEMA = [convert_to_openai_tool(t) for t in AGENT_TOOLS.values()]
