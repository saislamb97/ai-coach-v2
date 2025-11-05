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
from django.db import transaction
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from memory.models import Slides  # Django ORM

# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------
log = logging.getLogger(__name__)
DB_PRIMARY = os.getenv("DB_PRIMARY_ALIAS", "default")

# Shared tool context (session_id, bot_id, etc.) set by nodes.py before routing tools
_TOOL_CTX: ContextVar[Dict[str, Any]] = ContextVar("TOOL_CTX", default={})

def set_tool_context(ctx: Dict[str, Any]) -> None:
    _TOOL_CTX.set(dict(ctx or {}))

def get_tool_context() -> Dict[str, Any]:
    return dict(_TOOL_CTX.get() or {})

# --------------------------------------------------------------------------------------
# Editor.js helpers (minimal, safe)
# --------------------------------------------------------------------------------------
EditorJS = Dict[str, Any]
ALLOWED_BLOCK_TYPES = {"header", "paragraph", "list"}
MAX_BLOCKS = 120

def _now_ms() -> int:
    return int(time.time() * 1000)

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

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
            if level not in (1, 2, 3):
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

def minimal_editorjs(title: str, summary: str = "") -> EditorJS:
    title = (title or "Untitled Deck").strip()[:140]
    summary = (summary or "").strip()[:300]
    blocks: List[Dict[str, Any]] = [{"type": "header", "data": {"text": title, "level": 2}}]
    if summary:
        blocks.append({"type": "paragraph", "data": {"text": summary}})
    return {"time": _now_ms(), "version": "2.x", "blocks": blocks}

def _ensure_single_top_header(blocks: List[Dict[str, Any]], title: Optional[str]) -> List[Dict[str, Any]]:
    if not title:
        return blocks
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
    return {
        "added": [h for h in cur if h not in prev],
        "removed": [h for h in prev if h not in cur],
        "unchanged": [h for h in cur if h in prev],
    }

# --------------------------------------------------------------------------------------
# LLM helpers (outline generation)
# --------------------------------------------------------------------------------------
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

async def _ai_outline_from_prompt(*, prompt: str, max_sections: int = 6, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    model_name = (model or "gpt-4o-mini").strip()
    llm = ChatOpenAI(model=model_name, temperature=0.3, timeout=20)

    sys = (
        "Return STRICT JSON for a slide outline.\n"
        'Schema: {"title":"string","summary":"1-2 sentences",'
        '"sections":[{"header":"string","bullets":["point",...]}]}\n'
        f"Keep sections <= {max_sections}. Avoid fluff/duplicates."
    )
    user = (prompt or "").strip()[:4000]
    if not user:
        return None

    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user)])
        data = _extract_json_maybe(getattr(msg, "content", "") or "")
        if not isinstance(data, dict):
            return None

        title = (data.get("title") or "").strip()[:140] or "Untitled Deck"
        summary = (data.get("summary") or "").strip()[:300]
        sections = data.get("sections") or []

        blocks: List[Dict[str, Any]] = [{"type": "header", "data": {"text": title, "level": 2}}]
        if summary:
            blocks.append({"type": "paragraph", "data": {"text": summary}})
        for s in sections:
            hdr = (s.get("header") or "").strip()
            if not hdr:
                continue
            blocks.append({"type": "header", "data": {"text": hdr, "level": 3}})
            bullets = [str(x).strip() for x in (s.get("bullets") or []) if str(x).strip()]
            if bullets:
                blocks.append({"type": "list", "data": {"style": "unordered", "items": bullets[:8]}})
        ej = normalize_editorjs({"time": _now_ms(), "version": "2.x", "blocks": blocks})
        if not ej:
            return None
        return {"title": title, "summary": summary, "editorjs": ej}
    except Exception as e:
        log.warning("[slides:outline] synth failed: %s", e)
        return None

# --------------------------------------------------------------------------------------
# DB helpers (ALWAYS primary; re-read after write)
# --------------------------------------------------------------------------------------
@sync_to_async
def _db_fetch_latest_slides_snapshot(session_id: int) -> Optional[Dict[str, Any]]:
    """
    Always fetch the freshest slide row from the primary DB.
    """
    s = (
        Slides.objects.using(DB_PRIMARY)
        .filter(session_id=session_id)
        .order_by("-updated_at", "-id")
        .first()
    )
    if not s:
        return None
    s.refresh_from_db(using=DB_PRIMARY)
    return {
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

@sync_to_async
def _db_rotate_and_save_slides(session_id: int, payload: Dict[str, Any], updated_by: str) -> Dict[str, Any]:
    """
    Atomic rotate+save on primary, then re-read the latest snapshot to guarantee callers
    get the truly current state even under concurrency.
    """
    with transaction.atomic(using=DB_PRIMARY):
        s = (
            Slides.objects.using(DB_PRIMARY)
            .select_for_update()
            .filter(session_id=session_id)
            .first()
        )
        if not s:
            s = Slides.objects.using(DB_PRIMARY).create(session_id=session_id)

        title = (payload.get("title") or "Untitled Deck").strip()
        summary = (payload.get("summary") or "").strip()
        editorjs = payload.get("editorjs") or minimal_editorjs(title, summary)

        if hasattr(s, "rotate_and_update") and callable(getattr(s, "rotate_and_update")):
            s.rotate_and_update(title=title, summary=summary, editorjs=editorjs, updated_by=updated_by)
        else:
            from django.utils import timezone
            s.previous_title = s.title
            s.previous_summary = s.summary
            s.previous_editorjs = s.editorjs
            s.title = title
            s.summary = summary
            s.editorjs = editorjs
            s.updated_by = updated_by
            s.version = (s.version or 0) + 1
            s.updated_at = timezone.now()

        s.save(update_fields=[
            "previous_title","previous_summary","previous_editorjs",
            "title","summary","editorjs","updated_by","version","updated_at"
        ])

    # Fresh re-read (read-after-write) from primary
    fresh = (
        Slides.objects.using(DB_PRIMARY)
        .filter(session_id=session_id)
        .order_by("-updated_at", "-id")
        .first()
    )
    fresh.refresh_from_db(using=DB_PRIMARY)
    return {
        "title": fresh.title,
        "summary": fresh.summary,
        "editorjs": fresh.editorjs,
        "previous": {
            "title": fresh.previous_title,
            "summary": fresh.previous_summary,
            "editorjs": fresh.previous_editorjs,
        },
        "version": fresh.version,
        "thread_id": fresh.session.thread_id,
        "updated_by": fresh.updated_by,
        "updated_at": fresh.updated_at.isoformat(),
        "diff_headers": _diff_simple_headers(fresh.editorjs or {}, fresh.previous_editorjs or {}),
    }

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class WikipediaInput(BaseModel):
    query: str = Field(..., description="Topic to summarize in 1–2 sentences")

class SlideInput(BaseModel):
    prompt: Optional[str] = Field(None, description="Natural-language request for the deck")
    title: Optional[str] = Field(None, description="Deck title (optional)")
    summary: Optional[str] = Field(None, description="Short abstract (optional)")
    editorjs: Optional[EditorJS | List[Dict[str, Any]]] = Field(None, description="Editor.js doc or blocks[]")
    ai_enrich: bool = Field(True, description="If true, expand sections with the LLM")
    max_sections: int = Field(6, ge=1, le=16, description="Max sections when enriching")

    @field_validator("editorjs")
    @classmethod
    def _validate_editorjs(cls, v):
        if v is None:
            return v
        ej = normalize_editorjs(v)
        if not ej:
            raise ValueError("editorjs must contain non-empty blocks")
        return ej

class EmotionInput(BaseModel):
    text: str = Field(..., description="Text to classify into a single dominant emotion.")

class FetchSlidesInput(BaseModel):
    pass

# --------------------------------------------------------------------------------------
# Tools (clean outputs; always “latest” and explicit meta)
# --------------------------------------------------------------------------------------
def _meta(status: str, *, session_id: Optional[int]) -> Dict[str, Any]:
    return {
        "status": status,
        "fetched_at": _now_iso(),
        "db": DB_PRIMARY,
        "session_id": session_id,
    }

@tool("search_wikipedia", args_schema=WikipediaInput, description="Brief 1–2 sentence summary from Wikipedia.")
async def search_wikipedia(query: str) -> Dict[str, Any]:
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

    q = (query or "").strip()
    if not q:
        return {"status": "failed", "error": "empty_query"}

    try:
        hits, summary = await _search_and_summary(q)
        if not (hits and summary):
            return {"status": "failed", "error": "not_found", "meta": {"query": q}}
        return {"status": "ok", "data": {"title": hits[0], "summary": summary}}
    except Exception as e:
        log.exception("[tool:wikipedia] error")
        return {"status": "failed", "error": str(e)}

@tool("fetch_latest_slides", args_schema=FetchSlidesInput, description="Fetch the freshest slides snapshot from the primary DB.")
async def fetch_latest_slides() -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "error": "missing_session_context"}

    try:
        snap = await _db_fetch_latest_slides_snapshot(int(session_id))
        if not snap:
            return {"status": "not_found", "meta": _meta("not_found", session_id=session_id)}
        return {"status": "ok", "slides": snap, "meta": _meta("ok", session_id=session_id)}
    except Exception as e:
        log.exception("[tool:fetch_latest_slides] db error")
        return {"status": "failed", "error": str(e), "meta": _meta("failed", session_id=session_id)}

@tool(
    "generate_or_update_slides",
    args_schema=SlideInput,
    description="Create/update slides. Infers missing fields from 'prompt', persists, and returns the freshly re-read snapshot.",
)
async def generate_or_update_slides(
    prompt: Optional[str] = None,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    editorjs: Optional[EditorJS] = None,
    ai_enrich: bool = True,
    max_sections: int = 6,
) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "error": "missing_session_context"}

    # Normalize
    prompt_clean  = (prompt or "").strip()
    title_clean   = (title or "").strip()
    summary_clean = (summary or "").strip()
    ej = normalize_editorjs(editorjs) if editorjs is not None else None

    # Synthesize from prompt if empty
    if not (title_clean or summary_clean or (ej and ej.get("blocks"))):
        outline = await _ai_outline_from_prompt(prompt=prompt_clean, max_sections=max_sections) if prompt_clean else None
        if outline:
            title_clean   = outline.get("title") or title_clean
            summary_clean = outline.get("summary") or summary_clean
            ej = outline.get("editorjs") or ej

    # Fallback minimal deck
    if not (ej and ej.get("blocks")):
        title_clean = title_clean or (prompt_clean[:80] if prompt_clean else "Untitled Deck")
        ej = minimal_editorjs(title_clean, summary_clean)

    # Optional enrichment from the same prompt
    if ai_enrich and prompt_clean:
        outline2 = await _ai_outline_from_prompt(prompt=prompt_clean, max_sections=max_sections)
        if outline2 and (outline2.get("editorjs") or {}).get("blocks"):
            blocks = outline2["editorjs"]["blocks"]
            blocks = _ensure_single_top_header(blocks, title_clean or outline2.get("title"))
            ej = normalize_editorjs({"time": _now_ms(), "version": "2.x", "blocks": blocks}) or ej
            title_clean = title_clean or outline2.get("title") or title_clean
            summary_clean = summary_clean or outline2.get("summary") or summary_clean

    # Enforce single top header
    if ej and ej.get("blocks"):
        ej = normalize_editorjs({"time": _now_ms(), "version": "2.x", "blocks": _ensure_single_top_header(ej["blocks"], title_clean)}) or ej

    # Persist and re-read to guarantee "latest"
    try:
        saved = await _db_rotate_and_save_slides(int(session_id), {
            "title": title_clean or "Untitled Deck",
            "summary": summary_clean or "",
            "editorjs": ej or minimal_editorjs(title_clean or "Untitled Deck", summary_clean or ""),
        }, updated_by="tool:slides")

        # Fresh snapshot (explicit)
        snap = await _db_fetch_latest_slides_snapshot(int(session_id))
        snap = snap or saved  # should never be None, but be defensive

        return {"status": "ok", "slides": snap, "meta": _meta("ok", session_id=session_id)}
    except Exception as e:
        log.exception("[tool:generate_or_update_slides] persist error")
        return {"status": "failed", "error": str(e), "meta": _meta("failed", session_id=session_id)}

@tool(
    "emotion_analyze",
    args_schema=EmotionInput,
    description="Classify a single dominant emotion (Joy/Anger/Sadness/Surprise) with intensity 1..3.",
)
async def emotion_analyze(text: str) -> Dict[str, Any]:
    model_name = "gpt-4o-mini"
    llm = ChatOpenAI(model=model_name, temperature=0.0, timeout=8)
    sys = (
        "Return a SINGLE dominant emotion.\n"
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

# --------------------------------------------------------------------------------------
# Registry (exportable to the router)
# --------------------------------------------------------------------------------------
AGENT_TOOLS: Dict[str, Any] = {
    "search_wikipedia": search_wikipedia,
    "fetch_latest_slides": fetch_latest_slides,
    "generate_or_update_slides": generate_or_update_slides,
    "emotion_analyze": emotion_analyze,
}
TOOLS_SCHEMA = [convert_to_openai_tool(t) for t in AGENT_TOOLS.values()]
