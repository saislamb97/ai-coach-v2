from __future__ import annotations

import logging
from typing import Any, Dict

from asgiref.sync import sync_to_async
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .utils import _llm

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Emotion
# --------------------------------------------------------------------------------------
class EmotionInput(BaseModel):
    text: str = Field(..., description="Text to classify into a single dominant emotion.")

@tool(
    "emotion_detect",
    args_schema=EmotionInput,
    description="Classify a SINGLE dominant emotion (Joy, Anger, Sadness, Surprise) with intensity 1..3.",
)
async def emotion_detect(text: str) -> Dict[str, Any]:
    llm = _llm(temperature=0.0, timeout=8)
    sys = (
        "Return a SINGLE dominant emotion.\n"
        "Allowed names: Joy, Anger, Sadness, Surprise. Intensity: integer 1..3.\n"
        'Return STRICT JSON ONLY like: {"name":"Joy","intensity":2}'
    )
    user = (text or "").strip()[:8000]
    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user)])
        raw = (getattr(msg, "content", "") or "").strip()
        import json
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
# Wikipedia (adapted to unified return shape)
# --------------------------------------------------------------------------------------
class WikipediaInput(BaseModel):
    query: str = Field(..., description="Topic to look up on Wikipedia")

@tool("search_wikipedia", args_schema=WikipediaInput)
async def search_wikipedia(query: str) -> Dict[str, Any]:
    """
    Return a brief 1–2 sentence summary for a topic from Wikipedia.
    - Quiet for expected failures
    - Searches first; returns suggestions if nothing matches
    """
    q = (query or "").strip()
    if not q:
        return {"status": "failed", "summary": "Empty query.", "data": {}}

    try:
        import wikipedia  # type: ignore
        from wikipedia.exceptions import DisambiguationError, PageError  # type: ignore
    except Exception as e:
        return {"status": "failed", "summary": f"Wikipedia library not available: {e}", "data": {}}

    try:
        hits = await sync_to_async(wikipedia.search)(q, results=5)
        if not hits:
            return {"status": "not_found", "summary": "No results found on Wikipedia.", "data": {"query": q, "suggestions": []}}

        # Prefer exact/starts-with match, else top result
        lowered = q.lower()
        title = next((h for h in hits if h.lower() == lowered or h.lower().startswith(lowered)), hits[0])

        summary = await sync_to_async(wikipedia.summary)(title, sentences=2)

        return {
            "status": "ok",
            "summary": f"{title}: {summary}",
            "data": {"query": q, "title": title, "summary": summary, "suggestions": hits},
        }
    except DisambiguationError as e:  # type: ignore[name-defined]
        opts = (e.options or [])[:8]
        return {"status": "failed", "summary": f"Ambiguous query. Try one of: {', '.join(opts[:5])}", "data": {"query": q, "suggestions": opts}}
    except PageError:  # type: ignore[name-defined]
        return {"status": "not_found", "summary": "No Wikipedia page found.", "data": {"query": q}}
    except Exception as e:
        log.exception("[search_wikipedia] unexpected error for %r", q)
        return {"status": "failed", "summary": f"Wikipedia Error: {e}", "data": {"query": q}}
