from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import TypedDict, List, Dict, Any, Optional, Tuple

from asgiref.sync import sync_to_async
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from agent.models import Agent
from memory.models import Session

from .tokens import TokenLimits
from .utils import extract_sentences, build_text_system_prompt
from .tools import AGENT_TOOLS, TOOLS_SCHEMA, set_tool_context

from .tts_stt import synthesize_tts_async
from stream.text_sanitize import SanitizeOptions, sanitize_and_verbalize_text

log = logging.getLogger(__name__)

# =============================================================================
# Settings / constants
# =============================================================================
@dataclass(frozen=True)
class Settings:
    text_temperature: float = 0.3
    router_temperature: float = 0.1
    max_history_pairs: int = 50

    tool_timeout_s: int = 20
    poll_interval_s: float = 0.5
    max_tool_calls: int = 6

    assembly_reserved_tokens: int = 3000

    emotion_model: str = "gpt-4o-mini"
    emotion_temperature: float = 0.0
    emotion_timeout_s: float = 6.0
    emotion_retries: int = 2

    viseme_max_frames_to_store: int = 240  # last 24s @ 10fps

S = Settings()

# =============================================================================
# Shared State
# =============================================================================
class QAState(TypedDict, total=False):
    bot_id: str
    thread_id: str
    query: str

    queue: asyncio.Queue | None
    limits: TokenLimits
    model: str

    agent: Agent
    session: Session

    base_msgs: List[BaseMessage]
    user_msg: HumanMessage

    tools_started: List[Dict[str, Any]]
    slides_latest: Dict[str, Any]
    response: str

    emotion_state: Dict[str, Any]
    viseme_state: Dict[str, Any]

    slides_outline_text: str

    meta: Dict[str, Any]
    timings: Dict[str, int]

# =============================================================================
# DB helpers
# =============================================================================
def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)

@sync_to_async
def _get_agent_and_session(bot_id: str, thread_id: str) -> tuple[Agent, Session]:
    agent = (
        Agent.objects
        .filter(bot_id=bot_id, is_active=True)
        .select_related("voice", "user")
        .first()
    )
    if not agent:
        raise RuntimeError("Agent not found")
    session = (
        Session.objects
        .filter(thread_id=thread_id, agent_id=agent.id, is_active=True)
        .select_related("user", "agent")
        .first()
    )
    if not session:
        raise RuntimeError("Session not found for thread_id")
    return agent, session

@sync_to_async
def _fetch_history_pairs(session_id: int, limit: int) -> List[Tuple[str, str]]:
    from memory.models import Chat
    qs = Chat.objects.filter(session_id=session_id).order_by("-created_at")[:limit]
    items = list(qs)[::-1]
    return [(c.query or "", c.response or "") for c in items]

@sync_to_async
def _persist_chat(
    *,
    session: Session,
    query: str,
    response: str,
    emotion: Dict[str, Any] | None,
    viseme: Dict[str, Any] | None,
    meta: Dict[str, Any] | None
) -> None:
    from memory.models import Chat
    Chat.objects.create(
        session=session,
        query=query,
        response=response,
        emotion=emotion or {},
        viseme=viseme or {},
        meta=meta or {},
    )

@sync_to_async
def _upsert_slides(session: Session, slides_payload: Dict[str, Any]) -> Dict[str, Any]:
    from memory.models import Slides
    if not isinstance(slides_payload, dict) or not isinstance(slides_payload.get("editorjs"), dict):
        return {}
    s, _ = Slides.objects.get_or_create(session=session)
    s.title = str(slides_payload.get("title") or "")
    s.summary = str(slides_payload.get("summary") or "")
    s.editorjs = slides_payload["editorjs"]
    s.save(update_fields=["title", "summary", "editorjs", "updated_at"])
    return {
        "title": s.title,
        "summary": s.summary,
        "editorjs": s.editorjs,
        "updated_at": s.updated_at,
        "thread_id": session.thread_id,
    }

@sync_to_async
def _get_latest_slides_outline(session: Session) -> str:
    from memory.models import Slides
    s = Slides.objects.filter(session=session).first()
    if not s or not isinstance(getattr(s, "editorjs", None), dict):
        return ""
    ej = s.editorjs or {}
    blocks = ej.get("blocks") or []
    headers: List[str] = []
    for b in blocks:
        if (b.get("type") or "").lower() == "header":
            t = (b.get("data", {}).get("text") or "").strip()
            if t:
                headers.append(t)
        if len(headers) >= 12:
            break
    title = (s.title or "").strip()
    summary = (s.summary or "").strip()
    lines = []
    if title:
        lines.append(f"Title: {title}")
    if summary:
        lines.append(f"Summary: {summary}")
    if headers:
        lines.append("Sections:")
        lines.extend([f"- {h}" for h in headers])
    return "\n".join(lines)

# =============================================================================
# Emotion & Viseme helpers
# =============================================================================
def _emotion_state_init() -> Dict[str, Any]:
    return {
        "count": 0,
        "sum": {"Joy": 0, "Anger": 0, "Sadness": 0},
        "max": {"Joy": 1, "Anger": 1, "Sadness": 1},
        "last": {"Joy": 1, "Anger": 1, "Sadness": 1},
        "items": [],
    }

def _emotion_state_update(state: Dict[str, Any], emo: Dict[str, Any]) -> None:
    try:
        items = emo.get("items") or []
        by = {i.get("name"): int(i.get("intensity", 1)) for i in items if isinstance(i, dict) and "name" in i}
        for k in ("Joy", "Anger", "Sadness"):
            v = max(1, min(3, int(by.get(k, state["last"].get(k, 1)))))
            state["sum"][k] = int(state["sum"][k]) + v
            state["max"][k] = max(int(state["max"][k]), v)
            state["last"][k] = v
        state["count"] = int(state["count"]) + 1
        state["items"].append({"Joy": state["last"]["Joy"], "Anger": state["last"]["Anger"], "Sadness": state["last"]["Sadness"]})
        if len(state["items"]) > 20:
            state["items"].pop(0)
    except Exception:
        pass

def _emotion_state_finalize(state: Dict[str, Any]) -> Dict[str, Any]:
    c = max(1, int(state.get("count", 0)) or 1)
    avg = {
        "Joy": round(state["sum"]["Joy"] / c),
        "Anger": round(state["sum"]["Anger"] / c),
        "Sadness": round(state["sum"]["Sadness"] / c),
    }
    for k in avg:
        avg[k] = min(3, max(1, int(avg[k])))
    return {
        "items": [
            {"name": "Joy", "intensity": avg["Joy"]},
            {"name": "Anger", "intensity": avg["Anger"]},
            {"name": "Sadness", "intensity": avg["Sadness"]},
        ],
        "stats": {
            "count": c,
            "avg": avg,
            "max": {
                "Joy": int(state["max"]["Joy"]),
                "Anger": int(state["max"]["Anger"]),
                "Sadness": int(state["max"]["Sadness"]),
            },
            "last": {
                "Joy": int(state["last"]["Joy"]),
                "Anger": int(state["last"]["Anger"]),
                "Sadness": int(state["last"]["Sadness"]),
            },
        },
    }

def _viseme_state_init() -> Dict[str, Any]:
    return {"segments": 0, "frames_total": 0, "last_frames": []}

def _viseme_state_update(state: Dict[str, Any], frames: List[List[float]]) -> None:
    try:
        n = len(frames or [])
        state["segments"] = int(state.get("segments", 0)) + 1
        state["frames_total"] = int(state.get("frames_total", 0)) + n
        state["last_frames"] = frames[-S.viseme_max_frames_to_store:] if n > S.viseme_max_frames_to_store else frames
    except Exception:
        pass

def _viseme_state_finalize(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "segments": int(state.get("segments", 0)),
        "frames_total": int(state.get("frames_total", 0)),
        "last_frames": state.get("last_frames", []),
    }

# =============================================================================
# Prepare
# =============================================================================
async def n_prepare(state: QAState) -> QAState:
    t0 = time.perf_counter()
    agent, session = await _get_agent_and_session(state["bot_id"], state["thread_id"])
    state["agent"] = agent
    state["session"] = session

    pairs = await _fetch_history_pairs(session.id, limit=S.max_history_pairs)
    base_msgs: List[BaseMessage] = []
    for q, a in pairs:
        if q: base_msgs.append(HumanMessage(content=q))
        if a: base_msgs.append(AIMessage(content=a))
    state["base_msgs"] = base_msgs
    state["user_msg"] = HumanMessage(content=(state.get("query") or "").strip())

    try:
        state["slides_outline_text"] = await _get_latest_slides_outline(session)
    except Exception:
        state["slides_outline_text"] = ""

    state["tools_started"] = []
    state["slides_latest"] = {}
    state["emotion_state"] = _emotion_state_init()
    state["viseme_state"] = _viseme_state_init()
    state["meta"] = {}
    state["timings"] = {"prepare_ms": _elapsed_ms(t0)}
    return state

# =============================================================================
# Tools planner/executor (synchronous deck generation; no polling)
# =============================================================================
async def _tool_router_and_stream(state: QAState):
    """
    Runs independently of the text stream.
    The LLM decides when/if to call tools. Slide tool is synchronous.
    """
    router = ChatOpenAI(model=(state.get("model") or "gpt-4o-mini"), temperature=S.router_temperature)

    set_tool_context({
        "session_id": state["session"].id,
        "bot_id": str(state["agent"].bot_id),
        "agent_id": state["agent"].id,
    })

    planner_msgs: List[BaseMessage] = [
        SystemMessage(content=(
            "You can call tools to fetch facts or manage slides for THIS session.\n"
            "## Tools available\n"
            "1) search_wikipedia(query: string)\n"
            "   - Returns a brief 1–2 sentence summary.\n"
            "2) generate_or_update_slides(editorjs?, [title], [summary], ai_enrich, max_sections)\n"
            "   - Create or update a deck in Editor.js synchronously.\n"
            "## Mandates\n"
            "- NEVER include slide JSON in assistant text.\n"
            "- If the user asks for a presentation, slides, deck, PPT, or Keynote, "
            "  you MUST call generate_or_update_slides with ai_enrich=true.\n"
            "- If no editorjs provided, create a minimal deck from title/summary."
        )),
        *state["base_msgs"][-6:],
        state["user_msg"],
    ]

    tool_calls: List[Dict[str, Any]] = []
    try:
        ai = await router.ainvoke(planner_msgs, tools=TOOLS_SCHEMA, tool_choice="auto")
        tool_calls = (getattr(ai, "tool_calls", []) or [])[: S.max_tool_calls]
    except Exception:
        log.exception("[nodes:tool_router] router failed")
        tool_calls = []

    async def _exec_call(call: Dict[str, Any]):
        name = (call.get("name") or "").strip()
        args = dict(call.get("args") or {})
        impl = AGENT_TOOLS.get(name)
        if not impl:
            return
        try:
            if name == "generate_or_update_slides":
                args.setdefault("ai_enrich", True)
                args.setdefault("max_sections", 6)
            res = await impl.ainvoke(args)
        except Exception:
            log.exception("[nodes:tool_call] %s failed", name)
            return

        if name == "generate_or_update_slides":
            # Tool returns the payload directly (no status/job_id).
            data = res.get("data") if isinstance(res, dict) and "data" in res else res
            payload = {
                "title": (data or {}).get("title") or "",
                "summary": (data or {}).get("summary") or "",
                "editorjs": (data or {}).get("editorjs") or {},
            }
            state["slides_latest"] = payload
            if state.get("queue"):
                await state["queue"].put({"type": "slides_response", "slides": payload})
                await state["queue"].put({"type": "slides_done"})

    # Execute planned calls
    await asyncio.gather(*(_exec_call(c) for c in tool_calls))

    # Fallback: if the user clearly asked for slides and nothing was scheduled, run the tool once
    if not tool_calls:
        ask = (state.get("user_msg") or HumanMessage(content="")).content.lower()
        if any(k in ask for k in ("slide", "slides", "deck", "presentation", "ppt", "powerpoint", "keynote")):
            await _exec_call({
                "name": "generate_or_update_slides",
                "args": {
                    "title": (state["agent"].name or "Presentation")[:80],
                    "summary": (state.get("user_msg").content or "")[:160],
                    "ai_enrich": True,
                    "max_sections": 6,
                }
            })

# =============================================================================
# Text streaming with sentence-level emotion + TTS (in-node)
# =============================================================================
async def _stream_text_with_audio(state: QAState):
    model_name = state.get("model") or "gpt-4o-mini"
    llm = ChatOpenAI(model=model_name, temperature=S.text_temperature, streaming=True)

    utext = (state.get("user_msg") or HumanMessage(content="")).content.lower()
    wants_slide_qa = any(k in utext for k in ("summarize", "summary", "explain", "suggest", "improve", "revise", "outline"))
    slide_context = (state.get("slides_outline_text") or "").strip() if wants_slide_qa else ""
    slide_context_msg: List[BaseMessage] = []
    if slide_context:
        slide_context_msg = [
            SystemMessage(content=(
                "You have a compact outline of the CURRENT deck (no JSON). "
                "Use it to answer questions about the slides. Do NOT produce slide JSON.\n\n"
                + slide_context
            ))
        ]

    base: List[BaseMessage] = [
        SystemMessage(content=build_text_system_prompt(
            bot_name=state["agent"].name or "Assistant",
            bot_id=str(state["agent"].bot_id),
            instruction=(state["agent"].persona or "").strip(),
        )),
        *slide_context_msg,
        *state["base_msgs"],
        state["user_msg"],
    ]

    buf, final_text = "", ""

    voice = getattr(state["agent"], "voice", None)
    voice_service = (getattr(voice, "service", "") or "").strip() if voice else ""
    voice_id = (getattr(voice, "voice_id", "") or "").strip() if voice else ""

    async for chunk in llm.astream(base, tools=[], tool_choice="none"):
        token = (getattr(chunk, "content", "") or "")
        if not token:
            continue

        final_text += token
        buf += token

        if state.get("queue"):
            await state["queue"].put({"type": "text_token", "token": token})

        sents, remainder = extract_sentences(buf)
        buf = remainder

        for s in sents:
            emo = await _classify_emotion(sentence=s, state=state)
            _emotion_state_update(state["emotion_state"], emo)

            if state.get("queue"):
                await state["queue"].put({"type": "text_sentence", "text": s, "emotion": emo})

            if voice_service and voice_id and state.get("queue"):
                try:
                    cleaned = await asyncio.to_thread(sanitize_and_verbalize_text, s, "en_GB", SanitizeOptions())
                    audio, visemes = await synthesize_tts_async(cleaned, voice_service, voice_id)
                    import base64
                    b64 = base64.b64encode(audio).decode("utf-8")
                    await state["queue"].put({"type": "audio_response", "audio": b64, "viseme": visemes})
                    _viseme_state_update(state["viseme_state"], visemes)
                except Exception:
                    log.exception("[nodes:tts] sentence TTS failed")

    tail = buf.strip()
    if tail:
        emo = await _classify_emotion(sentence=tail, state=state)
        _emotion_state_update(state["emotion_state"], emo)
        if state.get("queue"):
            await state["queue"].put({"type": "text_sentence", "text": tail, "emotion": emo})
        if voice_service and voice_id and state.get("queue"):
            try:
                cleaned = await asyncio.to_thread(sanitize_and_verbalize_text, tail, "en_GB", SanitizeOptions())
                audio, visemes = await synthesize_tts_async(cleaned, voice_service, voice_id)
                import base64
                b64 = base64.b64encode(audio).decode("utf-8")
                await state["queue"].put({"type": "audio_response", "audio": b64, "viseme": visemes})
                _viseme_state_update(state["viseme_state"], visemes)
            except Exception:
                log.exception("[nodes:tts] tail TTS failed")

    state["response"] = final_text

# =============================================================================
# Emotion classifier
# =============================================================================
async def _classify_emotion(*, sentence: str, state: QAState) -> Dict[str, Any]:
    sys = (
        "You are an emotion classifier. Return STRICT JSON ONLY.\n"
        "There are 3 emotions: Joy, Anger, Sadness. Each must have intensity 1..3.\n"
        'Output: {"items":[{"name":"Joy","intensity":N},{"name":"Anger","intensity":N},{"name":"Sadness","intensity":N}]}\n'
    )
    user = f"# Sentence\n{(sentence or '').strip()}"
    attempt = 0
    while attempt <= S.emotion_retries:
        attempt += 1
        try:
            cls = ChatOpenAI(model=S.emotion_model, temperature=S.emotion_temperature, timeout=S.emotion_timeout_s)
            msg = await cls.ainvoke([SystemMessage(content=sys), HumanMessage(content=user)])
            raw = (getattr(msg, "content", "") or "").strip()
            data = json.loads(raw)
            items = data.get("items") or []
            if not isinstance(items, list) or len(items) != 3:
                raise ValueError("malformed")
            want = {"joy": "Joy", "anger": "Anger", "sadness": "Sadness"}
            norm = {}
            for it in items:
                nm = str(it.get("name") or "").lower().strip()
                iv = int(it.get("intensity", 1))
                iv = 1 if iv < 1 or iv > 3 else iv
                proper = want.get(nm)
                if not proper:
                    raise ValueError("bad name")
                norm[proper] = {"name": proper, "intensity": iv}
            return {"items": [norm.get("Joy", {"name":"Joy","intensity":1}),
                              norm.get("Anger", {"name":"Anger","intensity":1}),
                              norm.get("Sadness", {"name":"Sadness","intensity":1})]}
        except Exception:
            if attempt <= S.emotion_retries:
                continue
            break
    return {"items": [
        {"name": "Joy", "intensity": 1},
        {"name": "Anger", "intensity": 1},
        {"name": "Sadness", "intensity": 1},
    ]}

# =============================================================================
# Run both streams in parallel
# =============================================================================
async def n_run(state: QAState) -> QAState:
    t0 = time.perf_counter()
    await asyncio.gather(
        _stream_text_with_audio(state),
        _tool_router_and_stream(state),
    )
    state["timings"]["run_ms"] = _elapsed_ms(t0)
    return state

# =============================================================================
# Finalize & Persist
# =============================================================================
async def n_finalize_and_persist(state: QAState) -> QAState:
    timings = dict(state.get("timings") or {})
    meta = dict(state.get("meta") or {})
    meta.update({
        "schema": "engine.v3",
        "bot_id": state.get("bot_id"),
        "thread_id": state.get("thread_id"),
        "model": state.get("model") or "gpt-4o-mini",
        "timings_ms": timings,
        "response_len": len(state.get("response") or ""),
        "tools": list(AGENT_TOOLS.keys()),
    })

    final_emotion = _emotion_state_finalize(state.get("emotion_state", _emotion_state_init()))
    final_viseme = _viseme_state_finalize(state.get("viseme_state", _viseme_state_init()))

    try:
        await _persist_chat(
            session=state["session"],
            query=state.get("query") or "",
            response=state.get("response") or "",
            emotion=final_emotion,
            viseme=final_viseme,
            meta=meta,
        )
    except Exception:
        log.exception("[nodes:finalize] persist chat failed")

    try:
        if state.get("slides_latest"):
            await _upsert_slides(state["session"], state["slides_latest"])
    except Exception:
        log.exception("[nodes:finalize] persist slides failed")

    return state
