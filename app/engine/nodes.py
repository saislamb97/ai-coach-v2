from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import TypedDict, List, Dict, Any, Optional, Tuple

from asgiref.sync import sync_to_async
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from agent.models import Agent
from memory.models import Session
from .tokens import TokenLimits
from .tools import AGENT_TOOLS, TOOLS_SCHEMA, set_tool_context
from .tts_stt import synthesize_tts_async
from stream.text_sanitize import SanitizeOptions, sanitize_and_verbalize_text

from .utils import (
    extract_sentences,
    build_text_system_prompt,
    build_router_system_prompt,
    build_slides_context,
    elapsed_ms,
)

log = logging.getLogger(__name__)

# =============================================================================
# Settings / constants
# =============================================================================
@dataclass(frozen=True)
class Settings:
    text_temperature: float = 0.1
    router_temperature: float = 0.1
    max_history_pairs: int = 50
    tool_timeout_s: int = 20
    max_tool_calls: int = 6
    emotion_model: str = "gpt-4o-mini"
    emotion_timeout_s: float = 6.0
    viseme_max_frames_to_store: int = 8000
    vis_fps_env: int = int(os.getenv("VIS_FPS", "88"))

S = Settings()
FRAME_MS = max(10, int(round(1000.0 / max(1, S.vis_fps_env))))

# =============================================================================
# State
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

    text_finalized_event: asyncio.Event
    slides_ready_event: asyncio.Event

    slides_latest: Dict[str, Any]
    slides_tool_used: bool

    response: str
    emotion_final: Dict[str, Any]
    viseme_frames: List[List[float]]

    turn_slides_context: str
    meta: Dict[str, Any]
    timings: Dict[str, int]

# =============================================================================
# DB helpers
# =============================================================================
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
def _persist_chat(*, session: Session, query: str, response: str, emotion: Dict[str, Any] | None, viseme: Dict[str, Any] | None, meta: Dict[str, Any] | None) -> None:
    from memory.models import Chat
    Chat.objects.create(
        session=session,
        query=query,
        response=response,
        emotion=emotion or {},
        viseme=viseme or {},
        meta=meta or {},
    )

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

    state["slides_latest"] = {}
    state["slides_tool_used"] = False
    state["emotion_final"] = {}
    state["viseme_frames"] = []
    state["turn_slides_context"] = ""

    state["meta"] = {}
    state["timings"] = {"prepare_ms": elapsed_ms(t0)}
    state["text_finalized_event"] = asyncio.Event()
    state["slides_ready_event"] = asyncio.Event()
    return state

# =============================================================================
# Tools router/executor
# =============================================================================
async def _tool_router_and_stream(state: QAState):
    """
    Let the LLM decide naturally via the router prompt.
    Any slide writes happen inside the tool (persisted there).
    """
    router = ChatOpenAI(model=(state.get("model") or "gpt-4o-mini"), temperature=S.router_temperature)

    set_tool_context({
        "session_id": state["session"].id,
        "bot_id": str(state["agent"].bot_id),
        "agent_id": state["agent"].id,
    })

    planner_msgs: List[BaseMessage] = [
        SystemMessage(content=build_router_system_prompt()),
        state["user_msg"],  # route off the latest user turn
    ]

    tool_calls: List[Dict[str, Any]] = []
    try:
        ai = await router.ainvoke(planner_msgs, tools=TOOLS_SCHEMA, tool_choice="auto")
        tool_calls = (getattr(ai, "tool_calls", []) or [])[: S.max_tool_calls]
    except Exception:
        log.exception("[nodes:router] failed")
        tool_calls = []

    async def _exec(name: str, args: Dict[str, Any]):
        impl = AGENT_TOOLS.get(name)
        if not impl:
            return
        try:
            res = await impl.ainvoke(args)
        except Exception:
            log.exception("[nodes:tool] %s call failed", name)
            return

        if name in ("generate_or_update_slides", "fetch_latest_slides"):
            state["slides_tool_used"] = True
            snap = (res or {}).get("slides") or {}
            if snap:
                state["slides_latest"] = snap
                state["turn_slides_context"] = build_slides_context(snap)
                if state.get("queue"):
                    await state["queue"].put({"type": "slides_response", "slides": snap})
                    await state["queue"].put({"type": "slides_done"})

    # Execute tool calls in-order (predictable)
    for c in tool_calls:
        await _exec((c.get("name") or "").strip(), dict(c.get("args") or {}))

    # signal done
    state["slides_ready_event"].set()

# =============================================================================
# Text streaming + sentence-level audio
# =============================================================================
async def _stream_text_with_audio(state: QAState):
    # IMPORTANT: wait for tools to finish so slide context is available before any text
    await state["slides_ready_event"].wait()

    model_name = state.get("model") or "gpt-4o-mini"
    llm = ChatOpenAI(model=model_name, temperature=S.text_temperature, streaming=True)

    msgs: List[BaseMessage] = [
        SystemMessage(content=build_text_system_prompt(
            bot_name=state["agent"].name or "Assistant",
            bot_id=str(state["agent"].bot_id),
            instruction=(state["agent"].persona or "").strip(),
        ))
    ]

    if state["turn_slides_context"]:
        msgs.append(SystemMessage(content="Slides context:\n" + state["turn_slides_context"]))

    msgs.extend(state["base_msgs"])
    msgs.append(state["user_msg"])

    buf, final_text = "", ""

    voice = getattr(state["agent"], "voice", None)
    voice_service = (getattr(voice, "service", "") or "").strip() if voice else ""
    voice_id = (getattr(voice, "voice_id", "") or "").strip() if voice else ""

    async for chunk in llm.astream(msgs, tools=[], tool_choice="none"):
        token = (getattr(chunk, "content", "") or "")
        if not token:
            continue

        final_text += token
        buf += token

        if state.get("queue"):
            await state["queue"].put({"type": "text_token", "token": token})

        sents, remainder = extract_sentences(buf)
        buf = remainder

        if voice_service and voice_id and state.get("queue"):
            for s in sents:
                try:
                    cleaned = await asyncio.to_thread(sanitize_and_verbalize_text, s, "en_GB", SanitizeOptions())
                    audio, visemes = await synthesize_tts_async(cleaned, voice_service, voice_id)

                    if visemes and visemes[0] != [0.0]*15:
                        visemes = [[0.0]*15] + visemes
                    if visemes and visemes[-1] != [0.0]*15:
                        visemes = visemes + [[0.0]*15]
                    times = [i * (FRAME_MS / 1000.0) for i in range(len(visemes))]
                    fps = round(1000.0 / FRAME_MS, 3)

                    b64 = base64.b64encode(audio).decode("utf-8")
                    await state["queue"].put({
                        "type": "audio_response",
                        "audio": b64,
                        "viseme": visemes,
                        "viseme_format": "arkit15",
                        "viseme_times": times,
                        "viseme_fps": fps
                    })

                    if isinstance(visemes, list):
                        state["viseme_frames"].extend(visemes)
                        if len(state["viseme_frames"]) > S.viseme_max_frames_to_store:
                            state["viseme_frames"] = state["viseme_frames"][-S.viseme_max_frames_to_store:]
                except Exception:
                    log.exception("[nodes:tts] sentence TTS failed")

    # tail
    tail = buf.strip()
    if tail and voice_service and voice_id and state.get("queue"):
        try:
            cleaned = await asyncio.to_thread(sanitize_and_verbalize_text, tail, "en_GB", SanitizeOptions())
            audio, visemes = await synthesize_tts_async(cleaned, voice_service, voice_id)

            if visemes and visemes[0] != [0.0]*15:
                visemes = [[0.0]*15] + visemes
            if visemes and visemes[-1] != [0.0]*15:
                visemes = visemes + [[0.0]*15]
            times = [i * (FRAME_MS / 1000.0) for i in range(len(visemes))]
            fps = round(1000.0 / FRAME_MS, 3)

            b64 = base64.b64encode(audio).decode("utf-8")
            await state["queue"].put({
                "type": "audio_response",
                "audio": b64,
                "viseme": visemes,
                "viseme_format": "arkit15",
                "viseme_times": times,
                "viseme_fps": fps
            })

            if isinstance(visemes, list):
                state["viseme_frames"].extend(visemes)
                if len(state["viseme_frames"]) > S.viseme_max_frames_to_store:
                    state["viseme_frames"] = state["viseme_frames"][-S.viseme_max_frames_to_store:]
        except Exception:
            log.exception("[nodes:tts] tail TTS failed")

    state["response"] = final_text
    state["text_finalized_event"].set()

# =============================================================================
# Emotion (single result)
# =============================================================================
async def _emit_emotion_once(state: QAState):
    await state["text_finalized_event"].wait()
    text = (state.get("response") or "").strip()
    if not text:
        state["emotion_final"] = {"name": "Joy", "intensity": 1}
        return

    tool = AGENT_TOOLS.get("emotion_analyze")
    if not tool:
        state["emotion_final"] = {"name": "Joy", "intensity": 1}
        return

    try:
        res = await tool.ainvoke({"text": text})
        emo = {
            "name": str(res.get("name", "Joy")).title(),
            "intensity": int(res.get("intensity", 1)),
        }
        state["emotion_final"] = emo
        if state.get("queue"):
            await state["queue"].put({"type": "emotion", "emotion": emo})
    except Exception:
        log.exception("[nodes:emotion] tool failed")
        state["emotion_final"] = {"name": "Joy", "intensity": 1}

# =============================================================================
# Run + finalize
# =============================================================================
async def n_run(state: QAState) -> QAState:
    t0 = time.perf_counter()
    await asyncio.gather(
        _tool_router_and_stream(state),
        _stream_text_with_audio(state),
        _emit_emotion_once(state),
    )
    state["timings"]["run_ms"] = elapsed_ms(t0)
    return state

async def n_finalize_and_persist(state: QAState) -> QAState:
    timings = dict(state.get("timings") or {})
    meta = dict(state.get("meta") or {})
    meta.update({
        "schema": "engine.v8",
        "bot_id": state.get("bot_id"),
        "thread_id": state.get("thread_id"),
        "model": state.get("model") or "gpt-4o-mini",
        "timings_ms": timings,
        "response_len": len(state.get("response") or ""),
        "tools": list(AGENT_TOOLS.keys()),
        "frame_ms": FRAME_MS,
        "slides_tool_used": bool(state.get("slides_tool_used")),
    })

    viseme_json = {}
    if state.get("viseme_frames"):
        viseme_json = {"frame_ms": FRAME_MS, "frames": state["viseme_frames"]}

    try:
        await _persist_chat(
            session=state["session"],
            query=state.get("query") or "",
            response=state.get("response") or "",
            emotion=(state.get("emotion_final") or {}),
            viseme=viseme_json,
            meta=meta,
        )
    except Exception:
        log.exception("[nodes:finalize] persist chat failed")

    return state
