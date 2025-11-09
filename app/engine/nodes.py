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
    build_knowledge_answer_context,
    build_knowledge_list_context,   # NEW
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
    emotion_source: str = os.getenv("EMOTION_SOURCE", "query")  # "query" | "none"

S = Settings()

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
    emotion_ready_event: asyncio.Event

    slides_latest: Dict[str, Any]
    slides_tool_used: bool

    # Knowledge
    knowledge_result: Dict[str, Any]
    knowledge_tool_used: bool

    response: str
    emotion_final: Dict[str, Any]

    viseme_frames: List[List[float]]
    viseme_chunks: List[Dict[str, Any]]

    turn_slides_context: str
    turn_knowledge_context: str
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
    state["knowledge_result"] = {}
    state["knowledge_tool_used"] = False

    state["emotion_final"] = {}
    state["viseme_frames"] = []
    state["viseme_chunks"] = []
    state["turn_slides_context"] = ""
    state["turn_knowledge_context"] = ""

    state["meta"] = {}
    state["timings"] = {"prepare_ms": elapsed_ms(t0)}
    state["text_finalized_event"] = asyncio.Event()
    state["slides_ready_event"] = asyncio.Event()
    state["emotion_ready_event"] = asyncio.Event()
    return state

# =============================================================================
# Tools router/executor
# =============================================================================
async def _tool_router_and_stream(state: QAState):
    """
    Router decides tool calls based on explicit instructions.
    - We stream slides for ANY slides action or slides returned by knowledge.
    - We DO NOT stream knowledge; we only inject it as grounding into the text model.
    """
    router = ChatOpenAI(model=(state.get("model") or "gpt-4o-mini"), temperature=S.router_temperature)

    # Provide full context to tools (slides & knowledge need user/agent)
    set_tool_context({
        "session_id": state["session"].id,
        "bot_id": str(state["agent"].bot_id),
        "agent_id": state["agent"].id,
        "user_id": state["session"].user_id,
    })

    # Compact recent history for routing context
    recent_hist: List[str] = []
    for m in (state["base_msgs"] or [])[-6:]:
        label = "User" if isinstance(m, HumanMessage) else "Assistant"
        recent_hist.append(f"{label}: {getattr(m, 'content', '')[:500]}")
    hist_text = "\n".join(recent_hist)

    planner_msgs: List[BaseMessage] = [
        SystemMessage(content=build_router_system_prompt()),
        SystemMessage(content=("Recent context:\n" + hist_text) if hist_text else "Recent context: (none)"),
        state["user_msg"],
    ]

    tool_calls: List[Dict[str, Any]] = []
    try:
        ai = await router.ainvoke(planner_msgs, tools=TOOLS_SCHEMA, tool_choice="auto")
        tool_calls = (getattr(ai, "tool_calls", []) or [])[: S.max_tool_calls]
        log.debug("[nodes:router] tool_calls=%s", [c.get("name") for c in tool_calls])
    except Exception:
        log.exception("[nodes:router] failed")
        tool_calls = []

    async def _exec(name: str, args: Dict[str, Any]):
        impl = AGENT_TOOLS.get(name)
        if not impl:
            log.warning("[nodes:tool] unknown tool: %s", name)
            return
        try:
            res = await impl.ainvoke(args)
            log.debug("[nodes:tool] %s -> status=%s", name, (res or {}).get("status"))
        except Exception:
            log.exception("[nodes:tool] %s call failed", name)
            return

        # Slides-family: capture snapshot for context AND STREAM ALWAYS
        if name in ("generate_or_update_slides", "fetch_latest_slides", "revert_slides"):
            state["slides_tool_used"] = True
            snap = (res or {}).get("slides") or {}
            if snap:
                state["slides_latest"] = snap
                state["turn_slides_context"] = build_slides_context(snap)
            if state.get("queue"):
                await state["queue"].put({"type": "slides_response", "slides": snap})
                await state["queue"].put({"type": "slides_done"})

        # Knowledge answer: capture grounding (no streaming)
        if name == "answer_from_knowledge":
            state["knowledge_tool_used"] = True
            kres = res or {}
            state["knowledge_result"] = kres
            state["turn_knowledge_context"] = build_knowledge_answer_context(kres)

            # If knowledge tool created/updated slides, mirror slide streaming
            snap = (kres or {}).get("slides") or {}
            if snap:
                state["slides_tool_used"] = True
                state["slides_latest"] = snap
                state["turn_slides_context"] = build_slides_context(snap)
                if state.get("queue"):
                    await state["queue"].put({"type": "slides_response", "slides": snap})
                    await state["queue"].put({"type": "slides_done"})

        # Knowledge listing: feed as grounding (no streaming)
        if name == "list_knowledge":
            state["knowledge_tool_used"] = True
            assets = (res or {}).get("assets") or []
            state["turn_knowledge_context"] = build_knowledge_list_context(assets)

    for c in tool_calls:
        await _exec((c.get("name") or "").strip(), dict(c.get("args") or {}))

    # Unblock text even if the router chose no slides actions
    state["slides_ready_event"].set()

# =============================================================================
# Emotion (emit once BEFORE any text tokens)
# =============================================================================
async def _emit_emotion_before_text(state: QAState):
    try:
        if S.emotion_source == "none":
            emo = {"name": "Joy", "intensity": 1}
        else:
            tool = AGENT_TOOLS.get("emotion_analyze")
            raw = getattr(state.get("user_msg"), "content", None) or (state.get("query") or "")
            text = (raw or "").strip()
            if not tool or not text:
                emo = {"name": "Joy", "intensity": 1}
            else:
                res = await tool.ainvoke({"text": text})
                name = str((res or {}).get("name", "Joy")).title()
                intensity_val = (res or {}).get("intensity", 1)
                try:
                    intensity = int(intensity_val)
                except Exception:
                    intensity = 1
                if name not in ("Joy", "Anger", "Sadness", "Surprise"):
                    name = "Joy"
                intensity = max(1, min(3, intensity))
                emo = {"name": name, "intensity": intensity}

        state["emotion_final"] = emo
        if state.get("queue"):
            await state["queue"].put({"type": "emotion", "emotion": emo})
    except Exception:
        log.exception("[nodes:emotion] pre-text emotion failed")
        state["emotion_final"] = {"name": "Joy", "intensity": 1}
    finally:
        state["emotion_ready_event"].set()

# =============================================================================
# Text streaming + sentence-level audio (MP3 + ARKit15 visemes)
# =============================================================================
async def _stream_text_with_audio(state: QAState):
    # gate on tools + emotion
    await asyncio.gather(state["slides_ready_event"].wait(), state["emotion_ready_event"].wait())

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
        msgs.append(SystemMessage(content="SLIDES CONTEXT:\n" + state["turn_slides_context"]))

    if state["turn_knowledge_context"]:
        # include knowledge grounding as system info for the text LLM (no separate streaming)
        msgs.append(SystemMessage(content=state["turn_knowledge_context"]))

    msgs.extend(state["base_msgs"])
    msgs.append(state["user_msg"])

    buf, final_text = "", ""
    voice = getattr(state["agent"], "voice", None)
    voice_service = (getattr(voice, "service", "") or "").strip() if voice else ""
    voice_id = (getattr(voice, "voice_id", "") or "").strip() if voice else ""

    # NEW: keep continuous viseme timeline metadata
    chunk_index = 0
    offset_ms_accum = 0  # absolute offset for each chunk’s visemes

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
                    audio, v = await synthesize_tts_async(cleaned, voice_service, voice_id)

                    # Compute absolute offset & annotate chunk
                    dur_ms = int(v.get("duration_ms") or 0)
                    chunk_payload = {
                        "type": "audio_response",
                        "audio": base64.b64encode(audio).decode("utf-8"),
                        "audio_format": "mp3",            # explicit
                        "chunk_index": chunk_index,       # ordering
                        "offset_ms": offset_ms_accum,     # absolute timeline
                        **v,                               # includes viseme, viseme_times, viseme_fps, frame_ms, etc.
                    }

                    await state["queue"].put(chunk_payload)

                    # Maintain rolling preview of frames
                    frames = v.get("viseme") or []
                    if isinstance(frames, list):
                        state["viseme_frames"].extend(frames)
                        if len(state["viseme_frames"]) > S.viseme_max_frames_to_store:
                            state["viseme_frames"] = state["viseme_frames"][-S.viseme_max_frames_to_store:]

                    # Persist chunk-level viseme metadata
                    state["viseme_chunks"].append({**v, "chunk_index": chunk_index, "offset_ms": offset_ms_accum})

                    # advance timeline
                    offset_ms_accum += max(0, dur_ms)
                    chunk_index += 1
                except Exception:
                    log.exception("[nodes:tts] sentence TTS failed")

    # Tail flush
    tail = buf.strip()
    if tail and voice_service and voice_id and state.get("queue"):
        try:
            cleaned = await asyncio.to_thread(sanitize_and_verbalize_text, tail, "en_GB", SanitizeOptions())
            audio, v = await synthesize_tts_async(cleaned, voice_service, voice_id)

            dur_ms = int(v.get("duration_ms") or 0)
            chunk_payload = {
                "type": "audio_response",
                "audio": base64.b64encode(audio).decode("utf-8"),
                "audio_format": "mp3",
                "chunk_index": chunk_index,
                "offset_ms": offset_ms_accum,
                **v,
            }
            await state["queue"].put(chunk_payload)

            frames = v.get("viseme") or []
            if isinstance(frames, list):
                state["viseme_frames"].extend(frames)
                if len(state["viseme_frames"]) > S.viseme_max_frames_to_store:
                    state["viseme_frames"] = state["viseme_frames"][-S.viseme_max_frames_to_store:]

            state["viseme_chunks"].append({**v, "chunk_index": chunk_index, "offset_ms": offset_ms_accum})
            offset_ms_accum += max(0, dur_ms)
            chunk_index += 1
        except Exception:
            log.exception("[nodes:tts] tail TTS failed")

    state["response"] = final_text
    state["text_finalized_event"].set()

# =============================================================================
# Run + finalize
# =============================================================================
async def n_run(state: QAState) -> QAState:
    t0 = time.perf_counter()
    await asyncio.gather(
        _tool_router_and_stream(state),
        _emit_emotion_before_text(state),
        _stream_text_with_audio(state),
    )
    state["timings"]["run_ms"] = elapsed_ms(t0)
    return state

async def n_finalize_and_persist(state: QAState) -> QAState:
    timings = dict(state.get("timings") or {})
    meta = dict(state.get("meta") or {})
    frame_ms_meta: Optional[int] = None
    viseme_chunks = state.get("viseme_chunks") or []
    viseme_frames = state.get("viseme_frames") or []

    if viseme_chunks:
        try:
            frame_ms_meta = int(viseme_chunks[0].get("frame_ms") or 0) or None
        except Exception:
            frame_ms_meta = None

    meta.update({
        "schema": "engine.v10",
        "bot_id": state.get("bot_id"),
        "thread_id": state.get("thread_id"),
        "model": state.get("model") or "gpt-4o-mini",
        "timings_ms": timings,
        "response_len": len(state.get("response") or ""),
        "tools": list(AGENT_TOOLS.keys()),
        "frame_ms": frame_ms_meta,
        "slides_tool_used": bool(state.get("slides_tool_used")),
        "knowledge_tool_used": bool(state.get("knowledge_tool_used")),
        # NEW: helpful counters
        "viseme_chunks_count": len(viseme_chunks),
        "viseme_frames_count": len(viseme_frames),
    })

    viseme_json: Dict[str, Any] = {}
    if viseme_chunks or viseme_frames:
        viseme_json = {
            "chunks": viseme_chunks,
            "frames_preview": viseme_frames,
        }

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
