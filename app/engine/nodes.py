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
from engine.tools.utils import deck_facts_from_snapshot
from engine.tokens import TokenLimits
from memory.models import Session
from engine.tools.registry import AGENT_TOOLS, TOOLS_SCHEMA, set_tool_context
from .tts_stt import synthesize_tts_async
from stream.text_sanitize import SanitizeOptions, sanitize_and_verbalize_text

from .utils import (
    _deck_snapshot_system_message,
    extract_sentences,
    build_text_system_prompt,
    build_router_system_prompt,
    elapsed_ms,
)

log = logging.getLogger(__name__)

# =============================================================================
# Settings / constants
# =============================================================================
@dataclass(frozen=True)
class Settings:
    text_temperature: float = 0
    router_temperature: float = 0
    max_history_pairs: int = 50
    tool_timeout_s: int = 25               # wait for tool results so first reply includes them
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

    tool_summaries: List[str]

    response: str
    emotion_final: Dict[str, Any]

    viseme_frames: List[List[float]]
    viseme_chunks: List[Dict[str, Any]]

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
    state["tool_summaries"] = []

    state["emotion_final"] = {}
    state["viseme_frames"] = []
    state["viseme_chunks"] = []

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
    Decide and execute tool calls. We WAIT for tools to finish before starting text,
    so the very first assistant message already includes tool results.
    """
    router = ChatOpenAI(model=(state.get("model") or "gpt-4o-mini"), temperature=S.router_temperature)

    # Provide full context to tools
    set_tool_context({
        "session_id": state["session"].id,
        "bot_id": str(state["agent"].bot_id),
        "agent_id": state["agent"].id,
        "user_id": state["session"].user_id,
    })

    # Trimmed recent history for the router
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
        log.exception("[nodes:router] routing failed")

    async def _exec(name: str, args: Dict[str, Any]):
        impl = AGENT_TOOLS.get(name)
        if not impl:
            log.warning("[nodes:tool] unknown tool: %s", name)
            return

        # Always await the tool so we include its results in the first reply
        res: Dict[str, Any] | None = None
        try:
            res = await asyncio.wait_for(impl.ainvoke(args), timeout=S.tool_timeout_s)
            status = (res or {}).get("status", "ok")
            summary = (res or {}).get("summary", "").strip()
            # Store as multi-line (tools already emit bullet-friendly summaries)
            pretty = f"[{name}] {status}:\n{summary}" if "\n" in summary else f"[{name}] {status}: {summary}"
            state["tool_summaries"].append(pretty.strip())
            log.debug("[nodes:tool] %s -> %s", name, status)
        except asyncio.TimeoutError:
            msg = f"[{name}] timeout after {S.tool_timeout_s}s."
            state["tool_summaries"].append(msg)
            log.warning("[nodes:tool] %s", msg)
            return
        except Exception:
            log.exception("[nodes:tool] %s call failed", name)
            return

        data = (res or {}).get("data") or {}

        # If a deck snapshot is returned, stream it immediately to the UI
        slide_tools = {
            # slides tools (current names)
            "slides_generate_or_update",
            "slides_fetch",
            "slides_list_versions",
            "slides_diff",
            "slides_revert",
            "slides_add_sections",
            "slides_remove_sections",
            "slides_edit",
            # docs tools can also emit slides
            "documents_analyze",
            "documents_generate_slides",
        }
        if name in slide_tools:
            snap = data.get("slides") or {}
            if snap:
                state["slides_tool_used"] = True
                state["slides_latest"] = snap
                try:
                    facts = deck_facts_from_snapshot(snap)  # uses helper above
                    state["deck_facts"] = facts
                except Exception:
                    log.exception("[nodes:router] failed building deck facts")

                if state.get("queue"):
                    await state["queue"].put({"type": "slides_response", "slides": snap})
                    await state["queue"].put({"type": "slides_done"})

    # Execute tools sequentially (keeps ordering and is easier to reason about for summaries)
    for c in tool_calls:
        await _exec((c.get("name") or "").strip(), dict(c.get("args") or {}))

    # Unblock text even if no slides were produced
    state["slides_ready_event"].set()

# =============================================================================
# Emotion (emit once BEFORE any text tokens)
# =============================================================================
async def _emit_emotion_before_text(state: QAState):
    try:
        if S.emotion_source == "none":
            emo = {"name": "Joy", "intensity": 1}
        else:
            tool = AGENT_TOOLS.get("emotion_detect")
            raw = getattr(state.get("user_msg"), "content", None) or (state.get("query") or "")
            text = (raw or "").strip()
            if not tool or not text:
                emo = {"name": "Joy", "intensity": 1}
            else:
                res = await asyncio.wait_for(tool.ainvoke({"text": text}), timeout=S.emotion_timeout_s)
                data = (res or {}).get("data") or {}
                name = str(data.get("name", "Joy")).title()
                intensity_val = data.get("intensity", 1)
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
    # Wait for tools + emotion so first text contains tool results
    await asyncio.gather(state["slides_ready_event"].wait(), state["emotion_ready_event"].wait())

    model_name = state.get("model") or "gpt-4o-mini"
    llm = ChatOpenAI(model=model_name, temperature=S.text_temperature, streaming=True)

    # Inject agent info (name/persona/description) in the system message
    agent = state["agent"]
    msgs: List[BaseMessage] = [
        SystemMessage(content=build_text_system_prompt(
            bot_name=agent.name or "Assistant",
            bot_id=str(agent.bot_id),
            persona=(agent.persona or ""),
            description=(agent.description or ""),
        ))
    ]

    # Concise tool summaries become context for the text model
    if state.get("tool_summaries"):
        joined = "- " + "\n- ".join(s[:600] for s in state["tool_summaries"][:8])
        msgs.append(SystemMessage(content="TOOL UPDATES:\n" + joined))

    deck_msg = _deck_snapshot_system_message(state)
    if deck_msg:
        # place it right before history/user for maximum salience
        msgs.append(SystemMessage(content=deck_msg))

    msgs.extend(state["base_msgs"])
    msgs.append(state["user_msg"])

    buf, final_text = "", ""
    voice = getattr(agent, "voice", None)
    voice_service = (getattr(voice, "service", "") or "").strip() if voice else ""
    voice_id = (getattr(voice, "voice_id", "") or "").strip() if voice else ""

    # Continuous viseme timeline metadata
    chunk_index = 0
    offset_ms_accum = 0

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
    # Run tools + emotion first; text streaming waits on events (tools/emotion) before emitting any tokens
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
        "schema": "engine.v12",
        "bot_id": state.get("bot_id"),
        "thread_id": state.get("thread_id"),
        "model": state.get("model") or "gpt-4o-mini",
        "timings_ms": timings,
        "response_len": len(state.get("response") or ""),
        "tools": list(AGENT_TOOLS.keys()),
        "frame_ms": frame_ms_meta,
        "slides_tool_used": bool(state.get("slides_tool_used")),
        "tool_summaries": state.get("tool_summaries") or [],
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
