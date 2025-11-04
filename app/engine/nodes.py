# app/engine/nodes.py
from __future__ import annotations

import asyncio
import logging
import os
import time
import re
from dataclasses import dataclass
from typing import TypedDict, List, Dict, Any, Optional, Tuple

from asgiref.sync import sync_to_async
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from agent.models import Agent
from memory.models import Session, Slides
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

    # Emotion tool
    emotion_model: str = "gpt-4o-mini"
    emotion_timeout_s: float = 6.0

    # Viseme aggregation
    viseme_max_frames_to_store: int = 4000

    # Viseme timing (frame period in ms; e.g., 24 FPS -> ~41.7 ms)
    vis_fps_env: int = int(os.getenv("VIS_FPS", "24"))

S = Settings()
FRAME_MS = max(10, int(round(1000.0 / max(1, S.vis_fps_env))))

# Force primary DB for read-after-write
DB_PRIMARY = os.getenv("DB_PRIMARY_ALIAS", "default")


# =============================================================================
# Slide intent (deterministic triggers)
# =============================================================================
_SLIDE_WRITE_TRIGGER = re.compile(
    r"\b(create|make|generate|draft|prepare|build|update|revise|edit|modify|add|convert|turn)\b"
    r".{0,40}\b(slide|slides|deck|presentation|ppt|pptx)\b",
    re.IGNORECASE,
)

_SLIDE_READ_TRIGGER = re.compile(
    r"\b(show|view|see|display|open|fetch|load|share)\b"
    r".{0,40}\b(slide|slides|deck|presentation|ppt|pptx)\b"
    r"|(?:\blatest\b|\bcurrent\b|\bwhat changed\b|\brecent changes\b|\bupdate status\b|\bdiff\b|\bcompare\b)",
    re.IGNORECASE,
)

def _wants_slide_write(text: str) -> bool:
    return bool(_SLIDE_WRITE_TRIGGER.search((text or "")))

def _wants_slide_read(text: str) -> bool:
    return bool(_SLIDE_READ_TRIGGER.search((text or "")))


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

    # async plumbing
    text_finalized_event: asyncio.Event
    slides_ready_event: asyncio.Event  # gate text start on slides readiness

    # Slides / tools
    slides_latest: Dict[str, Any]      # snapshot dict (after fetch or persist)
    slides_tool_used: bool
    slides_not_found: bool
    slides_write_persisted: bool

    # Deterministic intent flags (NEW)
    must_write_intent: bool
    must_read_intent: bool

    response: str

    # Final single emotion result (no arrays)
    emotion_final: Dict[str, Any]

    # Viseme aggregation across sentences
    viseme_frames: List[List[float]]

    # Per-turn slides context for generation (when tools used)
    turn_slides_context: str

    meta: Dict[str, Any]
    timings: Dict[str, int]


# =============================================================================
# Small helpers
# =============================================================================
def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _flatten_editorjs_for_prompt(editorjs: Dict[str, Any], *, limit_blocks: int = 80) -> List[str]:
    if not isinstance(editorjs, dict):
        return []
    blocks = editorjs.get("blocks") or []
    lines: List[str] = []
    for b in blocks[:limit_blocks]:
        t = (b.get("type") or "").lower()
        d = b.get("data") or {}
        if t == "header":
            txt = (d.get("text") or "").strip()
            if txt:
                lines.append(f"# {txt}")
        elif t == "paragraph":
            txt = (d.get("text") or "").strip()
            if txt:
                lines.append(txt)
        elif t == "list":
            items = d.get("items") or []
            for it in items[:8]:
                s = str(it).strip()
                if s:
                    lines.append(f"- {s}")
    return lines


def _build_context_from_snapshot(snap: Dict[str, Any]) -> str:
    """Produce a compact, read-only textual context of current + previous slides."""
    title = snap.get("title") or ""
    summary = snap.get("summary") or ""
    editorjs = snap.get("editorjs") or {}
    prev = snap.get("previous") or {}
    prev_title = prev.get("title") or ""
    prev_summary = prev.get("summary") or ""
    prev_ej = prev.get("editorjs") or {}

    # NEW: helpful meta for grounding
    version = snap.get("version")
    updated_by = snap.get("updated_by") or ""
    updated_at = snap.get("updated_at") or ""

    parts: List[str] = []
    parts.append("## Slides Snapshot (use ONLY this; ignore earlier chat mentions of slides)")
    parts.append(f"Version: {version}  |  Updated by: {updated_by}  |  Updated at: {updated_at}")
    parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
    parts.extend(_flatten_editorjs_for_prompt(editorjs))

    parts.append("\n## Previous Snapshot")
    parts.append(f"Prev Title: {prev_title}")
    if prev_summary:
        parts.append(f"Prev Summary: {prev_summary}")
    parts.extend(_flatten_editorjs_for_prompt(prev_ej))

    # NEW: explicit header diff (added/removed/unchanged) to drive change logs
    diff = snap.get("diff_headers") or {}
    if diff:
        parts.append("\n## Header Diff")
        if diff.get("added"):
            parts.append("Added: " + "; ".join(diff["added"][:8]))
        if diff.get("removed"):
            parts.append("Removed: " + "; ".join(diff["removed"][:8]))
        if diff.get("unchanged"):
            parts.append("Unchanged: " + "; ".join(diff["unchanged"][:8]))

    out = "\n".join(parts)
    return out[:8000] + ("\n…" if len(out) > 8000 else "")


def _is_meaningful(title: Any, summary: Any, editorjs: Any) -> bool:
    def _empty_str(s: Any) -> bool:
        if not isinstance(s, str):
            return True
        v = s.strip().lower()
        return v == "" or v in {"string", "null", "none"}
    has_title = isinstance(title, str) and not _empty_str(title)
    has_summary = isinstance(summary, str) and not _empty_str(summary)
    has_blocks = isinstance(editorjs, dict) and bool(editorjs.get("blocks"))
    return has_title or has_summary or has_blocks


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

    # do NOT select_related("slides") to avoid stale relation cache
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
def _rotate_and_save_slides(session: Session, payload: Dict[str, Any], *, updated_by: str = "tool:slides") -> Dict[str, Any]:
    """
    Persist current + previous using Slides.rotate_and_update.
    Skip creation/rotation if nothing meaningful.
    Always hit primary and invalidate relation cache.
    """
    s = (
        Slides.objects.using(DB_PRIMARY)
        .filter(session=session)
        .first()
    )

    title = str(payload.get("title") or "").strip()
    summary = str(payload.get("summary") or "").strip()
    editorjs = payload.get("editorjs") or {}

    # NOOP: nothing meaningful → do not create or rotate
    if not _is_meaningful(title, summary, editorjs):
        if not s:
            return {}
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
            "thread_id": session.thread_id,
            "updated_by": s.updated_by,
            "updated_at": s.updated_at.isoformat(),
        }

    if not s:
        s = Slides.objects.using(DB_PRIMARY).create(session=session)

    # Rotate correctly: old -> previous_*, new -> current
    s.rotate_and_update(title=title, summary=summary, editorjs=editorjs, updated_by=updated_by)
    s.save(update_fields=[
        "previous_title", "previous_summary", "previous_editorjs",
        "title", "summary", "editorjs", "updated_by", "version", "updated_at"
    ])

    # ensure we return the committed, freshest row
    s.refresh_from_db(using=DB_PRIMARY)

    # Invalidate relation cache if it was populated on this Session instance
    try:
        delattr(session, "slides")
    except Exception:
        pass

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
        "thread_id": session.thread_id,
        "updated_by": s.updated_by,
        "updated_at": s.updated_at.isoformat(),
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

    # Deterministic intent flags (NEW)
    user_text = state["user_msg"].content
    state["must_write_intent"] = _wants_slide_write(user_text)
    state["must_read_intent"] = _wants_slide_read(user_text)

    state["slides_latest"] = {}
    state["slides_tool_used"] = False
    state["slides_not_found"] = False
    state["slides_write_persisted"] = False

    state["emotion_final"] = {}
    state["viseme_frames"] = []
    state["turn_slides_context"] = ""

    state["meta"] = {}
    state["timings"] = {"prepare_ms": _elapsed_ms(t0)}
    state["text_finalized_event"] = asyncio.Event()
    state["slides_ready_event"] = asyncio.Event()
    return state


# =============================================================================
# Tools planner/executor (non-blocking)
# =============================================================================
async def _tool_router_and_stream(state: QAState):
    """
    Decide tools and, if slides change, persist and build a fresh slides snapshot
    BEFORE text generation starts (we gate on slides_ready_event).
    IMPORTANT: To avoid bias from long chat history, we route with ONLY the user's
    current message + a strict tool policy system message.
    """
    router = ChatOpenAI(model=(state.get("model") or "gpt-4o-mini"), temperature=S.router_temperature)

    set_tool_context({
        "session_id": state["session"].id,
        "bot_id": str(state["agent"].bot_id),
        "agent_id": state["agent"].id,
    })

    planner_msgs: List[BaseMessage] = [
        SystemMessage(content=(
            "You may call tools for THIS session.\n"
            "Tools:\n"
            "  • fetch_latest_slides() — READ-ONLY. Use for any request to view/check/show/summarize/compare slides,\n"
            "    or phrases like: latest, current, what changed, recent changes, update status, diff, compare.\n"
            "  • generate_or_update_slides(editorjs?, title?, summary?, ai_enrich, max_sections) — WRITE.\n"
            "    STRICT policy:\n"
            "      - If the user's message contains any of these verbs with slide nouns, you MUST call this tool:\n"
            "        create/make/generate/draft/prepare/build/update/revise/edit/modify/add/convert/turn + slide/deck/presentation/ppt.\n"
            "      - Never attempt to create/modify slides in assistant text. Do not output JSON.\n"
            "      - If intent is ambiguous, ask one brief clarification; do NOT create a deck.\n"
            "  • search_wikipedia(query)\n"
            "HARD RULES:\n"
            "  1) Do NOT call generate_or_update_slides unless the user explicitly asks to create or update slides.\n"
            "  2) When the user explicitly asks to create/update, you MUST call generate_or_update_slides.\n"
            "     Never fabricate a deck or claim an update without a successful tool write.\n"
            "  3) If the user asks for latest/current changes or to show/compare the deck, call fetch_latest_slides.\n"
            "  4) Never include slide JSON in assistant text.\n"
            "\n"
            "Examples:\n"
            "User: 'Create slides on Q4 OKRs'  -> Call generate_or_update_slides\n"
            "User: 'Update the deck with a risk section' -> Call generate_or_update_slides\n"
            "User: 'Show me the latest deck'   -> Call fetch_latest_slides\n"
        )),
        state["user_msg"],  # intentionally exclude chat history here
    ]

    tools_schema = TOOLS_SCHEMA  # expose all tools; instruction above governs usage
    tool_calls: List[Dict[str, Any]] = []
    try:
        ai = await router.ainvoke(planner_msgs, tools=tools_schema, tool_choice="auto")
        tool_calls = (getattr(ai, "tool_calls", []) or [])[: S.max_tool_calls]
    except Exception:
        log.exception("[nodes:tool_router] router failed")
        tool_calls = []

    # NEW: deterministic fallbacks based on lexical intent
    if not tool_calls:
        if state.get("must_write_intent"):
            log.info("[router:fallback] Forcing generate_or_update_slides due to write intent")
            tool_calls = [{"name": "generate_or_update_slides", "args": {}}]
        elif state.get("must_read_intent"):
            log.info("[router:fallback] Forcing fetch_latest_slides due to read intent")
            tool_calls = [{"name": "fetch_latest_slides", "args": {}}]

    async def _exec_call(call: Dict[str, Any]):
        name = (call.get("name") or "").strip()
        args = dict(call.get("args") or {})
        impl = AGENT_TOOLS.get(name)
        if not impl:
            return

        try:
            if name == "generate_or_update_slides":
                # Ensure sane defaults for enrichment; actual persistence happens below.
                args.setdefault("ai_enrich", True)
                args.setdefault("max_sections", 6)

            res = await impl.ainvoke(args)
        except Exception:
            log.exception("[nodes:tool_call] %s failed", name)
            return

        # READ
        if name == "fetch_latest_slides":
            state["slides_tool_used"] = True
            if (res or {}).get("status") == "ok":
                snap = res.get("slides") or {}
                state["slides_latest"] = snap
                if state.get("queue"):
                    await state["queue"].put({"type": "slides_response", "slides": snap})
                    await state["queue"].put({"type": "slides_done"})
            elif (res or {}).get("status") == "not_found":
                state["slides_not_found"] = True
                if state.get("queue"):
                    await state["queue"].put({"type": "slides_response", "slides": {}})
                    await state["queue"].put({"type": "slides_done"})

        # WRITE (persist immediately, because the user explicitly asked)
        elif name == "generate_or_update_slides":
            state["slides_tool_used"] = True
            if res.get("no_write"):
                # no meaningful input; nothing to persist
                state["slides_write_persisted"] = False
                return
            try:
                persisted = await _rotate_and_save_slides(state["session"], res, updated_by="user")
                if persisted:
                    state["slides_latest"] = persisted
                    state["slides_write_persisted"] = True
                    # notify websocket clients with the fresh snapshot
                    if state.get("queue"):
                        await state["queue"].put({"type": "slides_response", "slides": persisted})
                        await state["queue"].put({"type": "slides_done"})
            except Exception:
                log.exception("[nodes:slides] persist failed")
                state["slides_write_persisted"] = False

    # Run selected tools concurrently
    await asyncio.gather(*(_exec_call(c) for c in tool_calls))

    # Build per-turn slides context ONLY from the latest snapshot (and only when a slide tool ran)
    if state["slides_tool_used"] and state["slides_latest"]:
        state["turn_slides_context"] = (
            "Use ONLY the snapshot below for slides-related answers. "
            "Ignore all earlier chat mentions about slides or titles. "
            "Do NOT output slide JSON.\n\n" + _build_context_from_snapshot(state["slides_latest"])
        )
    elif state["slides_tool_used"] and state["slides_not_found"]:
        state["turn_slides_context"] = "There are no slides for this session yet."
    else:
        state["turn_slides_context"] = ""

    # trace
    log.info(
        "[router:intent] must_write=%s must_read=%s tool_used=%s persisted=%s not_found=%s",
        state.get("must_write_intent"), state.get("must_read_intent"),
        state.get("slides_tool_used"), state.get("slides_write_persisted"),
        state.get("slides_not_found"),
    )

    # signal: slides context is finalized for this turn
    state["slides_ready_event"].set()


# =============================================================================
# Text streaming + sentence-level audio
# =============================================================================
async def _stream_text_with_audio(state: QAState):
    # Wait (briefly) so that slides context reflects any update from tools
    try:
        await asyncio.wait_for(state["slides_ready_event"].wait(), timeout=1.5)
    except Exception:
        # proceed regardless
        pass

    model_name = state.get("model") or "gpt-4o-mini"
    llm = ChatOpenAI(model=model_name, temperature=S.text_temperature, streaming=True)

    # Build messages. If slide tools were used this turn, DROP chat history to avoid bias.
    msgs: List[BaseMessage] = [
        SystemMessage(content=build_text_system_prompt(
            bot_name=state["agent"].name or "Assistant",
            bot_id=str(state["agent"].bot_id),
            instruction=(state["agent"].persona or "").strip(),
        ))
    ]

    # NEW: Guard if user asked to WRITE slides but nothing persisted.
    if state.get("must_write_intent") and not state.get("slides_write_persisted"):
        msgs.append(SystemMessage(content=(
            "Slides write was requested this turn but no deck has been saved yet.\n"
            "Do NOT claim slides were created or updated. Do NOT output slide JSON.\n"
            "Reply briefly: 'No changes saved yet.' Then ask for concrete inputs "
            "(title, audience, goal, and 3–6 section headers)."
        )))

    # NEW: Guard if user asked to READ slides but none exist.
    if state.get("must_read_intent") and not state.get("slides_latest"):
        msgs.append(SystemMessage(content=(
            "No slides are available for this session. State this briefly. "
            "Offer to create a new deck if the user wants; do not invent content."
        )))

    if state["slides_tool_used"]:
        # Guardrails so the model does not claim an update unless we actually persisted,
        # and always produces a change log for "latest/what changed" style questions.
        guard = (
            "Slides policy for this turn:\n"
            f"• slides_snapshot_available={str(bool(state.get('slides_latest'))).lower()}.\n"
            f"• slides_write_persisted={str(bool(state.get('slides_write_persisted'))).lower()}.\n"
            "• If the user asks for latest/current/what changed/update status/diff/compare:\n"
            "    - Produce a concise change log using ONLY the snapshots below:\n"
            "      • Title change: <previous> → <current> (only if different)\n"
            "      • Summary: 'updated' if changed; else 'no change'\n"
            "      • Sections added: list from 'Header Diff: Added'\n"
            "      • Sections removed: list from 'Header Diff: Removed'\n"
            "      • Unchanged topics: up to 3 from 'Header Diff: Unchanged'\n"
            "    - Do NOT mention 'No changes were saved' for these read-only requests.\n"
            "• Only say 'No changes were saved' when the user explicitly asked to create/update and slides_write_persisted=false.\n"
            "• Never output Editor.js JSON. Use ONLY the snapshot content that follows."
        )

        if state.get("turn_slides_context"):
            msgs.append(SystemMessage(content=guard + "\n\n" + state["turn_slides_context"]))
        msgs.append(state["user_msg"])
    else:
        # Normal path: include chat history
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
                    import base64
                    b64 = base64.b64encode(audio).decode("utf-8")

                    if visemes and visemes[0] != [0.0]*15:
                        visemes = [[0.0]*15] + visemes
                    if visemes and visemes[-1] != [0.0]*15:
                        visemes = visemes + [[0.0]*15]

                    # seconds since start for each viseme frame:
                    times = [i * (FRAME_MS / 1000.0) for i in range(len(visemes))]
                    fps = round(1000.0 / FRAME_MS, 3)

                    await state["queue"].put({
                        "type": "audio_response",
                        "audio": b64,
                        "viseme": visemes,
                        "viseme_format": "arkit15",
                        "viseme_times": times,     # precise sync
                        "viseme_fps": fps          # also provide fps for sanity
                    })

                    if isinstance(visemes, list):
                        state["viseme_frames"].extend(visemes)
                        if len(state["viseme_frames"]) > S.viseme_max_frames_to_store:
                            state["viseme_frames"] = state["viseme_frames"][-S.viseme_max_frames_to_store:]
                except Exception:
                    log.exception("[nodes:tts] sentence TTS failed")

    tail = buf.strip()
    if tail and voice_service and voice_id and state.get("queue"):
        try:
            cleaned = await asyncio.to_thread(sanitize_and_verbalize_text, tail, "en_GB", SanitizeOptions())
            audio, visemes = await synthesize_tts_async(cleaned, voice_service, voice_id)
            import base64
            b64 = base64.b64encode(audio).decode("utf-8")

            if visemes and visemes[0] != [0.0]*15:
                visemes = [[0.0]*15] + visemes
            if visemes and visemes[-1] != [0.0]*15:
                visemes = visemes + [[0.0]*15]

            # seconds since start for each viseme frame:
            times = [i * (FRAME_MS / 1000.0) for i in range(len(visemes))]
            fps = round(1000.0 / FRAME_MS, 3)

            await state["queue"].put({
                "type": "audio_response",
                "audio": b64,
                "viseme": visemes,
                "viseme_format": "arkit15",
                "viseme_times": times,     # precise sync
                "viseme_fps": fps          # also provide fps for sanity
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
# Run both streams in parallel
# =============================================================================
async def n_run(state: QAState) -> QAState:
    t0 = time.perf_counter()
    await asyncio.gather(
        _tool_router_and_stream(state),   # finalize slides first (gated)
        _stream_text_with_audio(state),   # text waits for slides_ready_event
        _emit_emotion_once(state),        # single emotion result, separate event
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
        "schema": "engine.v7",
        "bot_id": state.get("bot_id"),
        "thread_id": state.get("thread_id"),
        "model": state.get("model") or "gpt-4o-mini",
        "timings_ms": timings,
        "response_len": len(state.get("response") or ""),
        "tools": list(AGENT_TOOLS.keys()),
        "frame_ms": FRAME_MS,
        # helpful trace flags
        "slides_tool_used": bool(state.get("slides_tool_used")),
        "slides_write_persisted": bool(state.get("slides_write_persisted")),
    })

    viseme_json = {}
    if state.get("viseme_frames"):
        viseme_json = {
            "frame_ms": FRAME_MS,
            "frames": state["viseme_frames"],
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
