from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from contextlib import suppress
from time import perf_counter, time as _time
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from stream.ws_auth import validate_ws_api_key_and_origin
from agent.models import Agent
from memory.models import Session
from engine.rag import Rag
from engine.tts_stt import transcribe_openai_from_bytes_async

log = logging.getLogger(__name__)

MAX_MESSAGE_RATE_HZ = 20
MAX_AUDIO_BYTES = 25 * 1024 * 1024
STT_MODEL_DEFAULT = os.getenv("OPENAI_STT_MODEL", "whisper-1")
THREAD_ID_PATTERN = re.compile(r"^user_[0-9a-f]{16}$", re.I)


def _fmt_local(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _infer_ext_from_dataurl_or_hint(b64_or_dataurl: str, fmt_hint: Optional[str] = None) -> str:
    if isinstance(b64_or_dataurl, str) and b64_or_dataurl.startswith("data:"):
        m = re.match(r"^data:audio/([a-z0-9+.\-]+);base64,", b64_or_dataurl, re.I)
        if m:
            mime_ext = m.group(1).lower()
            if "wav" in mime_ext: return "wav"
            if "mpeg" in mime_ext or "mp3" in mime_ext or "mpga" in mime_ext: return "mp3"
            if "m4a" in mime_ext or "mp4" in mime_ext: return "m4a"
            if "ogg" in mime_ext or "oga" in mime_ext: return "ogg"
            if "webm" in mime_ext: return "webm"
    fmt = (fmt_hint or "").lower()
    if "wav" in fmt: return "wav"
    if "m4a" in fmt or "mp4" in fmt: return "m4a"
    if "ogg" in fmt or "oga" in fmt: return "ogg"
    if "webm" in fmt: return "webm"
    if "mpeg" in fmt or "mp3" in fmt or "mpga" in fmt: return "mp3"
    return "webm"


class ChatConsumer(AsyncJsonWebsocketConsumer):
    """
    Outbound events (forwarded from engine queue):
      - connected
      - response_start
      - text_token {token, local_time}
      - audio_response {audio, viseme, local_time}
      - emotion {emotion: {name, intensity}, local_time}
      - slides_response {slides, local_time}
      - slides_done
      - response_done {timings}
      - response_ended
      - audio_muted {muted: boolean}
      - stop_audio
      - error {message}
      - pong

    Inbound messages:
      - text_query {text, local_time?, muteAudio?}
      - audio_query {audio(base64|dataurl), format?, muteAudio?}
      - mute_audio
      - unmute_audio
      - stop_audio
      - ping
    """

    async def connect(self):
        auth, err = await sync_to_async(validate_ws_api_key_and_origin)(self.scope)
        if not auth:
            log.warning("[ws][connect] auth_failed code=4403 err=%s", err)
            await self.close(code=4403)
            return

        self.api_auth = auth
        self.user_id = auth.user_id

        self.thread_id: Optional[str] = None
        self.bot_id: Optional[str] = None
        self.website_language: str = "en"
        self.agent: Optional[Agent] = None
        self.session: Optional[Session] = None
        self.group_name: Optional[str] = None

        # audio control flags (connection-local)
        self.audio_muted: bool = False
        self.audio_stopped: bool = False

        self._last_msg_ts: float = 0.0

        # single-active-run
        self._queue: Optional[asyncio.Queue] = None
        self._pump_task: Optional[asyncio.Task] = None
        self._gen_task: Optional[asyncio.Task] = None

        qs = parse_qs((self.scope.get("query_string") or b"").decode("utf-8"))
        self.thread_id = (qs.get("thread_id") or [None])[0]
        self.bot_id = (qs.get("bot_id") or [None])[0]
        self.website_language = (qs.get("website_language") or ["en"])[0]

        if not (self.bot_id and self.thread_id):
            await self.close(code=4000)
            return
        if not THREAD_ID_PATTERN.fullmatch(self.thread_id):
            await self.close(code=4005)
            return

        self.agent = await sync_to_async(
            lambda: Agent.objects.filter(bot_id=self.bot_id, is_active=True).select_related("voice", "user").first()
        )()
        if not self.agent:
            await self.close(code=4004)
            return

        self.session = await sync_to_async(
            lambda: Session.objects.filter(thread_id=self.thread_id, agent_id=self.agent.id, is_active=True)
            .select_related("agent", "user").first()
        )()
        if not self.session:
            await self.close(code=4005)
            return

        self.group_name = f"chat_{self.bot_id}_{self.thread_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        await self.send_json({"type": "connected", "bot_id": self.bot_id, "thread_id": self.thread_id})
        log.info("[ws][connect] ok bot=%s thread=%s user=%s", self.bot_id, self.thread_id, self.user_id)

    async def disconnect(self, code):
        log.info("[ws][disconnect] code=%s bot=%s thread=%s", code, getattr(self, "bot_id", None), getattr(self, "thread_id", None))
        await self._cancel_current_run()
        if getattr(self, "group_name", None):
            with suppress(Exception):
                await self.channel_layer.group_discard(self.group_name, self.channel_name)

    def _allow_message(self) -> bool:
        now = _time()
        if now - self._last_msg_ts < 1.0 / MAX_MESSAGE_RATE_HZ:
            return False
        self._last_msg_ts = now
        return True

    async def receive_json(self, content, **kwargs):
        if not self._allow_message():
            return
        try:
            t = (content or {}).get("type")
            if t == "text_query":
                await self._handle_text_query(content)
            elif t == "audio_query":
                await self._handle_audio_query(content)
            elif t == "mute_audio":
                self.audio_muted = True
                await self.send_json({"type": "audio_muted", "muted": True})
            elif t == "unmute_audio":
                self.audio_muted = False
                await self.send_json({"type": "audio_muted", "muted": False})
            elif t == "stop_audio":
                self.audio_stopped = True
                await self.send_json({"type": "stop_audio"})
            elif t == "ping":
                await self.send_json({"type": "pong"})
            else:
                await self.send_json({"type": "error", "message": f"Unknown type: {t}"})
        except Exception:
            log.exception("[ws][recv] error")
            await self.send_json({"type": "error", "message": "Internal error"})

    async def _handle_text_query(self, data: Dict[str, Any]):
        text = (data.get("text") or "").strip()
        if not text:
            return

        if "muteAudio" in data:
            self.audio_muted = bool(data.get("muteAudio"))

        # new run resets stop flag
        self.audio_stopped = False

        await self._cancel_current_run()
        browser_time = data.get("local_time") or _fmt_local(datetime.now())

        await self.send_json({"type": "text_query", "text": text, "local_time": browser_time})
        log.info("[ws][text_query] bot=%s thread=%s len=%d muted=%s", self.bot_id, self.thread_id, len(text), self.audio_muted)

        self._queue = asyncio.Queue()
        self._pump_task = asyncio.create_task(self._pump_queue(self._queue, browser_time))
        rag = Rag(queue=self._queue)

        async def _gen():
            started = perf_counter()
            try:
                await self.send_json({"type": "response_start"})
                payload = await rag.run(query=text, bot_id=str(self.agent.bot_id), thread_id=self.thread_id)
                elapsed = round(perf_counter() - started, 3)
                timings = dict(payload.get("timings", {}))
                timings.setdefault("ws_total_sec", elapsed)
                await self.send_json({"type": "response_done", "timings": timings})
                log.info("[ws][response_done] bot=%s thread=%s timings=%s", self.bot_id, self.thread_id, timings)
            except Exception:
                log.exception("[ws][gen] failed")
                await self.send_json({"type": "error", "message": "Generation failed"})
            finally:
                # Gracefully close pump after normal completion
                try:
                    if self._queue is not None:
                        await self._queue.put(None)
                except Exception:
                    pass
                await self.send_json({"type": "response_ended"})

        self._gen_task = asyncio.create_task(_gen())

        # echo current mute state so client can reflect UI
        await self.send_json({"type": "audio_muted", "muted": self.audio_muted})

    async def _pump_queue(self, q, browser_time: str):
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                t = item.get("type")

                if t == "text_token":
                    out = dict(item)
                    out["local_time"] = browser_time
                    await self.send_json(out)

                elif t == "audio_response":
                    if self.audio_stopped or self.audio_muted:
                        continue
                    item["local_time"] = browser_time
                    await self.send_json(item)

                elif t == "emotion":
                    item["local_time"] = browser_time
                    await self.send_json(item)

                elif t in ("slides_response", "slides_done"):
                    if t != "slides_done":
                        item["local_time"] = browser_time
                    await self.send_json(item)

                elif t == "error":
                    await self.send_json({"type": "error", "message": item.get("message", "unknown")})

                else:
                    # Forward unknown engine events verbatim (future-proofing)
                    try:
                        item["local_time"] = browser_time
                    except Exception:
                        pass
                    await self.send_json(item)

        except asyncio.CancelledError:
            # normal during cancellations between runs
            pass
        except Exception:
            log.exception("[ws][pump] failed")

    async def _cancel_current_run(self):
        # Cancel generator
        if getattr(self, "_gen_task", None) and not self._gen_task.done():
            self._gen_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._gen_task
        self._gen_task = None

        # Cancel pump
        if getattr(self, "_pump_task", None) and not self._pump_task.done():
            self._pump_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._pump_task
        self._pump_task = None

        # Close any existing queue
        if getattr(self, "_queue", None) is not None:
            with suppress(Exception):
                await self._queue.put(None)
        self._queue = None

    async def _handle_audio_query(self, data: Dict[str, Any]):
        b64 = data.get("audio")
        if "muteAudio" in data:
            self.audio_muted = bool(data.get("muteAudio"))

        if not b64:
            await self.send_json({"type": "error", "message": "No audio provided"})
            return
        try:
            if isinstance(b64, str) and b64.startswith("data:"):
                payload = base64.b64decode(b64.split(",", 1)[1], validate=False)
            else:
                payload = base64.b64decode(b64, validate=False)
        except Exception:
            await self.send_json({"type": "error", "message": "Bad audio encoding"})
            return
        if len(payload) > MAX_AUDIO_BYTES:
            await self.send_json({"type": "error", "message": "Audio too large"})
            return

        ext = _infer_ext_from_dataurl_or_hint(b64, data.get("format"))
        filename = f"audio.{ext}"
        try:
            transcript = await transcribe_openai_from_bytes_async(payload, filename_hint=filename, model=STT_MODEL_DEFAULT)
            now_str = _fmt_local(datetime.now())
            await self._handle_text_query({"text": transcript, "local_time": now_str, "muteAudio": self.audio_muted})
        except Exception:
            log.exception("[ws][audio] processing_failed")
            await self.send_json({"type": "error", "message": "Audio processing failed"})
