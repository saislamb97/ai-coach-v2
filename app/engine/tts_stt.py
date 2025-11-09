# engine/tts_stt.py
from __future__ import annotations

import io
import os
import re
import logging
from io import BytesIO
from time import perf_counter
from typing import Optional, List, Tuple, Callable, Awaitable, Dict, TypedDict

# No pydub import here — ffmpeg not required

# Import the viseme generator & constants
from .arkit import generate_visemes, MS_PER_FRAME, ARKIT_DIM

log = logging.getLogger(__name__)

__all__ = [
    "transcribe_openai_async",
    "transcribe_openai_from_bytes_async",
    "synthesize_tts_async",
]

# -----------------------------------------------------------------------------
# Env helpers / core config (STT/TTS still use envs as before)
# -----------------------------------------------------------------------------
def _env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"{name} is required")
    return v or ""

# Audio postproc (normalization skipped to avoid decode/ffmpeg)
MP3_BITRATE = os.getenv("MP3_BITRATE", "192k")
DO_NORMALIZE = False  # force off; we avoid decoding to keep ffmpeg-free
TARGET_DBFS = float(os.getenv("TARGET_DBFS", "-1.0"))

# -----------------------------------------------------------------------------
# Utilities (normalization stub, duration estimate)
# -----------------------------------------------------------------------------
def _normalize_mp3(audio_bytes: bytes) -> bytes:
    # Intentionally no-op to avoid decoding/ffmpeg.
    # If you later add a pure-Python MP3 normalizer, wire it here.
    return audio_bytes

def _estimate_speech_ms(text: str) -> int:
    """Heuristic speech duration estimate with readable pauses; ffmpeg-free."""
    text = (text or "").strip()
    if not text:
        return 500
    words = re.findall(r"\b[\w']+\b", text)
    wpm = 185.0  # natural TTS speaking rate
    base_ms = int((len(words) / max(1.0, wpm)) * 60_000)
    # Pause contributions
    pauses = (
        180 * text.count(".") +
        180 * text.count("!") +
        180 * text.count("?") +
        120 * text.count(",") +
        140 * (text.count(";") + text.count(":"))
    )
    # Small floor/ceiling
    extras = max(300, min(1200, int(8 * len(text))))
    return max(300, base_ms + pauses + extras)

def _mp3_duration_ms(audio_bytes: bytes, fallback_ms: int) -> int:
    # Without decoding MP3 (no ffmpeg), return the heuristic fallback.
    return fallback_ms

# -----------------------------------------------------------------------------
# OpenAI Whisper STT
# -----------------------------------------------------------------------------
async def _openai_transcribe_from_filelike(fh: io.BytesIO, model: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=_env("OPENAI_API_KEY", required=True))
    t0 = perf_counter()
    res = await client.audio.transcriptions.create(model=model, file=fh)
    text = (getattr(res, "text", "") or "").strip()
    out = " ".join(text.split()) or "- No Audio -"
    log.info("[STT] model=%s dur=%.3fs len=%d", model, perf_counter() - t0, len(out))
    return out

async def transcribe_openai_from_bytes_async(
    data: bytes,
    filename_hint: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    model = (model or _env("OPENAI_STT_MODEL", "whisper-1")).strip()
    buf = io.BytesIO(data); buf.name = (filename_hint or "audio.webm").strip()
    return await _openai_transcribe_from_filelike(buf, model)

async def transcribe_openai_async(file_path: str, model: Optional[str] = None) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    return await transcribe_openai_from_bytes_async(
        data, filename_hint=os.path.basename(file_path), model=model
    )

# -----------------------------------------------------------------------------
# TTS engines (all must return MP3 bytes; still using env-config)
# -----------------------------------------------------------------------------
async def _tts_gtts(text: str, voice_id: str) -> bytes:
    from gtts import gTTS
    import asyncio

    lang, tld = "en", "com"
    if "@" in (voice_id or ""):
        lang, tld = [s.strip() for s in voice_id.split("@", 1)]

    def _work() -> bytes:
        out = BytesIO()
        gTTS(text=text, lang=lang, tld=tld).write_to_fp(out)  # MP3
        out.seek(0)
        return out.read()

    t0 = perf_counter()
    data = await asyncio.to_thread(_work)
    log.info("[TTS][gTTS] lang=%s tld=%s bytes=%d dur=%.3fs", lang, tld, len(data), perf_counter() - t0)
    return data

async def _tts_elevenlabs(text: str, voice_id: str) -> bytes:
    from elevenlabs import ElevenLabs

    if not voice_id:
        raise ValueError("elevenlabs: voice_id is required")
    if voice_id.lower().startswith("elevenlabs::"):
        voice_id = voice_id.split("::", 1)[1].strip()

    client = ElevenLabs(api_key=_env("ELEVENLABS_API_KEY", required=True))
    model_id = _env("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")

    t0 = perf_counter()
    stream = client.text_to_speech.convert(voice_id=voice_id, text=text, model_id=model_id)  # MP3 stream
    data = b"".join(stream)
    if not data:
        raise RuntimeError("No audio from ElevenLabs")
    log.info("[TTS][EL] model=%s voice=%s bytes=%d dur=%.3fs", model_id, voice_id, len(data), perf_counter() - t0)
    return data

async def _tts_openai(text: str, voice_id: str) -> bytes:
    from openai import AsyncOpenAI

    if not voice_id:
        raise ValueError("openai: voice_id is required (e.g., 'openai::alloy' or 'alloy')")
    if voice_id.lower().startswith("openai::"):
        voice_id = voice_id.split("::", 1)[1].strip()

    client = AsyncOpenAI(api_key=_env("OPENAI_API_KEY", required=True))
    model = _env("OPENAI_TTS_MODEL", "tts-1").strip()

    async def _call(model_name: str) -> bytes:
        resp = await client.audio.speech.create(
            model=model_name,
            voice=voice_id,
            input=text,
            response_format="mp3",  # ensure MP3
        )
        data = getattr(resp, "content", None)
        if data is None and hasattr(resp, "read"):
            data = await resp.read()
        if isinstance(data, bytearray):
            data = bytes(data)
        if not data:
            raise RuntimeError("No audio from OpenAI TTS")
        return data

    t0 = perf_counter()
    try:
        data = await _call(model)
    except Exception:
        if model != "tts-1":
            data = await _call("tts-1")
        else:
            raise
    log.info("[TTS][OpenAI] model=%s voice=%s bytes=%d dur=%.3fs", model, voice_id, len(data), perf_counter() - t0)
    return data

_TTS: Dict[str, Callable[[str, str], Awaitable[bytes]]] = {
    "gtts": _tts_gtts,
    "elevenlabs": _tts_elevenlabs,
    "openai": _tts_openai,
}

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
class VisemeData(TypedDict):
    viseme: List[List[float]]
    viseme_times: List[float]
    viseme_format: str
    viseme_fps: float
    duration_ms: int
    frame_ms: int

def _parse(service: Optional[str], voice_id: Optional[str]) -> Tuple[str, str]:
    svc = (service or "").strip().lower()
    vid = (voice_id or "").strip()
    if not svc and "::" not in vid:
        raise ValueError("TTS service missing. Provide service or use 'service::voice_id'.")
    if "::" in vid:
        pfx, raw = vid.split("::", 1)
        svc = pfx.strip().lower()
        vid = raw.strip()
    if not svc:
        raise ValueError("TTS service missing after parsing voice_id.")
    return svc, vid

async def synthesize_tts_async(text: str, service: str | None, voice_id: str | None) -> Tuple[bytes, VisemeData]:
    if not (text or "").strip():
        raise ValueError("Empty text for TTS.")

    svc, vid = _parse(service, voice_id)
    engine = _TTS.get(svc)
    if not engine:
        raise ValueError(f"Unsupported TTS service: {svc}")

    # 1) TTS audio (MP3 bytes)
    audio = await engine(text, vid)
    if isinstance(audio, bytearray):
        audio = bytes(audio)
    if not isinstance(audio, bytes):
        raise TypeError(f"TTS returned {type(audio).__name__}, expected bytes")

    # 2) Normalize (skipped by design to avoid decoding)
    try:
        audio = _normalize_mp3(audio)
    except Exception as e:
        log.warning("[TTS] normalization skipped: %s", e)

    # 3) Duration (ffmpeg-free heuristic)
    est_ms = _estimate_speech_ms(text)
    dur_ms = _mp3_duration_ms(audio, fallback_ms=est_ms)
    dur_ms = max(20, dur_ms)
    dur_s = dur_ms / 1000.0

    # 4) Dense ARKit-15 visemes covering the whole clip (arkit has no env/ffmpeg)
    frames = generate_visemes(text, dur_ms, audio_bytes=None)  # audio_bytes intentionally unused
    if len(frames) < 2:
        frames = [[0.0]*ARKIT_DIM, [0.0]*ARKIT_DIM]
        dur_ms = max(dur_ms, MS_PER_FRAME)
        dur_s = dur_ms / 1000.0

    # 5) Authoritative timeline 0..duration inclusive
    n = len(frames)
    step = dur_s / max(1, n - 1)
    times = [i * step for i in range(n)]
    fps = (n - 1) / max(1e-6, dur_s)

    try:
        peak = max((max(f) for f in frames), default=0.0)
        log.info("[Viseme] n=%d frame_ms=%d peak=%.3f dur=%.3fs eff_fps=%.2f",
                 n, MS_PER_FRAME, peak, dur_s, fps)
    except Exception:
        pass

    payload: VisemeData = {
        "viseme": frames,
        "viseme_times": times,
        "viseme_format": "arkit15",
        "viseme_fps": fps,
        "duration_ms": int(dur_ms),
        "frame_ms": MS_PER_FRAME,
        "viseme_profile": "arkit15-v2"
    }
    return audio, payload
