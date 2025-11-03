# engine/tts_stt.py
from __future__ import annotations

import io
import os
import logging
from io import BytesIO
from time import perf_counter
from typing import Optional, List, Tuple, Callable, Awaitable, Any, Dict
from pydub import AudioSegment

log = logging.getLogger(__name__)

__all__ = [
    "transcribe_openai_async",
    "transcribe_openai_from_bytes_async",
    "synthesize_tts_async",
]

# =========================================================
# Small, boring helpers
# =========================================================

def _env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"{name} is required")
    return v

MP3_BITRATE = "192k"
DO_NORMALIZE = True            # set False to skip normalization globally
TARGET_DBFS = -1.0             # peak normalization target
ARKIT_DIM = 15
VIS_FPS = int(os.getenv("VIS_FPS", "100"))

def _normalize_mp3(audio_bytes: bytes) -> bytes:
    if not DO_NORMALIZE:
        return audio_bytes
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    change = TARGET_DBFS - seg.max_dBFS
    seg2 = seg.apply_gain(change) if change > 0 else seg
    out = io.BytesIO()
    seg2.export(out, format="mp3", bitrate=MP3_BITRATE)
    return out.getvalue()

def _mp3_duration_ms(audio_bytes: bytes, fallback_ms: int) -> int:
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        return int(seg.duration_seconds * 1000)
    except Exception:
        return fallback_ms

def _visemes(text: str, duration_ms: int, fps: int = VIS_FPS) -> List[List[float]]:
    # ultra-compact ARKit-ish approximation
    if duration_ms <= 0:
        duration_ms = 300
    frames = max(1, int(round(duration_ms * (fps / 1000.0))))
    t = (text or "").lower()
    vowels, labials = set("aeiou"), set("mbp")

    weights: List[Tuple[str, float]] = []
    for ch in t:
        if ch in vowels:
            weights.append(("v", 1.0))
        elif ch in labials:
            weights.append(("l", 0.9))
        elif ch.isalpha():
            weights.append(("c", 0.5))
        elif ch in ".!?":
            weights.append(("p", 0.2))
        else:
            weights.append(("o", 0.0))

    span = max(1, len(weights) // frames)
    out: List[List[float]] = []
    idx = 0
    for f in range(frames):
        jaw = funnel = close_ = 0.05
        for typ, w in weights[idx: idx + span]:
            if typ == "v":
                jaw += 0.5 * w; funnel += 0.4 * w
            elif typ == "l":
                close_ += 0.6 * w
            elif typ == "c":
                jaw += 0.2 * w
            elif typ == "p":
                jaw *= 0.7; funnel *= 0.7; close_ *= 0.9
        idx += span
        # light envelope for smoother edges
        attack = min(1.0, (f + 1) / 3.0)
        release = min(1.0, (frames - f) / 3.0)
        env = attack if attack < release else release
        jaw = max(0.0, min(1.0, jaw * env))
        funnel = max(0.0, min(1.0, funnel * env))
        close_ = max(0.0, min(1.0, close_ * env))
        frame = [0.0] * ARKIT_DIM
        frame[0], frame[1], frame[2] = round(jaw, 3), round(funnel, 3), round(close_, 3)
        out.append(frame)
    return out

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

# =========================================================
# STT (OpenAI Whisper)
# =========================================================

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
    return await transcribe_openai_from_bytes_async(data, filename_hint=os.path.basename(file_path), model=model)

# =========================================================
# TTS engines (all return MP3 bytes)
# =========================================================

async def _tts_gtts(text: str, voice_id: str) -> bytes:
    from gtts import gTTS
    import asyncio

    lang, tld = "en", "com"
    if "@" in (voice_id or ""):
        lang, tld = [s.strip() for s in voice_id.split("@", 1)]

    def _work() -> bytes:
        out = BytesIO()
        gTTS(text=text, lang=lang, tld=tld).write_to_fp(out)
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
    stream = client.text_to_speech.convert(voice_id=voice_id, text=text, model_id=model_id)
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
            response_format="mp3",  # IMPORTANT: not 'format'
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

_TTS: dict[str, Callable[[str, str], Awaitable[bytes]]] = {
    "gtts": _tts_gtts,
    "elevenlabs": _tts_elevenlabs,
    "openai": _tts_openai,
}

# =========================================================
# Public synth API
# =========================================================

async def synthesize_tts_async(text: str, service: str | None, voice_id: str | None) -> Tuple[bytes, List[List[float]]]:
    """
    Synthesize TTS and approximate visemes.
    Returns (mp3_bytes, viseme_frames).
    """
    if not (text or "").strip():
        raise ValueError("Empty text for TTS.")

    svc, vid = _parse(service, voice_id)
    engine = _TTS.get(svc)
    if not engine:
        raise ValueError(f"Unsupported TTS service: {svc}")

    audio = await engine(text, vid)
    if isinstance(audio, bytearray):
        audio = bytes(audio)
    if not isinstance(audio, bytes):
        raise TypeError(f"TTS returned {type(audio).__name__}, expected bytes")

    # normalize (best-effort)
    try:
        audio = _normalize_mp3(audio)
    except Exception as e:
        log.warning("[TTS] normalization skipped: %s", e)

    # visemes
    dur = _mp3_duration_ms(audio, fallback_ms=max(300, int(len(text) * 40)))
    return audio, _visemes(text, dur)
