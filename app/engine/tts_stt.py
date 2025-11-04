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
# Small helpers / constants
# =========================================================

def _env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"{name} is required")
    return v

MP3_BITRATE = "192k"
DO_NORMALIZE = True              # set False to skip normalization globally
TARGET_DBFS = -1.0               # peak normalization target
ARKIT_DIM = 15
VIS_FPS = int(os.getenv("VIS_FPS", "100"))  # frames per second for visemes

def _normalize_mp3(audio_bytes: bytes) -> bytes:
    """Peak-normalize MP3 (best-effort)."""
    if not DO_NORMALIZE:
        return audio_bytes
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    try:
        change = TARGET_DBFS - seg.max_dBFS
    except Exception:
        # very old pydub can lack max_dBFS on silence; skip
        return audio_bytes
    seg2 = seg.apply_gain(change) if change > 0 else seg
    out = io.BytesIO()
    seg2.export(out, format="mp3", bitrate=MP3_BITRATE)
    return out.getvalue()

def _mp3_duration_ms(audio_bytes: bytes, fallback_ms: int) -> int:
    """Get duration using pydub; fall back if parsing fails."""
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        return int(seg.duration_seconds * 1000)
    except Exception:
        return fallback_ms

# =========================================================
# Viseme synthesis (ARKit-like, 15 dims, audio-driven)
# =========================================================

# ARKit-ish index mapping for a compact 15-dim mouth/tongue set
JAW_OPEN           = 0
MOUTH_FUNNEL       = 1
MOUTH_CLOSE        = 2
MOUTH_PUCKER       = 3
MOUTH_SMILE_L      = 4
MOUTH_SMILE_R      = 5
MOUTH_LEFT         = 6
MOUTH_RIGHT        = 7
MOUTH_FROWN_L      = 8
MOUTH_FROWN_R      = 9
MOUTH_DIMPLE_L     = 10
MOUTH_DIMPLE_R     = 11
MOUTH_STRETCH_L    = 12
MOUTH_STRETCH_R    = 13
TONGUE_OUT         = 14

_VOWELS   = set("aeiou")
_LABIALS  = set("bmp")
_FRIC     = set("fvszx")                # teeth/lips → stretch/press
_ALVEOL   = set("tdlnr")                # alveolar ridge → slight stretch & jaw
_VEL_GLOT = set("kgqh")                 # back/throat → jaw + funnel

# canonical shapes per class (base weights 0..1)
_BASE_SHAPES: Dict[str, Dict[int, float]] = {
    "VOWEL": {JAW_OPEN: .65, MOUTH_FUNNEL: .45, MOUTH_CLOSE: .05, MOUTH_PUCKER: .15,
              MOUTH_SMILE_L: .05, MOUTH_SMILE_R: .05},
    "LABIAL": {MOUTH_CLOSE: .85, MOUTH_PUCKER: .55, JAW_OPEN: .10},
    "FRIC": {MOUTH_STRETCH_L: .45, MOUTH_STRETCH_R: .45, JAW_OPEN: .20},
    "ALV": {JAW_OPEN: .25, MOUTH_STRETCH_L: .20, MOUTH_STRETCH_R: .20, TONGUE_OUT: .10},
    "VEL": {JAW_OPEN: .35, MOUTH_FUNNEL: .25},
    "PAUSE": {MOUTH_CLOSE: .15},
    "REST": {JAW_OPEN: .05, MOUTH_CLOSE: .05},
}

def _class_for_char(ch: str) -> str:
    c = ch.lower()
    if c in _VOWELS:   return "VOWEL"
    if c in _LABIALS:  return "LABIAL"
    if c in _FRIC:     return "FRIC"
    if c in _ALVEOL:   return "ALV"
    if c in _VEL_GLOT: return "VEL"
    if c in ".!?":     return "PAUSE"
    return "REST"

def _rms_envelope_from_audio(audio_bytes: bytes, frames: int) -> List[float]:
    """Compute simple 0..1 RMS per frame to drive mouth intensity."""
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        ms_per_frame = max(1, int(round(1000.0 / VIS_FPS)))
        env = []
        for i in range(frames):
            start = i * ms_per_frame
            chunk = seg[start:start + ms_per_frame]
            env.append(chunk.rms or 0)
        mx = max(env) or 1
        return [min(1.0, e / mx) for e in env]
    except Exception:
        # fallback: gentle rise/fall envelope so mouth never looks “dead”
        return [0.2 + 0.8 * (min(i, frames - 1 - i) / max(1, frames // 2)) for i in range(frames)]

def _shape_for_class(cls: str, amp: float) -> List[float]:
    out = [0.0] * ARKIT_DIM
    base = _BASE_SHAPES.get(cls, _BASE_SHAPES["REST"])
    for idx, val in base.items():
        out[idx] = val * amp
    # slight “smile” boost on strong vowels
    if cls == "VOWEL" and amp > 0.5:
        out[MOUTH_SMILE_L] = min(1.0, out[MOUTH_SMILE_L] + 0.04 * (amp - 0.5))
        out[MOUTH_SMILE_R] = min(1.0, out[MOUTH_SMILE_R] + 0.04 * (amp - 0.5))
    return out

def _spread_lr(shape: List[float], amp: float, drift: float) -> None:
    """Inject tiny L/R asymmetry to avoid robotic symmetry."""
    shape[MOUTH_LEFT]       = 0.06 * amp * max(0.0, 1.0 - abs(drift))
    shape[MOUTH_RIGHT]      = 0.06 * amp * max(0.0, 1.0 - abs(drift))
    shape[MOUTH_SMILE_L]    = min(1.0, shape[MOUTH_SMILE_L] + 0.02 * amp * max(0.0, drift))
    shape[MOUTH_SMILE_R]    = min(1.0, shape[MOUTH_SMILE_R] + 0.02 * amp * max(0.0, -drift))
    shape[MOUTH_FROWN_L]    = min(1.0, 0.02 * amp * max(0.0, -drift))
    shape[MOUTH_FROWN_R]    = min(1.0, 0.02 * amp * max(0.0, drift))
    shape[MOUTH_DIMPLE_L]   = min(1.0, 0.02 * amp)
    shape[MOUTH_DIMPLE_R]   = min(1.0, 0.02 * amp)

def _smooth(prev: List[float], cur: List[float], alpha: float = 0.6) -> List[float]:
    """Exponential smoothing between frames."""
    return [alpha * p + (1 - alpha) * c for p, c in zip(prev, cur)]

def _visemes(
    text: str,
    duration_ms: int,
    audio_bytes: bytes | None = None,
    fps: int = VIS_FPS,
) -> List[List[float]]:
    """
    Return a list of frames; each frame is a 15-element array in [0..1].
    Frames are synced to VIS_FPS and total length = duration_ms.
    """
    if duration_ms <= 0:
        duration_ms = 300
    frames = max(1, int(round(duration_ms * (fps / 1000.0))))

    # Loudness envelope (0..1) to drive amplitude
    env = _rms_envelope_from_audio(audio_bytes, frames) if audio_bytes else [0.35] * frames

    # Very rough “phoneme” classes from characters → one class per frame
    chars = [c for c in (text or "") if not c.isspace()]
    span = max(1, len(chars) // frames)
    seq = [_class_for_char(chars[i * span]) if i * span < len(chars) else "REST" for i in range(frames)]

    out: List[List[float]] = []
    prev = [0.0] * ARKIT_DIM
    for i in range(frames):
        cls = seq[i]
        amp = max(0.12, env[i])  # keep a minimum so mouth never fully dies
        cur = _shape_for_class(cls, amp)

        # periodic drift for L/R
        drift = ((i % 7) - 3) / 12.0  # ~[-0.25..0.25]
        _spread_lr(cur, amp, drift)

        # attack/release smoothing
        cur = _smooth(prev, cur, alpha=0.6)
        prev = cur

        out.append([max(0.0, min(1.0, round(v, 3))) for v in cur])

    return out

def _parse(service: Optional[str], voice_id: Optional[str]) -> Tuple[str, str]:
    """Parse 'service::voice' or separate service + voice_id."""
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

    # visemes: use real audio envelope to drive intensity
    dur = _mp3_duration_ms(audio, fallback_ms=max(300, int(len(text) * 40)))
    vis = _visemes(text, dur, audio_bytes=audio)

    return audio, vis
