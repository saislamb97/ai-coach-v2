# engine/tts_stt.py
from __future__ import annotations

import io
import os
import logging
from io import BytesIO
from time import perf_counter
from typing import Optional, List, Tuple, Callable, Awaitable, Dict, Any

from pydub import AudioSegment

log = logging.getLogger(__name__)

__all__ = [
    "transcribe_openai_async",
    "transcribe_openai_from_bytes_async",
    "synthesize_tts_async",
]

# =============================================================================
# Env helpers / core config
# =============================================================================

def _env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"{name} is required")
    return v

# Audio postproc
MP3_BITRATE = os.getenv("MP3_BITRATE", "192k")
DO_NORMALIZE = os.getenv("DO_NORMALIZE", "1") not in ("0", "false", "False")
TARGET_DBFS = float(os.getenv("TARGET_DBFS", "-1.0"))

# Viseme timing (frame period in ms; e.g., 24 FPS -> ~41.7 ms)
VIS_FPS = int(os.getenv("VIS_FPS", "24"))
MS_PER_FRAME = max(10, int(round(1000.0 / max(1, VIS_FPS))))
ARKIT_DIM = 15

# Viseme amplitude shaping
VIS_GAIN = float(os.getenv("VIS_GAIN", "2.0"))              # global gain
VIS_GAMMA = float(os.getenv("VIS_GAMMA", "0.6"))            # <1 boosts mid values
VIS_MIN_FLOOR = float(os.getenv("VIS_MIN_FLOOR", "0.02"))   # never completely static
VIS_SILENCE_DBFS = float(os.getenv("VIS_SILENCE_DBFS", "-55"))  # floor for silence
VIS_ATTACK_ALPHA = float(os.getenv("VIS_ATTACK_ALPHA", "0.55"))  # smoothing weights
VIS_RELEASE_ALPHA = float(os.getenv("VIS_RELEASE_ALPHA", "0.70"))

# =============================================================================
# Utilities
# =============================================================================

def _normalize_mp3(audio_bytes: bytes) -> bytes:
    """
    Peak-normalize MP3 toward TARGET_DBFS (best effort).
    """
    if not DO_NORMALIZE:
        return audio_bytes
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    try:
        change = TARGET_DBFS - seg.max_dBFS
    except Exception:
        return audio_bytes
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

# =============================================================================
# Viseme synthesis (Convai-like; 15 ARKit weights per frame)
# =============================================================================

# Compact ARKit-ish indices (15 dims)
JAW_OPEN, MOUTH_FUNNEL, MOUTH_CLOSE, MOUTH_PUCKER, \
MOUTH_SMILE_L, MOUTH_SMILE_R, MOUTH_LEFT, MOUTH_RIGHT, \
MOUTH_FROWN_L, MOUTH_FROWN_R, MOUTH_DIMPLE_L, MOUTH_DIMPLE_R, \
MOUTH_STRETCH_L, MOUTH_STRETCH_R, TONGUE_OUT = range(15)

# Stronger base shapes (so peaks can approach ~0.9 after gain/gamma)
_BASE_SHAPES: Dict[str, Dict[int, float]] = {
    "VOWEL":  {JAW_OPEN:.95,  MOUTH_FUNNEL:.75, MOUTH_CLOSE:.08, MOUTH_PUCKER:.20, MOUTH_SMILE_L:.10, MOUTH_SMILE_R:.10},
    "LABIAL": {MOUTH_CLOSE:.95, MOUTH_PUCKER:.70, JAW_OPEN:.15},                # b/p/m
    "FRIC":   {MOUTH_STRETCH_L:.70, MOUTH_STRETCH_R:.70, JAW_OPEN:.25},         # f/v/s/z/sh
    "ALV":    {JAW_OPEN:.35, MOUTH_STRETCH_L:.30, MOUTH_STRETCH_R:.30, TONGUE_OUT:.12},  # t/d/l/n/r
    "VEL":    {JAW_OPEN:.45, MOUTH_FUNNEL:.35},                                  # k/g/q/h
    "PAUSE":  {MOUTH_CLOSE:.25},
    "REST":   {JAW_OPEN:.08, MOUTH_CLOSE:.06},
}

# Optional better phoneme mapping (if 'phonemizer' is installed)
try:
    import phonemizer  # type: ignore
    from phonemizer.backend import EspeakBackend  # type: ignore

    def _phoneme_classes(text: str, lang: str = "en-us") -> List[str]:
        """
        Map phonemes -> coarse mouth classes.
        """
        backend = EspeakBackend(language=lang, punctuation_marks=";:,.!?¡¿—…" , with_stress=False)
        phones = backend.phonemize([text], strip=True, njobs=1)[0].split()
        classes: List[str] = []
        for ph in phones:
            p = ph.lower()
            if p in {"a", "e", "i", "o", "u", "@", "3", "A", "E", "I", "O", "U"}:
                classes.append("VOWEL")
            elif p in {"b", "p", "m"}:
                classes.append("LABIAL")
            elif p in {"f", "v", "s", "z", "S", "Z", "h", "x"}:
                classes.append("FRIC")
            elif p in {"t", "d", "l", "n", "r"}:
                classes.append("ALV")
            elif p in {"k", "g", "q"}:
                classes.append("VEL")
            else:
                classes.append("REST")
        return classes

    _HAS_PHONEMES = True
except Exception:
    _HAS_PHONEMES = False

_VOWELS = set("aeiou")
_LABIALS = set("bmp")
_FRIC = set("fvszx")
_ALVEOL = set("tdlnr")
_VEL_GLOT = set("kgqh")

def _char_class(c: str) -> str:
    c = c.lower()
    if c in _VOWELS:   return "VOWEL"
    if c in _LABIALS:  return "LABIAL"
    if c in _FRIC:     return "FRIC"
    if c in _ALVEOL:   return "ALV"
    if c in _VEL_GLOT: return "VEL"
    if c in ".!?":     return "PAUSE"
    return "REST"

def _db_to_unit(db: float, floor_db: float) -> float:
    # Map dBFS (<=0) to 0..1; floor_db -> 0
    if db <= floor_db:
        return 0.0
    return (db - floor_db) / (0.0 - floor_db)

def _envelope_from_audio_dbfs(audio_bytes: bytes, frames: int) -> List[float]:
    """
    dBFS per 100ms chunk → normalize by 90th percentile → apply gain/gamma.
    """
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        env = []
        for i in range(frames):
            chunk = seg[i * MS_PER_FRAME:(i + 1) * MS_PER_FRAME]
            db = chunk.dBFS if chunk.dBFS != float("-inf") else VIS_SILENCE_DBFS
            env.append(_db_to_unit(db, VIS_SILENCE_DBFS))
        if env:
            ref = sorted(env)[max(0, int(0.9 * (len(env) - 1)))]
            ref = max(ref, 1e-3)
            env = [min(1.0, (e / ref)) for e in env]
        # perceptual remap + global gain
        env = [min(1.0, VIS_GAIN * (e ** VIS_GAMMA)) for e in env]
        return env
    except Exception:
        # fallback trapezoid
        return [min(1.0, max(0.0, (min(i, frames-1-i) / max(1, frames//2)))) for i in range(frames)]

def _shape_for_class(cls: str, amp: float) -> List[float]:
    out = [0.0] * ARKIT_DIM
    for idx, base in _BASE_SHAPES.get(cls, _BASE_SHAPES["REST"]).items():
        out[idx] = base * amp
    if cls == "VOWEL" and amp > 0.5:
        out[MOUTH_SMILE_L] = min(1.0, out[MOUTH_SMILE_L] + 0.05 * (amp - 0.5))
        out[MOUTH_SMILE_R] = min(1.0, out[MOUTH_SMILE_R] + 0.05 * (amp - 0.5))
    return out

def _spread_lr(shape: List[float], amp: float, drift: float) -> None:
    # tiny asymmetry for liveliness
    shape[MOUTH_LEFT]     = 0.08 * amp * max(0.0, 1.0 - abs(drift))
    shape[MOUTH_RIGHT]    = 0.08 * amp * max(0.0, 1.0 - abs(drift))
    shape[MOUTH_DIMPLE_L] = min(1.0, shape[MOUTH_DIMPLE_L] + 0.03 * amp)
    shape[MOUTH_DIMPLE_R] = min(1.0, shape[MOUTH_DIMPLE_R] + 0.03 * amp)
    shape[MOUTH_FROWN_L]  = min(1.0, shape[MOUTH_FROWN_L] + 0.02 * amp * max(0.0, -drift))
    shape[MOUTH_FROWN_R]  = min(1.0, shape[MOUTH_FROWN_R] + 0.02 * amp * max(0.0,  drift))

def _smooth(prev: List[float], cur: List[float], alpha: float) -> List[float]:
    return [alpha * p + (1 - alpha) * c for p, c in zip(prev, cur)]

def _sequence_classes(text: str, frames: int) -> List[str]:
    """
    Produce a sequence of coarse mouth classes with length == frames.
    If phonemizer is available, use it; otherwise, character heuristic.
    """
    if _HAS_PHONEMES:
        classes = _phoneme_classes(text or "")
    else:
        chars = [c for c in (text or "") if not c.isspace()]
        classes = [_char_class(c) for c in chars]
    if not classes:
        classes = ["REST"]

    span = max(1, len(classes) // frames)
    seq = [classes[min(i * span, len(classes) - 1)] for i in range(frames)]
    return seq

def _autoscale_frames(frames: List[List[float]], target_peak: float = 0.85, hard_cap: float = 3.0) -> List[List[float]]:
    """Scale all frames so peaks land near target_peak, with a hard multiplier cap."""
    peak = 0.0
    for f in frames:
        for v in f[:ARKIT_DIM]:
            if v > peak: peak = v
    if peak <= 1e-6:
        return frames
    scale = min(hard_cap, max(0.5, target_peak / peak))
    return [[min(1.0, v * scale) for v in f] for f in frames]

def _convai_frames(text: str, duration_ms: int, audio_bytes: Optional[bytes]) -> List[List[float]]:
    if duration_ms <= 0:
        duration_ms = 300
    frames = max(1, int(round(duration_ms / MS_PER_FRAME)))

    env = _envelope_from_audio_dbfs(audio_bytes, frames) if audio_bytes else [0.35] * frames
    seq = _sequence_classes(text, frames)

    out: List[List[float]] = []
    prev = [0.0] * ARKIT_DIM
    for i in range(frames):
        amp = max(VIS_MIN_FLOOR, min(1.0, env[i]))
        cur = _shape_for_class(seq[i], amp)
        drift = ((i % 7) - 3) / 10.0  # [-0.3..0.3]
        _spread_lr(cur, amp, drift)

        # attack/release smoothing
        alpha = VIS_ATTACK_ALPHA if i == 0 or amp >= prev[JAW_OPEN] else VIS_RELEASE_ALPHA
        cur = _smooth(prev, cur, alpha)

        # clamp + round for compact payloads
        cur = [min(1.0, max(0.0, round(v, 3))) for v in cur]
        out.append(cur)
        prev = cur
    out = _autoscale_frames(out, target_peak=0.85, hard_cap=3.0)
    return out

# =============================================================================
# OpenAI Whisper STT
# =============================================================================

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

# =============================================================================
# TTS engines (return MP3 bytes)
# =============================================================================

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
            response_format="mp3",
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

# =============================================================================
# Public API
# =============================================================================

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

async def synthesize_tts_async(text: str, service: str | None, voice_id: str | None) -> Tuple[bytes, List[List[float]]]:
    """
    Synthesize TTS and produce Convai-style visemes.
    Returns: (mp3_bytes, frames[frame_idx][15 weights in 0..1])
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

    # Normalize (best-effort)
    try:
        audio = _normalize_mp3(audio)
    except Exception as e:
        log.warning("[TTS] normalization skipped: %s", e)

    # Convai-like visemes (100ms index timing by default)
    dur = _mp3_duration_ms(audio, fallback_ms=max(300, int(len(text) * 40)))
    vis = _convai_frames(text, dur, audio_bytes=audio)

    # Debug: report amplitude spread for quick tuning
    try:
        peak = max((max(frame) for frame in vis), default=0.0)
        log.info("[Viseme] frames=%d ms_per_frame=%d peak=%.3f", len(vis), MS_PER_FRAME, peak)
    except Exception:
        pass

    return audio, vis
