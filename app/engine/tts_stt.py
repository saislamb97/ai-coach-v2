# engine/tts_stt.py
from __future__ import annotations

import io
import os
import json
import math
import logging
from io import BytesIO
from time import perf_counter
from typing import Optional, List, Tuple, Callable, Awaitable, Dict, Any, TypedDict

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

# =============================================================================
# Viseme timing
# =============================================================================
VIS_FPS = int(os.getenv("VIS_FPS", "60"))
MS_PER_FRAME = max(10, int(round(1000.0 / max(1, VIS_FPS))))
ARKIT_DIM = 15  # ARKit-15 compact order used throughout

# Amplitude shaping / smoothing
VIS_GAIN = float(os.getenv("VIS_GAIN", "2.0"))
VIS_GAMMA = float(os.getenv("VIS_GAMMA", "0.6"))
VIS_MIN_FLOOR = float(os.getenv("VIS_MIN_FLOOR", "0.02"))
VIS_SILENCE_DBFS = float(os.getenv("VIS_SILENCE_DBFS", "-55"))

# Inertia (higher alpha = slower)
VIS_ATTACK_ALPHA = float(os.getenv("VIS_ATTACK_ALPHA", "0.72"))
VIS_RELEASE_ALPHA = float(os.getenv("VIS_RELEASE_ALPHA", "0.86"))

# Extra low-pass each frame (0..1, small)
VIS_EXTRA_SMOOTH = float(os.getenv("VIS_EXTRA_SMOOTH", "0.15"))

# Smooth the loudness envelope to avoid chattery jaw (frames)
VIS_ENV_SMOOTH_WIN = int(os.getenv("VIS_ENV_SMOOTH_WIN", "5"))

# L/R drift (natural asymmetry)
VIS_DRIFT_PERIOD = int(os.getenv("VIS_DRIFT_PERIOD", "11"))
VIS_DRIFT_AMPL = float(os.getenv("VIS_DRIFT_AMPL", "0.12"))

# Micro jitter (subtle, low-frequency life-like movement)
VIS_MICRO_JITTER = float(os.getenv("VIS_MICRO_JITTER", "0.015"))
VIS_MICRO_FREQ = float(os.getenv("VIS_MICRO_FREQ", "0.8"))  # Hz

# Final shaping / scaling
VIS_TARGET_PEAK = float(os.getenv("VIS_TARGET_PEAK", "0.85"))
VIS_COLUMN_GAIN = float(os.getenv("VIS_COLUMN_GAIN", "1.00"))
VIS_SHAPE_GAINS = os.getenv("VIS_SHAPE_GAINS", "").strip()
VIS_HARD_CAP = float(os.getenv("VIS_HARD_CAP", "3.0"))

# Extra natural-talking tunables
VIS_JAW_ENV_GAIN     = float(os.getenv("VIS_JAW_ENV_GAIN", "0.55"))   # how strongly env opens jaw
VIS_JAW_ONSET_BOOST  = float(os.getenv("VIS_JAW_ONSET_BOOST", "0.25"))# extra jaw pop on syllable onsets
VIS_ONSET_THRESHOLD  = float(os.getenv("VIS_ONSET_THRESHOLD", "0.06"))# min delta(env) to count as onset
VIS_SMILE_BIAS       = float(os.getenv("VIS_SMILE_BIAS", "0.35"))     # lower = less constant smiling

# Make default attack faster and extra smoothing lighter unless user overrides
if "VIS_ATTACK_ALPHA" not in os.environ:
    VIS_ATTACK_ALPHA = 0.48  # quicker openings feel more speech-like
if "VIS_EXTRA_SMOOTH" not in os.environ:
    VIS_EXTRA_SMOOTH = 0.10

# =============================================================================
# Utilities
# =============================================================================

def _normalize_mp3(audio_bytes: bytes) -> bytes:
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

def _moving_average(seq: List[float], win: int) -> List[float]:
    if not seq or win <= 1:
        return seq
    win = max(2, min(win, len(seq)))
    out: List[float] = []
    acc = 0.0
    buf: List[float] = []
    for v in seq:
        buf.append(v); acc += v
        if len(buf) > win:
            acc -= buf.pop(0)
        out.append(acc / len(buf))
    if len(out) > 2:
        out[-1] = (out[-1] + out[-2]) / 2.0
    return out

def _lf_sine(frame_idx: int, fps: int, freq_hz: float, phase: float = 0.0) -> float:
    t = frame_idx / max(1, fps)
    return math.sin(2.0 * math.pi * freq_hz * t + phase)

# =============================================================================
# Dense viseme synthesis (ARKit-15)
# =============================================================================

# ARKit-ish compact indices
JAW_OPEN, MOUTH_FUNNEL, MOUTH_CLOSE, MOUTH_PUCKER, \
MOUTH_SMILE_L, MOUTH_SMILE_R, MOUTH_LEFT, MOUTH_RIGHT, \
MOUTH_FROWN_L, MOUTH_FROWN_R, MOUTH_DIMPLE_L, MOUTH_DIMPLE_R, \
MOUTH_STRETCH_L, MOUTH_STRETCH_R, TONGUE_OUT = range(15)

# Base blends per coarse class (rebalanced for speech realism)
_BASE_SHAPES: Dict[str, Dict[int, float]] = {
    "VOWEL":  {JAW_OPEN:.98,  MOUTH_FUNNEL:.62, MOUTH_CLOSE:.05, MOUTH_PUCKER:.18,
               MOUTH_SMILE_L:.06*VIS_SMILE_BIAS, MOUTH_SMILE_R:.06*VIS_SMILE_BIAS},
    "LABIAL": {MOUTH_CLOSE:.92, MOUTH_PUCKER:.58, JAW_OPEN:.18},                          # b/p/m
    "FRIC":   {MOUTH_STRETCH_L:.72, MOUTH_STRETCH_R:.72, JAW_OPEN:.30},                   # f/v/s/z/sh
    "ALV":    {JAW_OPEN:.42, MOUTH_STRETCH_L:.32, MOUTH_STRETCH_R:.32, TONGUE_OUT:.14},   # t/d/l/n/r
    "VEL":    {JAW_OPEN:.50, MOUTH_FUNNEL:.32},                                           # k/g/q/h
    "PAUSE":  {MOUTH_CLOSE:.28},
    "REST":   {JAW_OPEN:.10, MOUTH_CLOSE:.06},
}

# Optional phoneme classes (only g2p_en; otherwise fallback to char classes)
VIS_USE_G2P = os.getenv("VIS_USE_G2P", "1").lower() not in ("0", "false")
_HAS_G2P = False
_g2p = None  # type: ignore

try:
    if VIS_USE_G2P:
        from g2p_en import G2p  # lightweight preferred backend
        _g2p = G2p()
        _HAS_G2P = True
        log.info("[Viseme] using g2p_en backend")
except Exception as e:
    log.info("[Viseme] g2p_en unavailable, using character-class fallback (%s)", e)

_VOWEL_ARPA = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
_LAB_ARPA   = {"P","B","M"}
_FRIC_ARPA  = {"F","V","S","Z","SH","ZH","HH","TH","DH"}
_ALV_ARPA   = {"T","D","L","N","R"}
_VEL_ARPA   = {"K","G","NG"}

def _arpa_to_class(ph: str) -> str:
    base = "".join(ch for ch in (ph or "") if not ch.isdigit())
    if base in _VOWEL_ARPA: return "VOWEL"
    if base in _LAB_ARPA:   return "LABIAL"
    if base in _FRIC_ARPA:  return "FRIC"
    if base in _ALV_ARPA:   return "ALV"
    if base in _VEL_ARPA:   return "VEL"
    return "REST"

def _g2p_classes(text: str) -> List[str]:
    if not _HAS_G2P or not (text or "").strip():
        return []
    phones: List[str] = []
    try:
        for tok in _g2p(text or ""):  # type: ignore
            if tok and tok != " ":
                phones.append(tok)
    except Exception:
        phones = []
    return [_arpa_to_class(ph) for ph in phones] if phones else []

# cheap fallback if no G2P
_VOWELS=set("aeiou"); _LABIALS=set("bmp"); _FRIC=set("fvszx"); _ALVEOL=set("tdlnr"); _VEL_GLOT=set("kgqh")
def _char_class(c: str) -> str:
    c=c.lower()
    if c in _VOWELS:   return "VOWEL"
    if c in _LABIALS:  return "LABIAL"
    if c in _FRIC:     return "FRIC"
    if c in _ALVEOL:   return "ALV"
    if c in _VEL_GLOT: return "VEL"
    if c in ".!?":     return "PAUSE"
    return "REST"

def _is_vowel_class(cls: str) -> bool:
    return cls == "VOWEL"

def _is_voiced_class(cls: str) -> bool:
    # treat everything except hard PAUSE/REST as voiced-ish for jaw biasing
    return cls in ("VOWEL", "LABIAL", "FRIC", "ALV", "VEL")

def _db_to_unit(db: float, floor_db: float) -> float:
    if db <= floor_db: return 0.0
    return (db - floor_db) / (0.0 - floor_db)

def _envelope_from_audio_dbfs(audio_bytes: bytes, frames: int) -> List[float]:
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        env=[]
        for i in range(frames):
            chunk = seg[i*MS_PER_FRAME:(i+1)*MS_PER_FRAME]
            db = chunk.dBFS if chunk.dBFS != float("-inf") else VIS_SILENCE_DBFS
            env.append(_db_to_unit(db, VIS_SILENCE_DBFS))
        if env:
            ref = sorted(env)[max(0, int(0.9*(len(env)-1)))]
            ref = max(ref, 1e-3)
            env = [min(1.0, (e/ref)) for e in env]
        env = [min(1.0, VIS_GAIN * (e ** VIS_GAMMA)) for e in env]
        if VIS_ENV_SMOOTH_WIN > 1:
            env = _moving_average(env, VIS_ENV_SMOOTH_WIN)
        return env
    except Exception:
        raw = [min(1.0, max(0.0, (min(i, frames-1-i)/max(1, frames//2)))) for i in range(frames)]
        if VIS_ENV_SMOOTH_WIN > 1:
            raw = _moving_average(raw, VIS_ENV_SMOOTH_WIN)
        return raw

def _shape_for_class(cls: str, amp: float) -> List[float]:
    out=[0.0]*ARKIT_DIM
    for idx, base in _BASE_SHAPES.get(cls, _BASE_SHAPES["REST"]).items():
        out[idx]=base*amp
    # reduced smile creep; gated by bias
    if cls=="VOWEL" and amp>0.55:
        out[MOUTH_SMILE_L]=min(1.0, out[MOUTH_SMILE_L]+0.03*(amp-0.55)*VIS_SMILE_BIAS)
        out[MOUTH_SMILE_R]=min(1.0, out[MOUTH_SMILE_R]+0.03*(amp-0.55)*VIS_SMILE_BIAS)
    return out

def _spread_lr(shape: List[float], amp: float, drift: float) -> None:
    lateral = max(0.0, 1.0 - abs(drift))
    shape[MOUTH_LEFT]     = 0.12 * amp * lateral
    shape[MOUTH_RIGHT]    = 0.12 * amp * lateral
    shape[MOUTH_DIMPLE_L] = min(1.0, shape[MOUTH_DIMPLE_L] + 0.06 * amp)
    shape[MOUTH_DIMPLE_R] = min(1.0, shape[MOUTH_DIMPLE_R] + 0.06 * amp)
    shape[MOUTH_FROWN_L]  = min(1.0, shape[MOUTH_FROWN_L] + 0.035 * amp * max(0.0, -drift))
    shape[MOUTH_FROWN_R]  = min(1.0, shape[MOUTH_FROWN_R] + 0.035 * amp * max(0.0,  drift))

def _smooth(prev: List[float], cur: List[float], alpha: float) -> List[float]:
    return [alpha*p + (1-alpha)*c for p,c in zip(prev,cur)]

def _sequence_classes(text: str, frames: int) -> List[str]:
    classes = _g2p_classes(text or "")
    if not classes:
        chars = [c for c in (text or "") if not c.isspace()]
        classes = [_char_class(c) for c in chars] or ["REST"]
    span = max(1, len(classes)//frames)
    return [classes[min(i*span, len(classes)-1)] for i in range(frames)]

def _parse_shape_gains(s: str) -> List[float]:
    if not s:
        return [1.0] * ARKIT_DIM
    try:
        v = json.loads(s)
        if isinstance(v, dict):
            out = [1.0] * ARKIT_DIM
            for k, val in v.items():
                try:
                    idx = int(k)
                    if 0 <= idx < ARKIT_DIM:
                        out[idx] = float(val)
                except Exception:
                    pass
            return out
        if isinstance(v, (list, tuple)) and len(v) >= ARKIT_DIM:
            return [float(x) for x in v[:ARKIT_DIM]]
    except Exception:
        pass
    try:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) >= ARKIT_DIM:
            return parts[:ARKIT_DIM]
    except Exception:
        pass
    return [1.0] * ARKIT_DIM

def _apply_column_gains(frames: List[List[float]], scalar: float, per_shape: List[float]) -> List[List[float]]:
    if not frames:
        return frames
    s = scalar if scalar and scalar > 0 else 1.0
    g = per_shape if per_shape and len(per_shape) == ARKIT_DIM else [1.0] * ARKIT_DIM
    out: List[List[float]] = []
    for f in frames:
        out.append([min(1.0, max(0.0, v * s * g[i])) for i, v in enumerate(f)])
    return out

def _autoscale_frames(frames: List[List[float]], target_peak: float = VIS_TARGET_PEAK, hard_cap: float = VIS_HARD_CAP) -> List[List[float]]:
    peak = max((max(f[:ARKIT_DIM]) for f in frames), default=0.0)
    if peak <= 1e-6:
        return frames
    scale = min(hard_cap, max(0.5, target_peak / peak))
    return [[min(1.0, v * scale) for v in f] for f in frames]

def _neutralize_edges(frames: List[List[float]]) -> None:
    if not frames:
        return
    frames[0] = [0.0]*ARKIT_DIM
    frames[-1] = [0.0]*ARKIT_DIM

def _convai_frames(text: str, duration_ms: int, audio_bytes: Optional[bytes]) -> List[List[float]]:
    """
    Main viseme generator (ARKit-15):
      1) envelope & phonetic class sequence (with envelope smoothing)
      2) shape per frame + inertia smoothing + asymmetry
      3) micro-jitter for life-like movement
      4) per-column gains
      5) final autoscale toward VIS_TARGET_PEAK
    """
    if duration_ms <= 0:
        duration_ms = 300
    frames = max(2, int(round(duration_ms / MS_PER_FRAME)))

    env = _envelope_from_audio_dbfs(audio_bytes, frames) if audio_bytes else [0.35]*frames
    seq = _sequence_classes(text, frames)

    # Onset detector: positive first difference of smoothed envelope
    env_diff = [0.0]*frames
    for i in range(1, frames):
        d = env[i] - env[i-1]
        env_diff[i] = d if d > 0 else 0.0
    if VIS_ENV_SMOOTH_WIN > 1:
        env_diff = _moving_average(env_diff, max(2, min(3, VIS_ENV_SMOOTH_WIN//2)))

    out: List[List[float]] = []
    prev = [0.0]*ARKIT_DIM

    # smooth sine drift in [-1,1]
    drift_wave = [math.sin(2.0*math.pi * (i / max(1.0, float(VIS_DRIFT_PERIOD)))) for i in range(frames)]

    for i in range(frames):
        amp = max(VIS_MIN_FLOOR, min(1.0, env[i]))
        onset = max(0.0, env_diff[i] - VIS_ONSET_THRESHOLD)
        cls = seq[i]

        # 1) base pose from class
        cur = _shape_for_class(cls, amp)

        # 2) jaw driven by energy + vowel bias + onset pop
        jaw_env = min(1.0, (VIS_JAW_ENV_GAIN * amp) + (0.18 * amp * amp))  # quadratic opens more on loud vowels
        if _is_vowel_class(cls):
            jaw_env = min(1.0, jaw_env + 0.12*amp)  # vowels should open more
        jaw_boost = max(0.0, VIS_JAW_ONSET_BOOST * onset)
        cur[JAW_OPEN] = min(1.0, max(cur[JAW_OPEN], jaw_env + jaw_boost))

        # 3) unclench: when speaking, reduce mouthClose pressure so lips part
        if _is_voiced_class(cls) and amp > 0.12:
            cur[MOUTH_CLOSE] *= (0.45 + 0.3*(1.0 - amp))  # less close with louder amp

        # 4) onset also narrows lips a bit (funnel) to show articulation bursts
        if onset > 0.0:
            cur[MOUTH_FUNNEL] = min(1.0, cur[MOUTH_FUNNEL] + 0.15*onset)

        # 5) natural asymmetry + cheeks (scaled by drift amplitude)
        drift = drift_wave[i] * VIS_DRIFT_AMPL
        _spread_lr(cur, amp, drift)

        # 6) inertia smoothing (faster attack / slower release)
        alpha = VIS_ATTACK_ALPHA if (i == 0 or cur[JAW_OPEN] >= prev[JAW_OPEN]) else VIS_RELEASE_ALPHA
        cur = _smooth(prev, cur, alpha)

        # 7) extra global low-pass for creaminess
        if VIS_EXTRA_SMOOTH > 1e-3:
            cur = _smooth(cur, prev, VIS_EXTRA_SMOOTH)

        # 8) subtle micro-jitter
        if VIS_MICRO_JITTER > 0.0:
            jitter = VIS_MICRO_JITTER * _lf_sine(i, VIS_FPS, VIS_MICRO_FREQ, phase=0.5)
            cur[MOUTH_SMILE_L] = min(1.0, max(0.0, cur[MOUTH_SMILE_L] + jitter))
            cur[MOUTH_SMILE_R] = min(1.0, max(0.0, cur[MOUTH_SMILE_R] - jitter))
            cur[MOUTH_DIMPLE_L] = min(1.0, max(0.0, cur[MOUTH_DIMPLE_L] + 0.6*jitter))
            cur[MOUTH_DIMPLE_R] = min(1.0, max(0.0, cur[MOUTH_DIMPLE_R] - 0.6*jitter))

        # 9) clamp + quantize
        cur = [min(1.0, max(0.0, round(v, 3))) for v in cur]
        out.append(cur)
        prev = cur

    out = _apply_column_gains(out, VIS_COLUMN_GAIN, _parse_shape_gains(VIS_SHAPE_GAINS))
    out = _autoscale_frames(out, target_peak=VIS_TARGET_PEAK, hard_cap=VIS_HARD_CAP)
    _neutralize_edges(out)  # ensure clean open/close at clip edges
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

class VisemeData(TypedDict):
    viseme: List[List[float]]      # ARKit-15 frames
    viseme_times: List[float]      # seconds, length == len(viseme), spans [0, duration]
    viseme_format: str             # "arkit15"
    viseme_fps: float              # (N-1)/duration_seconds
    duration_ms: int               # exact MP3 duration
    frame_ms: int                  # MS_PER_FRAME (for reference)

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
    """
    Synthesize TTS and produce dense ARKit-15 visemes with an exact, backend-defined timeline.

    Returns:
        (audio_mp3_bytes, {
            "viseme": [[15]*N],          # ARKit-15 frames
            "viseme_times": [float]*N,   # seconds, 0..duration inclusive
            "viseme_format": "arkit15",
            "viseme_fps": float,         # (N-1)/duration_seconds (for logging)
            "duration_ms": int,          # exact MP3 duration
            "frame_ms": int              # MS_PER_FRAME
        })
    """
    if not (text or "").strip():
        raise ValueError("Empty text for TTS.")

    svc, vid = _parse(service, voice_id)
    engine = _TTS.get(svc)
    if not engine:
        raise ValueError(f"Unsupported TTS service: {svc}")

    # 1) TTS audio
    audio = await engine(text, vid)
    if isinstance(audio, bytearray):
        audio = bytes(audio)
    if not isinstance(audio, bytes):
        raise TypeError(f"TTS returned {type(audio).__name__}, expected bytes")

    # 2) Normalize (best-effort)
    try:
        audio = _normalize_mp3(audio)
    except Exception as e:
        log.warning("[TTS] normalization skipped: %s", e)

    # 3) Exact duration from the actual MP3
    dur_ms = _mp3_duration_ms(audio, fallback_ms=max(300, int(len(text) * 40)))
    dur_ms = max(20, dur_ms)  # guard
    dur_s = dur_ms / 1000.0

    # 4) Dense per-frame ARKit-15 visemes covering the WHOLE clip duration
    frames = _convai_frames(text, dur_ms, audio_bytes=audio)
    if len(frames) < 2:
        frames = [[0.0]*ARKIT_DIM, [0.0]*ARKIT_DIM]
        dur_ms = max(dur_ms, MS_PER_FRAME)
        dur_s = dur_ms / 1000.0

    # 5) Authoritative timeline 0..duration (inclusive); matches audioEl.duration
    n = len(frames)
    step = dur_s / max(1, n - 1)
    times = [i * step for i in range(n)]
    fps = (n - 1) / dur_s

    # Minimal log for debugging
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
    }
    return audio, payload
