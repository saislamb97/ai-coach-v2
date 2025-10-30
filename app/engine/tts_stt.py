# engine/tts_stt.py
from __future__ import annotations

import io
import logging
import mimetypes
import os
import re
from io import BytesIO
from time import perf_counter
from math import ceil
from typing import Optional, List, Tuple

from pydub import AudioSegment

log = logging.getLogger(__name__)

def _env(name: str, default: Optional[str] = None, *, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"{name} is required")
    return v

# ------------------ Audio normalization -------------------
def normalize_peak_mp3(audio_bytes: bytes, target_dbfs: float = -1.0) -> bytes:
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    change = target_dbfs - seg.max_dBFS
    louder = seg.apply_gain(change) if change > 0 else seg
    out = io.BytesIO()
    louder.export(out, format="mp3", bitrate="192k")
    return out.getvalue()

def normalize_lufs(audio_bytes: bytes, target_rms: float = -20.0) -> bytes:
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    change = target_rms - seg.dBFS
    normalized = seg.apply_gain(change)
    out = io.BytesIO()
    normalized.export(out, format="mp3", bitrate="192k")
    return out.getvalue()

# ---------------------- STT -------------------------------
async def _openai_transcribe_from_filelike(fh: io.BytesIO, model: str) -> str:
    from openai import AsyncOpenAI
    api_key = _env("OPENAI_API_KEY", required=True)
    client = AsyncOpenAI(api_key=api_key)
    t0 = perf_counter()
    res = await client.audio.transcriptions.create(model=model, file=fh)
    text = (getattr(res, "text", "") or "").strip()
    import re as _re
    out = _re.sub(r"\s+", " ", text) or "- No Audio -"
    log.info("[STT] ok model=%s dur=%.3fs text_len=%d", model, perf_counter() - t0, len(out))
    return out

async def transcribe_openai_from_bytes_async(
    data: bytes,
    filename_hint: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    model = (model or _env("OPENAI_STT_MODEL", "whisper-1")).strip()
    name = (filename_hint or "audio.webm").strip()
    buf = io.BytesIO(data)
    buf.name = name
    log.info("[STT] start len=%d model=%s", len(data), model)
    return await _openai_transcribe_from_filelike(buf, model)

async def transcribe_openai_async(file_path: str, model: Optional[str] = None) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    return await transcribe_openai_from_bytes_async(data, filename_hint=os.path.basename(file_path), model=model)

# -------------------- Viseme frames -----------------------
_ARKIT_DIM = 15

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

def _empty_frame() -> List[float]:
    return [0.0] * _ARKIT_DIM

def _approximate_visemes_for_text(text: str, duration_ms: int) -> List[List[float]]:
    if duration_ms <= 0:
        duration_ms = 300
    frames = max(1, int(duration_ms / 100.0))
    t = (text or "").lower()
    vowels, labials = set("aeiou"), set("mbp")

    char_w = []
    for ch in t:
        if ch in vowels:
            char_w.append(("v", 1.0))
        elif ch in labials:
            char_w.append(("l", 0.9))
        elif ch.isalpha():
            char_w.append(("c", 0.5))
        elif ch in ".!?":
            char_w.append(("p", 0.2))
        else:
            char_w.append(("o", 0.0))

    wp_frame = max(1, len(char_w) // frames)
    frames_out: List[List[float]] = []
    idx = 0
    for f in range(frames):
        jaw, funnel, close_ = 0.05, 0.05, 0.05
        win = char_w[idx: idx + wp_frame]
        idx += wp_frame
        for typ, w in win:
            if typ == "v":
                jaw += 0.5 * w
                funnel += 0.4 * w
            elif typ == "l":
                close_ += 0.6 * w
            elif typ == "c":
                jaw += 0.2 * w
            elif typ == "p":
                jaw *= 0.7; funnel *= 0.7; close_ *= 0.9
        attack = min(1.0, (f + 1) / 3.0)
        release = min(1.0, (frames - f) / 3.0)
        env = min(attack, release)
        jaw = _clamp(jaw * env); funnel = _clamp(funnel * env); close_ = _clamp(close_ * env)
        frame = _empty_frame()
        frame[0], frame[1], frame[2] = round(jaw, 3), round(funnel, 3), round(close_, 3)
        frames_out.append(frame)
    return frames_out

# ------------------------- TTS ----------------------------
async def _speak_gtts_async(sentence: str, voice_id: str) -> bytes:
    from gtts import gTTS
    lang, tld = "en", "com"
    if "@" in (voice_id or ""):
        lang, tld = [s.strip() for s in voice_id.split("@", 1)]
    import asyncio
    def _work():
        out = BytesIO()
        gTTS(text=sentence, lang=lang, tld=tld).write_to_fp(out)
        out.seek(0)
        return out.read()
    t0 = perf_counter()
    audio = await asyncio.to_thread(_work)
    log.info("[TTS][gTTS] ok lang=%s tld=%s bytes=%d dur=%.3fs", lang, tld, len(audio), perf_counter() - t0)
    return audio

async def _speak_elevenlabs_async(sentence: str, voice_id: str) -> bytes:
    from elevenlabs import ElevenLabs
    api_key = _env("ELEVENLABS_API_KEY", required=True)
    model_id = _env("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
    if not voice_id:
        raise ValueError("elevenlabs_tts: voice_id is required")
    if voice_id.lower().startswith("elevenlabs::"):
        voice_id = voice_id.split("::", 1)[1].strip()
    client = ElevenLabs(api_key=api_key)
    t0 = perf_counter()
    stream = client.text_to_speech.convert(voice_id=voice_id, text=sentence, model_id=model_id)
    audio = b"".join(stream)
    if not audio:
        raise RuntimeError("No audio returned from ElevenLabs")
    log.info("[TTS][EL] ok model=%s voice=%s bytes=%d dur=%.3fs", model_id, voice_id, len(audio), perf_counter() - t0)
    return audio

async def synthesize_tts_async(text: str, service: str, voice_id: str) -> Tuple[bytes, List[List[float]]]:
    service = (service or "").strip().lower()
    voice_id = (voice_id or "").strip()
    if not text.strip():
        raise ValueError("Empty text for TTS.")
    if not service and "::" not in voice_id:
        raise ValueError("TTS service missing.")

    if "::" in voice_id:
        pfx, raw = voice_id.split("::", 1)
        service = pfx.lower()
        voice_id = raw.strip()

    if service == "gtts":
        audio = await _speak_gtts_async(text, voice_id)
    elif service == "elevenlabs":
        audio = await _speak_elevenlabs_async(text, voice_id)
    else:
        raise ValueError(f"Unsupported TTS service: {service}")

    if isinstance(audio, bytearray):
        audio = bytes(audio)
    if not isinstance(audio, (bytes,)):
        raise TypeError(f"TTS returned {type(audio).__name__}, expected bytes")

    try:
        audio = normalize_peak_mp3(audio, target_dbfs=-1.0)
    except Exception as e:
        log.warning("[TTS] normalize_skip err=%s", e)

    try:
        seg = AudioSegment.from_file(io.BytesIO(audio), format="mp3")
        duration_ms = int(seg.duration_seconds * 1000)
    except Exception:
        duration_ms = max(300, int(len(text) * 40))

    visemes = _approximate_visemes_for_text(text, duration_ms)
    return audio, visemes
