# engine/arkit.py
from __future__ import annotations

import math
import logging
from typing import List, Dict, Any, Optional

log = logging.getLogger(__name__)

__all__ = ["ARKIT_DIM", "MS_PER_FRAME", "generate_visemes"]

# -----------------------------------------------------------------------------
# Self-contained config (no envs)
# -----------------------------------------------------------------------------
VIS_FPS: int = 90
MS_PER_FRAME: int = max(10, int(round(1000.0 / max(1, VIS_FPS))))
ARKIT_DIM: int = 15  # ARKit-15 compact order

# Natural look: no global exaggeration and gentle per-channel gains
EXAGGERATION: float = 1.00
GAIN_JAW: float      = 1.25
GAIN_FUNNEL: float   = 1.05
GAIN_PUCKER: float   = 1.05
GAIN_STRETCH: float  = 0.95
GAIN_CLOSE: float    = 0.85
GAIN_SMILE: float    = 0.50
TARGET_PEAK: float   = 0.92   # a bit lower peak for headroom
HARD_CAP: float      = 2.0

# Dynamics (smoother openings/closings)
ATTACK_ALPHA: float  = 0.68   # toward prev -> slower; we still keep some snap
RELEASE_ALPHA: float = 0.90
EXTRA_SMOOTH: float  = 0.12
MIN_FLOOR: float     = 0.02

# Subtle drift only (no jitter)
DRIFT_PERIOD_FR: int = 11
DRIFT_AMPL: float    = 0.06
MICRO_JITTER: float  = 0.0
MICRO_FREQ_HZ: float = 0.85

# Onset/phoneme shaping
JAW_ENV_GAIN: float    = 0.84
JAW_ONSET_BOOST: float = 0.16
ONSET_THRESHOLD: float = 0.08
SMILE_BIAS: float      = 0.0   # disable smile creep

# -----------------------------------------------------------------------------
# ARKit 15 compact indices
# -----------------------------------------------------------------------------
JAW_OPEN, MOUTH_FUNNEL, MOUTH_CLOSE, MOUTH_PUCKER, \
MOUTH_SMILE_L, MOUTH_SMILE_R, MOUTH_LEFT, MOUTH_RIGHT, \
MOUTH_FROWN_L, MOUTH_FROWN_R, MOUTH_DIMPLE_L, MOUTH_DIMPLE_R, \
MOUTH_STRETCH_L, MOUTH_STRETCH_R, TONGUE_OUT = range(15)

# Base blends — neutral smiles, softer stretch
_BASE_SHAPES: Dict[str, Dict[int, float]] = {
    "VOWEL":  {JAW_OPEN:.95,  MOUTH_FUNNEL:.58, MOUTH_CLOSE:.05, MOUTH_PUCKER:.18},
    "LABIAL": {MOUTH_CLOSE:.95, MOUTH_PUCKER:.50, JAW_OPEN:.14},
    "FRIC":   {MOUTH_STRETCH_L:.42, MOUTH_STRETCH_R:.42, JAW_OPEN:.32},
    "ALV":    {JAW_OPEN:.44, MOUTH_STRETCH_L:.22, MOUTH_STRETCH_R:.22, TONGUE_OUT:.14},
    "VEL":    {JAW_OPEN:.50, MOUTH_FUNNEL:.30},
    "PAUSE":  {MOUTH_CLOSE:.30},
    "REST":   {JAW_OPEN:.10, MOUTH_CLOSE:.06},
}

# -----------------------------------------------------------------------------
# Lightweight phoneme classes
# -----------------------------------------------------------------------------
_VOWEL_ARPA = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
_LAB_ARPA   = {"P","B","M"}
_FRIC_ARPA  = {"F","V","S","Z","SH","ZH","HH","TH","DH"}
_ALV_ARPA   = {"T","D","L","N","R"}
_VEL_ARPA   = {"K","G","NG"}

def _arpa_to_class(ph: str) -> str:
    base = "".join(ch for ch in (ph or "") if not ch.isdigit()).upper()
    if base in _VOWEL_ARPA: return "VOWEL"
    if base in _LAB_ARPA:   return "LABIAL"
    if base in _FRIC_ARPA:  return "FRIC"
    if base in _ALV_ARPA:   return "ALV"
    if base in _VEL_ARPA:   return "VEL"
    return "REST"

_G2P = None
try:
    from g2p_en import G2p  # type: ignore
    _G2P = G2p()
    log.info("[ARKIT] g2p_en active for phoneme sequence.")
except Exception:
    log.info("[ARKIT] g2p_en not installed, falling back to character classes.")

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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _smooth(prev: List[float], cur: List[float], alpha: float) -> List[float]:
    return [alpha*p + (1-alpha)*c for p,c in zip(prev,cur)]

def _spread_lr(shape: List[float], amp: float, drift: float) -> None:
    # milder corners to avoid smile impression
    lateral = max(0.0, 1.0 - abs(drift))
    shape[MOUTH_LEFT]     = 0.06 * amp * lateral
    shape[MOUTH_RIGHT]    = 0.06 * amp * lateral
    shape[MOUTH_DIMPLE_L] = min(1.0, shape[MOUTH_DIMPLE_L] + 0.015 * amp)
    shape[MOUTH_DIMPLE_R] = min(1.0, shape[MOUTH_DIMPLE_R] + 0.015 * amp)
    shape[MOUTH_FROWN_L]  = min(1.0, shape[MOUTH_FROWN_L] + 0.012 * amp * max(0.0, -drift))
    shape[MOUTH_FROWN_R]  = min(1.0, shape[MOUTH_FROWN_R] + 0.012 * amp * max(0.0,  drift))

def _shape_for_class(cls: str, amp: float) -> List[float]:
    out = [0.0]*ARKIT_DIM
    for idx, base in _BASE_SHAPES.get(cls, _BASE_SHAPES["REST"]).items():
        out[idx] = base * amp
    return out

def _autoscale(frames: List[List[float]], target_peak: float = TARGET_PEAK, hard_cap: float = HARD_CAP) -> List[List[float]]:
    peak = max((max(f[:ARKIT_DIM]) for f in frames), default=0.0)
    if peak <= 1e-6: return frames
    scale = min(hard_cap, max(0.5, target_peak / peak))
    return [[min(1.0, v * scale) for v in f] for f in frames]

def _neutralize_edges(frames: List[List[float]]) -> None:
    if not frames: return
    frames[0] = [0.0]*ARKIT_DIM
    frames[-1] = [0.0]*ARKIT_DIM

def _ease_inout(t: float) -> float:
    # smooth Hermite curve 3t^2 - 2t^3
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return t*t*(3 - 2*t)

# -----------------------------------------------------------------------------
# Text → (phoneme classes, durations)
# -----------------------------------------------------------------------------
def _phoneme_classes(text: str) -> List[str]:
    if _G2P:
        try:
            phones = []
            for tok in _G2P(text or ""):  # type: ignore
                if tok and tok != " ":
                    phones.append(tok)
            if phones:
                return [_arpa_to_class(p) for p in phones]
        except Exception:
            pass
    chars = [c for c in (text or "") if not c.isspace()]
    return [_char_class(c) for c in chars] or ["REST"]

def _unit_base_ms(cls: str, nxt: Optional[str]) -> int:
    if cls == "VOWEL":  return 140
    if cls == "LABIAL": return 90
    if cls == "FRIC":   return 105
    if cls == "ALV":    return 95
    if cls == "VEL":    return 95
    if cls == "PAUSE":  return 220
    return 70

def _amplitude_for_class(cls: str) -> float:
    if cls == "VOWEL":  return 1.00
    if cls == "LABIAL": return 0.70
    if cls == "FRIC":   return 0.60
    if cls == "ALV":    return 0.66
    if cls == "VEL":    return 0.66
    if cls == "PAUSE":  return 0.10
    return 0.35

def _sequence_from_text(text: str, frames: int, duration_ms: int) -> List[Dict[str, Any]]:
    classes = _phoneme_classes(text) or ["REST"]
    units: List[Dict[str, Any]] = []
    total = 0
    for i, cls in enumerate(classes):
        base = _unit_base_ms(cls, classes[i+1] if i+1 < len(classes) else None)
        if i < len(text) and text[i] in ".!?":
            cls, base = "PAUSE", 240
        elif i < len(text) and text[i] in ",;:":
            cls, base = "PAUSE", 160
        amp = max(MIN_FLOOR, _amplitude_for_class(cls))
        units.append({"cls": cls, "dur": base, "amp": amp}); total += base
    if not units:
        units = [{"cls":"REST", "dur":300, "amp":0.35}]; total = 300

    scale = max(0.25, duration_ms / max(1, total))
    t = 0
    for u in units:
        u["dur"] = max(MS_PER_FRAME, int(round(u["dur"] * scale)))
        u["start_ms"] = t; t += u["dur"]; u["end_ms"] = t
    if units and units[-1]["end_ms"] != duration_ms:
        units[-1]["end_ms"] = duration_ms
        units[-1]["dur"] = max(MS_PER_FRAME, duration_ms - units[-1]["start_ms"])
    return units

# -----------------------------------------------------------------------------
# Core generator
# -----------------------------------------------------------------------------
def generate_visemes(text: str, duration_ms: int, audio_bytes: Optional[bytes]) -> List[List[float]]:
    duration_ms = max(20, int(duration_ms or 300))
    n_frames = max(2, int(round(duration_ms / MS_PER_FRAME)))
    units = _sequence_from_text(text or "", n_frames, duration_ms)

    out: List[List[float]] = []
    prev = [0.0]*ARKIT_DIM
    drift_wave = [math.sin(2.0*math.pi * (i / max(1.0, float(DRIFT_PERIOD_FR)))) for i in range(n_frames)]

    # mark unit boundaries for mild onsets
    onset_flags = [0.0]*n_frames
    for i in range(1, n_frames):
        t_ms = i * MS_PER_FRAME
        if any(u["start_ms"] <= t_ms < u["start_ms"] + MS_PER_FRAME for u in units):
            onset_flags[i] = 1.0

    def _unit_index_at(t_ms: int) -> int:
        for idx, u in enumerate(units):
            if u["start_ms"] <= t_ms < u["end_ms"]:
                return idx
        return len(units)-1

    def _shape_for_unit(u: Dict[str, Any]) -> List[float]:
        amp = max(MIN_FLOOR, min(1.0, float(u["amp"])))
        s = _shape_for_class(u["cls"], amp)

        # Jaw envelope
        jaw_env = min(1.0, (JAW_ENV_GAIN * amp) + (0.22 * amp * amp))
        if u["cls"] == "VOWEL": jaw_env = min(1.0, jaw_env + 0.14*amp)
        s[JAW_OPEN] = max(s[JAW_OPEN], jaw_env)

        # Speech clamp: keep lips from fully closing on voiced segments
        if u["cls"] != "LABIAL" and amp > 0.10:
            s[MOUTH_CLOSE] *= (0.34 + 0.34*(1.0 - amp))

        # Labials: enforce closure + reduce jaw
        if u["cls"] == "LABIAL":
            s[MOUTH_CLOSE] = max(s[MOUTH_CLOSE], 0.85 * amp)
            s[JAW_OPEN]    = min(s[JAW_OPEN], 0.25 * (0.8 + 0.2*amp))

        return s

    for i in range(n_frames):
        t_ms = i * MS_PER_FRAME
        idx = _unit_index_at(t_ms)
        u = units[idx]

        # Per-unit phase [0..1] with eased interior for smooth transitions
        phase = (t_ms - u["start_ms"]) / max(1.0, float(u["dur"]))
        phase_e = _ease_inout(phase)

        # Base shapes for prev/cur/next (for coarticulation)
        cur = _shape_for_unit(u)
        prev_u = units[idx-1] if idx-1 >= 0 else None
        next_u = units[idx+1] if idx+1 < len(units) else None
        prev_shape = _shape_for_unit(prev_u) if prev_u else cur
        next_shape = _shape_for_unit(next_u) if next_u else cur

        # Coarticulation cross-fade: 30% head/tail blending windows
        prev_w = max(0.0, (0.30 - phase_e) / 0.30) if prev_u else 0.0
        next_w = max(0.0, (phase_e - 0.70) / 0.30) if next_u else 0.0
        cur_w  = 1.0
        w_sum  = max(1e-6, prev_w + cur_w + next_w)
        mix = [ (prev_shape[k]*prev_w + cur[k]*cur_w + next_shape[k]*next_w) / w_sum for k in range(ARKIT_DIM) ]
        cur = mix

        # Onset rounding: tiny funnel lift on detected boundary
        onset = max(0.0, onset_flags[i] - ONSET_THRESHOLD)
        if onset > 1e-6:
            cur[MOUTH_FUNNEL] = min(1.0, cur[MOUTH_FUNNEL] + 0.12*onset)

        # L/R spread + tiny drift
        drift = drift_wave[i] * DRIFT_AMPL
        _spread_lr(cur, max(MIN_FLOOR, float(u["amp"])), drift)

        # Anti-smile / anti-stretch when open
        jaw = cur[JAW_OPEN]
        vowelish = max(jaw, 0.6*cur[MOUTH_FUNNEL] + 0.4*cur[MOUTH_PUCKER])
        smile_k = max(0.0, 1.0 - 0.85*vowelish)
        cur[MOUTH_SMILE_L] *= smile_k
        cur[MOUTH_SMILE_R] *= smile_k
        dimple_k = max(0.0, 1.0 - 0.60*vowelish)
        cur[MOUTH_DIMPLE_L] *= dimple_k
        cur[MOUTH_DIMPLE_R] *= dimple_k
        stretch_k = max(0.65, 1.0 - 0.35*vowelish)
        cur[MOUTH_STRETCH_L] *= stretch_k
        cur[MOUTH_STRETCH_R] *= stretch_k

        # Directional temporal smoothing
        alpha = ATTACK_ALPHA if (i == 0 or cur[JAW_OPEN] >= prev[JAW_OPEN]) else RELEASE_ALPHA
        cur = _smooth(prev, cur, alpha)
        if EXTRA_SMOOTH > 1e-3:
            cur = _smooth(cur, prev, EXTRA_SMOOTH)

        # Late gains (no exaggeration)
        v = cur[:]
        v[JAW_OPEN]        = min(1.0, v[JAW_OPEN]        * GAIN_JAW)
        v[MOUTH_FUNNEL]    = min(1.0, v[MOUTH_FUNNEL]    * GAIN_FUNNEL)
        v[MOUTH_PUCKER]    = min(1.0, v[MOUTH_PUCKER]    * GAIN_PUCKER)
        v[MOUTH_STRETCH_L] = min(1.0, v[MOUTH_STRETCH_L] * GAIN_STRETCH)
        v[MOUTH_STRETCH_R] = min(1.0, v[MOUTH_STRETCH_R] * GAIN_STRETCH)
        v[MOUTH_CLOSE]     = min(1.0, v[MOUTH_CLOSE]     * GAIN_CLOSE)
        v[MOUTH_SMILE_L]   = min(1.0, v[MOUTH_SMILE_L]   * GAIN_SMILE)
        v[MOUTH_SMILE_R]   = min(1.0, v[MOUTH_SMILE_R]   * GAIN_SMILE)

        v = [min(1.0, max(0.0, round(x, 3))) for x in v]
        out.append(v); prev = v

    out = _autoscale(out, target_peak=TARGET_PEAK, hard_cap=HARD_CAP)
    _neutralize_edges(out)
    return out
