# constants.py
from __future__ import annotations

from typing import Dict, List, Tuple
import pycountry
from django.conf import settings


# ============================
# BASE URL
# ============================
BASE_URL: str = getattr(settings, "BASE_URL", "http://127.0.0.1:8002").rstrip("/")


# ============================
# TEXT GENERATION MODELS
# ============================
TEXT_MODELS: Dict[str, Dict[str, int | str]] = {
    "gpt-4o":       {"label": "GPT-4o (128k)",       "limit": 128_000},
    "gpt-4o-mini":  {"label": "GPT-4o Mini (64k)",   "limit": 64_000},
    "gpt-4o-turbo": {"label": "GPT-4o Turbo (128k)", "limit": 128_000},
}

TEXT_MODEL_CHOICES: List[Tuple[str, str]] = sorted((k, v["label"]) for k, v in TEXT_MODELS.items())
TEXT_LIMITS: Dict[str, int] = {k: int(v["limit"]) for k, v in TEXT_MODELS.items()}


# ============================
# EMBEDDING MODELS
# ============================
EMBED_MODELS: Dict[str, Dict[str, int | str]] = {
    "text-embedding-3-small": {"label": "Text Embedding 3 Small (8k)",  "limit": 8_000},
    "text-embedding-3-large": {"label": "Text Embedding 3 Large (16k)", "limit": 16_000},
}

EMBED_MODEL_CHOICES: List[Tuple[str, str]] = sorted((k, v["label"]) for k, v in EMBED_MODELS.items())
EMBED_LIMITS: Dict[str, int] = {k: int(v["limit"]) for k, v in EMBED_MODELS.items()}


# ============================
# DEFAULT MODEL SELECTION
# ============================
DEFAULT_TEXT_MODEL: str = min(TEXT_LIMITS, key=TEXT_LIMITS.get)
DEFAULT_TEXT_LIMIT: int = TEXT_LIMITS[DEFAULT_TEXT_MODEL]
DEFAULT_EMBED_MODEL: str = min(EMBED_LIMITS, key=EMBED_LIMITS.get)
DEFAULT_EMBED_LIMIT: int = EMBED_LIMITS[DEFAULT_EMBED_MODEL]


# ============================
# UI / FORM CHOICES
# ============================
GENDER_CHOICES: List[Tuple[str, str]] = [
    ("male", "Male"),
    ("female", "Female"),
]

# Services aligned with Voice.service defaults
SERVICE_CHOICES: List[Tuple[str, str]] = [
    ("gtts", "gTTS"),
    ("elevenlabs", "ElevenLabs"),
    ("openai", "OpenAI")
]


# ============================
# COUNTRY CHOICES
# ============================
def get_country_choices() -> List[Tuple[str, str]]:
    """
    Sorted list of (alpha-2, name), with Palestine override:
      - Exclude default 'IL' and 'PS' from pycountry
      - Append ('PS', 'Palestine') at the end
    """
    countries = [(c.alpha_2, c.name) for c in pycountry.countries if c.alpha_2 not in ("IL", "PS")]
    countries.append(("PS", "Palestine"))
    return sorted(countries, key=lambda x: x[1])


COUNTRY_CHOICES: List[Tuple[str, str]] = get_country_choices()


# ============================
# VOICE CATALOGS
# ============================
# ---- gTTS ----
# gTTS doesn't have named voices; voice_id encodes language + accent via TLD: "{lang}@{tld}"
GTTS_VOICES: Dict[str, Dict[str, str]] = {
    "en@com":    {"label": "English",      "gender": "male"},
    "es@com":    {"label": "Spanish",      "gender": "female"},
    "fr@fr":     {"label": "French",       "gender": "female"},
    "de@de":     {"label": "German",       "gender": "male"},
    "it@it":     {"label": "Italian",      "gender": "female"},
    "pt@com.br": {"label": "Portuguese",   "gender": "female"},
    "hi@co.in":  {"label": "Hindi",        "gender": "male"},
    "ar@com.sa": {"label": "Arabic",       "gender": "male"},
    "ja@co.jp":  {"label": "Japanese",     "gender": "female"},
    "ko@co.kr":  {"label": "Korean",       "gender": "female"},
    "zh-cn@com.hk": {"label": "Chinese",   "gender": "female"},
}

# ---- OpenAI TTS ----
OPENAI_VOICES: Dict[str, Dict[str, str]] = {
    "alloy":   {"label": "Alloy",   "gender": "male"},
    "echo":    {"label": "Echo",    "gender": "male"},
    "fable":   {"label": "Fable",   "gender": "male"},
    "onyx":    {"label": "Onyx",    "gender": "male"},
    "nova":    {"label": "Nova",    "gender": "female"},
    "shimmer": {"label": "Shimmer", "gender": "female"},
    "ash":     {"label": "Ash",     "gender": "male"},
    "ballad":  {"label": "Ballad",  "gender": "female"},
    "coral":   {"label": "Coral",   "gender": "female"},
    "sage":    {"label": "Sage",    "gender": "male"},
    "verse":   {"label": "Verse",   "gender": "male"},
}

# ---- ElevenLabs (popular defaults subset) ----
# Full catalog should be fetched from their API per-account if you need exhaustiveness.
ELEVENLABS_VOICES: Dict[str, Dict[str, str]] = {
    "EXAVITQu4vr4xnSDxMaL": {"label": "Sarah",     "gender": "female"},
    "21m00Tcm4TlvDq8ikWAM": {"label": "Rachel",    "gender": "female"},
    "cjVigY5qzO86Huf0OWal": {"label": "Eric",      "gender": "male"},
    "TxGEqnHWrfWFTfGW9XjX": {"label": "Elli",      "gender": "female"},
    "pMsXgVXv3BLzUgSXRplE": {"label": "Sam",       "gender": "male"},
    "XrExE9yKIg1WjnnlVkGX": {"label": "Matilda",   "gender": "female"},
    "JBFqnCBsd6RMkjVDRZzb": {"label": "George",    "gender": "male"},
    "Xb7hH8MSUJpSbSDYk0k2": {"label": "Alice",     "gender": "female"},
    "IKne3meq5aSn9XLyUdCD": {"label": "Charlie",   "gender": "male"},
    "SAz9YHcvj6GT2YYXdXww": {"label": "River",     "gender": "male"},
    "bIHbv24MWmeRgasZH58o": {"label": "Will",      "gender": "male"},
    "nPczCjzI2devNBz1zQrb": {"label": "Brian",     "gender": "male"},
    "9BWtsMINqrJLrRacOk9x": {"label": "Aria",      "gender": "female"},
    "onwK4e9ZLuTAKqWW03F9": {"label": "Daniel",    "gender": "male"},
    "pqHfZKP75CvOlQylNhV4": {"label": "Bill",      "gender": "male"},
    "TX3LPaxmHKxFdv7VOQHJ": {"label": "Liam",      "gender": "male"},
    "FGY2WhTYpPnrIDTdsKH5": {"label": "Laura",     "gender": "female"},
    "cgSgspJ2msm6clMCkdW9": {"label": "Jessica",   "gender": "female"},
    "N2lVS1w4EtoT3dr4eOWO": {"label": "Callum",    "gender": "male"},
    "XB0fDUnXU5powFXDhCwa": {"label": "Charlotte", "gender": "female"},
}


# ============================
# VOICE CHOICE HELPERS
# ============================
# Grouped (optgroup) choices for admin selects — sorted by label
ALL_VOICE_CHOICES: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("gTTS",       sorted([(f"gtts::{vid}",      meta["label"]) for vid, meta in GTTS_VOICES.items()], key=lambda x: x[1])),
    ("OpenAI", sorted([(f"openai::{vid}",  meta["label"]) for vid, meta in OPENAI_VOICES.items()], key=lambda x: x[1])),
    ("ElevenLabs", sorted([(f"elevenlabs::{vid}", meta["label"]) for vid, meta in ELEVENLABS_VOICES.items()], key=lambda x: x[1])),
]
