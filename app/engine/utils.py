from __future__ import annotations

import logging
import regex
from typing import Any, Dict, List, Optional, Tuple
from textwrap import dedent
from django.utils.html import strip_tags

log = logging.getLogger(__name__)

# --- sentence splitting (same as your latest good version) ---
_COMMON_ABBREVIATIONS = {"mr","mrs","ms","dr","prof","sr","jr","st","vs","etc","e.g","i.e","a.m","p.m","u.s","u.k","usa","uk","no","inc","dept","fig","al","eds","repr"}
_ABBR_PATTERN = "|".join(regex.escape(a) for a in sorted(_COMMON_ABBREVIATIONS, key=len, reverse=True))
_SENTENCE_BOUNDARY_RE = regex.compile(
    rf"""(?:
            (?<=\.\.\.)|(?<=…)|(?<=[.!?])|(?<=[。！？])
         )
         (?:(?<=['"”’\)\]\}}]))?
         (?<!\b(?:{_ABBR_PATTERN})\.)
         (?<!\d)\.?(?!\d)
         (?=\s|$)""",
    flags=regex.VERBOSE | regex.IGNORECASE,
)
def _collapse_ws(s: str) -> str:
    return "" if not s else regex.sub(r"[ \t\f\v\u00A0]+", " ", s)
def extract_sentences(buffer: str) -> Tuple[List[str], str]:
    text = (buffer or "")
    if not text.strip():
        return [], ""
    text = _collapse_ws(text)
    parts = [p.strip() for p in _SENTENCE_BOUNDARY_RE.split(text) if p and p.strip()]
    if len(parts) <= 1:
        return [], text
    sentences, remainder = parts[:-1], parts[-1]
    fixed: List[str] = []
    for s in sentences:
        if s and (s[-1] in ".!?…。！？" or s.endswith('."') or s.endswith('.”') or s.endswith(".'") or s.endswith(".’")):
            fixed.append(s)
        else:
            fixed.append(s + ".")
    return fixed, remainder

# --- prompt helpers ---
def _base_preamble(bot_name: str, bot_id: str, instruction: Optional[str]) -> str:
    safe_name = (bot_name or "Assistant").strip()
    safe_instr = strip_tags(instruction or "").strip()
    core = dedent(f"""
        You are {safe_name} (bot_id={bot_id}).
        Be clear, concise, and factual. Prefer short sentences (≤ 20 words).
        Avoid filler and apologies. Never output Editor.js or any JSON.
    """).strip()
    return f"{core}\n{safe_instr}" if safe_instr else core

def build_text_system_prompt(*, bot_name: str, bot_id: str, instruction: Optional[str] = None) -> str:
    """
    TEXT stream prompt.
    - If the user asks for slides/a deck/presentation/PowerPoint/Keynote (create/make/update):
      * The system MUST call the slide tool to generate/update the deck.
      * In your text, state: “I’ve created/updated the deck …”.
      * Then: “Here is the summary:” followed by 3–7 bullets (≤ 12 words).
      * No JSON, no code fences, no tool names.
    - If the user asks to summarize/explain/suggest improvements for an existing deck, answer concisely in prose/bullets.
    - Otherwise, answer normally in concise prose. Prefer exact names, dates, metrics.
    - Safety: refuse unsafe requests briefly with safer alternatives.
    """
    pre = _base_preamble(bot_name, bot_id, instruction)
    rules = dedent("""
        Rules:
        1) For deck requests: assume the slide tool runs now; affirm creation/update, then give “Here is the summary:” (3–7 bullets, ≤ 12 words).
        2) For deck Q&A/improvements: concise prose/bullets only.
        3) Never include Editor.js or any JSON in text output.
    """).strip()
    return f"{pre}\n\n{rules}"
