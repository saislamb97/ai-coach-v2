from __future__ import annotations

import logging
import regex
from typing import Any, Dict, List, Optional, Tuple
from textwrap import dedent
from django.utils.html import strip_tags

log = logging.getLogger(__name__)

# --- sentence splitting (unchanged) ---
_COMMON_ABBREVIATIONS = {
    "mr","mrs","ms","dr","prof","sr","jr","st","vs","etc","e.g","i.e",
    "a.m","p.m","u.s","u.k","usa","uk","no","inc","dept","fig","al","eds","repr"
}
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

# --- prompt helpers (UPDATED) ---
def _base_preamble(bot_name: str, bot_id: str, instruction: Optional[str]) -> str:
    safe_name = (bot_name or "Assistant").strip()
    safe_instr = strip_tags(instruction or "").strip()
    core = dedent(f"""
        You are {safe_name} (bot_id={bot_id}).
        Be clear, concise, and factual. Prefer short sentences (≤ 20 words).
        Avoid filler and apologies. Never output Editor.js or any JSON.
        If another system message provides policy flags (e.g., slides_write_persisted), you MUST obey them.
    """).strip()
    return f"{core}\n{safe_instr}" if safe_instr else core

def build_text_system_prompt(*, bot_name: str, bot_id: str, instruction: Optional[str] = None) -> str:
    """
    TEXT stream prompt.

    Slides-aware behavior:
    - If a separate system message includes **Slides Snapshot** (and optionally **Previous Snapshot**),
      talk about slides using ONLY those lines. Quote Title/Summary verbatim if needed.
    - If the user asks for "latest", "current", "what changed", "diff", or "compare":
        * Produce a concise change log derived ONLY from the two snapshots:
            • Title: <previous> → <current> (include only if different)
            • Summary: "updated" if changed; otherwise "no change"
            • Sections added: list new headers (≤ 6, short phrases)
            • Sections removed: list removed headers (≤ 6)
            • Notable unchanged topics: up to 3 headers that appear in both
        * If no differences are present, say "No changes detected in the deck."
    - Never claim slides were updated unless a system message states slides_write_persisted=true.
      If slides_write_persisted=false (or absent) after an update request, say
      "No changes saved yet." and ask what to change specifically.

    Deck-creation/update behavior:
    - When the user explicitly asks to create/make/update/edit slides, the slide tool may run.
      If slides_write_persisted=true, say: “I’ve created/updated the deck \"<Title>\".”
      Then: “Here is the summary:” with 3–7 bullets (each ≤ 12 words).
      If slides_write_persisted=false, say “No changes saved yet.” and ask for concrete instructions.

    General behavior:
    - For slide Q&A/improvements (not creation), answer concisely in prose/bullets.
    - For non-slide topics, answer normally in concise prose. Prefer exact names, dates, metrics.
    - Safety: briefly refuse unsafe requests and suggest safer alternatives.
    - Do NOT include code fences, tool names, or any JSON in your reply.
    """
    pre = _base_preamble(bot_name, bot_id, instruction)
    rules = dedent("""
        Rules:
        1) Use ONLY provided snapshots for slide content; do not invent sections.
        2) Keep bullets short and scannable; avoid redundancy.
        3) Never output Editor.js or any JSON in text output.
    """).strip()
    return f"{pre}\n\n{rules}"
