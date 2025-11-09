from __future__ import annotations

import logging
import regex
import re
from typing import Any, Dict, List, Tuple
from django.utils.html import strip_tags

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def elapsed_ms(t0: float) -> int:
    import time
    return int((time.perf_counter() - t0) * 1000)

def _collapse_ws(s: str) -> str:
    return "" if not s else regex.sub(r"[ \t\f\v\u00A0]+", " ", s)

# -----------------------------------------------------------------------------
# Sentence splitting (kept compact + robust)
# -----------------------------------------------------------------------------
_COMMON_ABBR = {
    "mr","mrs","ms","dr","prof","sr","jr","st","vs","etc","e.g","i.e",
    "a.m","p.m","u.s","u.k","usa","uk","no","inc","dept","fig","al","eds","repr"
}
_ABBR_PATTERN = "|".join(regex.escape(a) for a in sorted(_COMMON_ABBR, key=len, reverse=True))
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

# -----------------------------------------------------------------------------
# System prompts for router & text model
# -----------------------------------------------------------------------------
def build_router_system_prompt() -> str:
    """
    Router rules (no backwards compat; use documents_*):
    - LIST/SHOW documents -> documents_list(query=<term or empty>)
    - FETCH a doc by title/filename/keywords -> documents_fetch(query=...)
    - QUESTION about content -> documents_analyze(question=..., search_query=<term or empty>)
      IMPORTANT: If user says “those files / available files” with no filter, set search_query="".
    - SLIDES FROM docs -> documents_analyze(..., make_slides=true) OR documents_generate_slides(query=...)
    - CREATE/UPDATE slides (freeform) -> slides_generate_or_update(prompt=<msg>, max_sections=6)
    - VIEW/OPEN/CHECK slides -> slides_fetch_latest()
    - LIST VERSIONS -> slides_list_versions()
    - DIFF latest vs prev (or specific) -> slides_diff_latest(compare_to_version=<opt>)
    - REVERT slides -> slides_revert(version=<number if given>)
    Return a single tool call per intent.
    """
    return (
        "Router rules:\n"
        "- If user asks to LIST or SHOW documents -> call documents_list(query=<term or empty>).\n"
        "- If user asks to FETCH a document by title/filename/keywords -> call documents_fetch(query=...).\n"
        "- If user asks a QUESTION about content -> call documents_analyze(question=..., search_query=<optional term>).\n"
        "- If the user references 'those files' or 'available files' without naming any, set search_query=\"\".\n"
        "- If user asks to CREATE/UPDATE/GENERATE slides from document's contents -> prefer documents_analyze(..., make_slides=true) or documents_generate_slides(query=...).\n"
        "- If user asks whether slides exist, wants to VIEW, OPEN, or CHECK slides -> call slides_fetch_latest().\n"
        "- If user asks to CREATE/UPDATE/GENERATE slides on a topic -> call slides_generate_or_update(prompt=<msg>, max_sections=6).\n"
        "- If user asks to LIST versions -> call slides_list_versions().\n"
        "- If user asks what changed -> call slides_diff_latest(compare_to_version=<optional>).\n"
        "- If user asks to REVERT slides -> call slides_revert(version=<number if given>).\n"
        "- Otherwise: answer normally (no tools)."
    ).strip()

def build_text_system_prompt(*, bot_name: str, bot_id: str, instruction: str | None = None) -> str:
    """
    Keep responses crisp. Always show basic info for documents (titles + file_names).
    Never say you cannot disclose document details — the user owns these files.
    """
    safe_name = (bot_name or "Assistant").strip()
    safe_instr = strip_tags(instruction or "").strip()
    pre = f"You are {safe_name} (bot_id={bot_id}). Be concise and specific. Prefer short bullets."
    if safe_instr:
        pre = f"{pre}\n{safe_instr}"

    rules = (
        "Runtime policy:\n"
        "- When listing documents, include titles and file_names in the summary.\n"
        "- Rely on tool summaries; do NOT invent file names or deck versions.\n"
        "- Do not quote content you haven't seen; summarize instead.\n"
        "- If slides were updated, acknowledge briefly; the UI shows the deck itself.\n"
        "- If documents were analyzed, summarize key insights; the UI holds payloads.\n"
        "- If no tool context applies, answer normally."
    )
    return f"{pre}\n\n{rules}"
