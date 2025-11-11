from __future__ import annotations

import logging
import regex
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
    Router rules (updated for new slides tools):
    - LIST/SHOW documents -> documents_list(query=<term or empty>)
    - FETCH document by title/filename/keywords -> documents_fetch(query=...)
    - QUESTION about content -> documents_analyze(question=..., search_query=<term or empty>)
      If user says “those files / available files” with no filter, set search_query="".
    - SLIDES FROM docs -> documents_analyze(..., make_slides=true) OR documents_generate_slides(query=...)
    - CREATE/UPDATE slides (freeform) -> slides_generate_or_update(prompt=<msg>, max_sections=6)
    - VIEW/OPEN/CHECK slides -> slides_fetch_latest()
    - LIST VERSIONS -> slides_list_versions()
    - DIFF latest vs prev (or specific) -> slides_diff_latest(compare_to_version=<opt>)
    - REVERT slides -> slides_revert(version=<number if given>)
    - ADD sections -> slides_add_sections(sections=[{header,bullets}], after=<opt>, count=<opt>)
      • Enforce 3..9 sections; if already at 9, tool will block and ask permission to remove.
    - REMOVE sections -> slides_remove_sections(header=<opt>, count=<n>, all_matches=<bool>)
    - EDIT deck -> slides_edit(new_title=<opt>, new_summary=<opt>, section_edits=[...]) or replace_editorjs=<doc>
    Return a single tool call per intent. Do NOT tell the user you'll do it later—call the tool.
    """
    return (
        "Router rules:\n"
        "- LIST/SHOW documents -> documents_list(query=<term or empty>).\n"
        "- FETCH a document by title/filename/keywords -> documents_fetch(query=...).\n"
        "- QUESTION about content -> documents_analyze(question=..., search_query=<optional term>).\n"
        "- If user references 'those files' with no names, set search_query=\"\".\n"
        "- CREATE/UPDATE slides from docs -> documents_analyze(..., make_slides=true) or documents_generate_slides(query=...).\n"
        "- CREATE/UPDATE slides on a topic -> slides_generate_or_update(prompt=<msg>, max_sections=6).\n"
        "- VIEW/OPEN/CHECK slides -> slides_fetch_latest().\n"
        "- LIST versions -> slides_list_versions().\n"
        "- DIFF -> slides_diff_latest(compare_to_version=<optional>).\n"
        "- REVERT -> slides_revert(version=<number if given>).\n"
        "- ADD sections -> slides_add_sections(sections=[{header,bullets}], after=<opt>, count=<opt>).\n"
        "- REMOVE sections -> slides_remove_sections(header=<opt>, count=<n>, all_matches=<bool>).\n"
        "- EDIT deck -> slides_edit(new_title=<opt>, new_summary=<opt>, section_edits=[...]) or replace_editorjs=<doc>.\n"
        "- Return exactly ONE tool call when a tool is applicable."
    ).strip()

def build_text_system_prompt(*, bot_name: str, bot_id: str, persona: str = "", description: str = "") -> str:
    """
    Provide current agent info so the LLM behaves accordingly.
    - name -> who is speaking
    - persona -> tone/behaviors
    - description -> capabilities/role specifics
    Keep responses crisp. Prefer short bullets. Summarize tool results.
    """
    safe_name = (bot_name or "Assistant").strip()
    safe_persona = strip_tags(persona or "").strip()
    safe_desc = strip_tags(description or "").strip()

    identity = f"You are {safe_name} (bot_id={bot_id})."
    traits = []
    if safe_persona:
        traits.append(f"Persona: {safe_persona}")
    if safe_desc:
        traits.append(f"Description: {safe_desc}")
    header = identity + (("\n" + "\n".join(traits)) if traits else "")

    rules = (
        "Runtime policy:\n"
        "- Be concise and specific; prefer short bullets.\n"
        "- If tool updates exist, summarize them briefly; the UI shows full payloads.\n"
        "- When listing documents, include titles and file_names in the summary.\n"
        "- Rely on tool summaries; do NOT invent file names or deck versions.\n"
        "- Do not quote content you haven't seen; summarize instead.\n"
        "- If slides were updated, acknowledge briefly; the UI shows the deck itself.\n"
        "- If documents were analyzed, summarize key insights succinctly.\n"
    )
    return f"{header}\n\n{rules}"
