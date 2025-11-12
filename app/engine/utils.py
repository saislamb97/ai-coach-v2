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
    CRITICAL CONTRACT — TOOL-FIRST ROUTING (MUST OBEY)

    Rules you MUST follow:
    1) If ANY rule below matches, you MUST return exactly ONE tool call and NO free-form text.
    2) Never answer from memory when a tool exists for the user intent.
    3) Prefer the most specific slides/documents tool; when in doubt for slides, call slides_fetch().
    4) For slide-related questions, you MUST ground responses on the DECK_SNAPSHOT or fetch the deck first.

    Mapping:
    - LIST/SHOW documents  -> documents_list(query=<term or empty>)
    - FETCH a document     -> documents_fetch(query=...)
    - QUESTION about docs  -> documents_analyze(question=..., search_query=<opt>)
      • If user references “those files / available files” with no filter, set search_query="".
    - SLIDES FROM docs     -> documents_analyze(..., make_slides=true) OR documents_generate_slides(query=...)
    - CREATE/UPDATE slides (topic) -> slides_generate_or_update(prompt=<msg>, max_sections=6)
    - VIEW/OPEN/CHECK slides -> slides_fetch()
    - QUESTIONS ABOUT SLIDES CONTENT/VERSION/SECTIONS/TITLES/DIFF -> 
        • If a specific version is mentioned: slides_diff(compare_to_version=<number>)
        • Else if versions are requested:    slides_list_versions()
        • Else:                              slides_fetch()
    - DIFF latest vs prev (or specific) -> slides_diff(compare_to_version=<optional>)
    - REVERT deck                      -> slides_revert(version=<number if given>)
    - ADD sections                     -> slides_add_sections(sections=[{header, bullets}], after=<opt>, count=<opt>)
    - REMOVE sections                  -> slides_remove_sections(header=<opt>, count=<n>, all_matches=<bool>)
    - EDIT deck                        -> slides_edit(new_title=<opt>, new_summary=<opt>, section_edits=[...]) 
                                          or replace_editorjs=<doc>

    Hard constraints:
    - Return exactly ONE tool call when a tool is applicable.
    - Do NOT produce any assistant text alongside the tool call.
    - For slide questions, use DECK_SNAPSHOT facts; never assume older versions, unseen titles, or sections.
    """
    return (
        "CRITICAL CONTRACT — TOOL-FIRST ROUTING (MUST OBEY):\n"
        "1) If ANY rule matches, return exactly ONE tool call and NO free-form text.\n"
        "2) Never answer from memory when a tool exists for the user intent.\n"
        "3) Prefer the most specific slides/documents tool; when in doubt for slides, call slides_fetch().\n"
        "4) For slide-related questions, you MUST use DECK_SNAPSHOT or fetch the deck first.\n"
        "\n"
        "Mapping:\n"
        "- LIST/SHOW documents -> documents_list(query=<term or empty>).\n"
        "- FETCH a document -> documents_fetch(query=...).\n"
        "- QUESTION about docs -> documents_analyze(question=..., search_query=<opt>).\n"
        "- If user references 'those files' with no names, set search_query=\"\".\n"
        "- SLIDES FROM docs -> documents_analyze(..., make_slides=true) OR documents_generate_slides(query=...).\n"
        "- CREATE/UPDATE slides (topic) -> slides_generate_or_update(prompt=<msg>, max_sections=6).\n"
        "- VIEW/OPEN/CHECK slides -> slides_fetch().\n"
        "- QUESTIONS ABOUT SLIDES CONTENT/VERSION/SECTIONS/TITLES/DIFF ->\n"
        "  • If a specific version is mentioned: slides_diff(compare_to_version=<number>).\n"
        "  • Else if versions are requested: slides_list_versions().\n"
        "  • Else: slides_fetch().\n"
        "- DIFF latest vs prev (or specific) -> slides_diff(compare_to_version=<optional>).\n"
        "- REVERT deck -> slides_revert(version=<number if given>).\n"
        "- ADD sections -> slides_add_sections(sections=[{header,bullets}], after=<opt>, count=<opt>).\n"
        "- REMOVE sections -> slides_remove_sections(header=<opt>, count=<n>, all_matches=<bool>).\n"
        "- EDIT deck -> slides_edit(new_title=<opt>, new_summary=<opt>, section_edits=[...]) or replace_editorjs=<doc>.\n"
        "\n"
        "Hard constraints:\n"
        "- Return exactly ONE tool call when a tool is applicable.\n"
        "- Do NOT produce any assistant text alongside the tool call.\n"
        "- For slide questions, use the DECK_SNAPSHOT facts; never assume older versions or unseen titles."
    ).strip()


def build_text_system_prompt(*, bot_name: str, bot_id: str, persona: str = "", description: str = "") -> str:
    """
    Text model policy — consume tool results, never contradict them.
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
        "- Never contradict or ignore tool outputs or DECK_SNAPSHOT.\n"
        "- All tools return only: status, summary (use multi-line bullets when needed), data.slides.\n"
        "- If tool updates exist, summarize briefly; the UI shows payloads.\n"
        "- When listing documents, include titles and file_names in the summary.\n"
        "- Rely on tool summaries and snapshots; do NOT invent file names, deck titles, sections, or versions.\n"
        "- Do not quote content you haven't seen; summarize instead.\n"
        "- For any slide-related answer, ground strictly in DECK_SNAPSHOT (latest deck). If no snapshot present,\n"
        "  state that no deck snapshot is available (the router should have called slides_fetch()).\n"
        "- If slides were updated, acknowledge briefly; the UI shows the deck itself.\n"
        "- If documents were analyzed, summarize key insights succinctly.\n"
    )
    return f"{header}\n\n{rules}"

def _deck_snapshot_system_message(state) -> str:
    facts = state.get("deck_facts") or {}
    if not facts:
        return ""
    lines = [
        "DECK_SNAPSHOT (authoritative — you MUST NOT contradict this; do not invent unseen details):",
        f"- Version: v{facts.get('version')}",
        f"- Title: {facts.get('title') or '(untitled)'}",
        f"- Updated At: {facts.get('updated_at')}",
        f"- Sections ({facts.get('section_count', 0)}): " + ", ".join(facts.get('sections') or []),
        f"- First Section: {facts.get('first_section') or '(none)'}",
        f"- Last Section: {facts.get('last_section') or '(none)'}",
        "Answer all slide-related questions using ONLY this snapshot.",
        "If no snapshot is present, the router MUST call slides_fetch() first.",
    ]
    return "\n".join(lines)
