from __future__ import annotations

import logging
import regex
from typing import Any, Dict, List, Optional, Tuple
from textwrap import dedent
from django.utils.html import strip_tags
import json

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
# Sentence splitting (unchanged behavior)
# -----------------------------------------------------------------------------
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
# Slides context helpers (compact & unambiguous)
# -----------------------------------------------------------------------------
def _flatten_editorjs_for_prompt(editorjs: Dict[str, Any], *, limit_blocks: int = 60) -> List[str]:
    if not isinstance(editorjs, dict):
        return []
    blocks = editorjs.get("blocks") or []
    lines: List[str] = []
    for b in blocks[:limit_blocks]:
        t = (b.get("type") or "").lower()
        d = b.get("data") or {}
        if t == "header":
            txt = (d.get("text") or "").strip()
            if txt:
                lines.append(f"# {txt}")
        elif t == "paragraph":
            txt = (d.get("text") or "").strip()
            if txt:
                lines.append(txt)
        elif t == "list":
            items = d.get("items") or []
            for it in items[:8]:
                s = str(it).strip()
                if s:
                    lines.append(f"- {s}")
    return lines

def build_slides_context(snap: Dict[str, Any]) -> str:
    """
    Produce a concise, LLM-friendly summary of the deck with an explicit, exhaustive
    change list based ONLY on `snap["diff"]` (no backward compatibility).
    """
    title = snap.get("title") or ""
    summary = snap.get("summary") or ""
    editorjs = snap.get("editorjs") or {}

    version = snap.get("version")
    updated_by = snap.get("updated_by") or ""
    updated_at = snap.get("updated_at") or ""

    lines: List[str] = []

    # CURRENT — the only source of truth
    lines.append("## CURRENT DECK — SOURCE OF TRUTH")
    lines.append(f"Meta: v{version} • by {updated_by} • at {updated_at}")
    lines.append(f"Title: {title or '—'}")
    if summary:
        lines.append(f"Summary: {summary}")
    flats = _flatten_editorjs_for_prompt(editorjs)
    lines.extend(flats if flats else ["[No sections]"])

    # Exhaustive DIFF (CURRENT vs PREVIOUS)
    d = snap.get("diff") or {}
    if d:
        lines.append("\n## DIFF (CURRENT vs PREVIOUS)")
        fc = d.get("fields_changed", [])
        lines.append("Fields changed: " + (", ".join(fc) if fc else "none"))

        # Title / Summary diffs
        tdiff = d.get("title") or {}
        if tdiff.get("changed"):
            lines.append(f"Title: '{tdiff.get('from','')}' -> '{tdiff.get('to','')}'")
        sdiff = d.get("summary") or {}
        if sdiff.get("changed"):
            lines.append("Summary changed.")

        # Editor.js ops
        ej = d.get("editorjs") or {}
        if ej.get("changed"):
            bc = ej.get("block_count", {})
            lines.append(f"Blocks: {bc.get('from',0)} -> {bc.get('to',0)}")

            hdr = ej.get("headers", {})
            if hdr.get("added"):
                lines.append("Headers added: " + ", ".join(hdr["added"][:12]))
            if hdr.get("removed"):
                lines.append("Headers removed: " + ", ".join(hdr["removed"][:12]))
            if hdr.get("renamed"):
                for r in hdr["renamed"][:12]:
                    lines.append(f"Header[{r.get('index')}]: '{r.get('from','')}' -> '{r.get('to','')}'")

            # List ops (first 30 for brevity; caller has full JSON in slides)
            ops = ej.get("ops", [])
            for op in ops[:30]:
                kind = op.get("op")
                if kind == "update":
                    lines.append(f"UPDATE[{op.get('index')}] {op.get('type')}: {json.dumps(op.get('changes'), ensure_ascii=False)}")
                elif kind == "move":
                    lines.append(f"MOVE {op.get('type')}: {op.get('from')} -> {op.get('to')}")
                elif kind == "add":
                    lines.append(f"ADD[{op.get('index')}] {op.get('type')}")
                elif kind == "remove":
                    lines.append(f"REMOVE[{op.get('index')}] {op.get('type')}")

    # PREVIOUS — HISTORY ONLY (optional; does not feed current facts)
    prev = snap.get("previous") or {}
    prev_title = prev.get("title") or ""
    prev_summary = prev.get("summary") or ""
    prev_ej = prev.get("editorjs") or {}
    if prev_title or prev_summary or (prev_ej or {}).get("blocks"):
        lines.append("\n## PREVIOUS DECK — HISTORY ONLY (do NOT use for current facts)")
        if prev_title:
            lines.append(f"Prev Title: {prev_title}")
        if prev_summary:
            lines.append(f"Prev Summary: {prev_summary}")
        lines.extend(_flatten_editorjs_for_prompt(prev_ej))

    out = "\n".join(lines)
    return out[:8000] + ("\n…" if len(out) > 8000 else "")

# -----------------------------------------------------------------------------
# Prompts (strong slide-tool policy)
# -----------------------------------------------------------------------------
def build_router_system_prompt() -> str:
    """
    STRONG policy: any slide/deck/presentation request MUST use slide tools.
    The model must not draft slide content in the chat.
    """
    return dedent("""
        You can call tools. Follow this policy strictly:

        SLIDES-ONLY VIA TOOLS
        - For ANY request that mentions a slide/deck/presentation (create, generate, make, update, edit, revise, add, convert, improve):
            -> Call generate_or_update_slides(prompt=USER_MESSAGE, ai_enrich=true, max_sections=6). Do not ask the user for title/audience/goal; the tool will infer.
        - For ANY request to show/view/see/check/what changed/diff/updates in slides:
            -> Call fetch_latest_slides() and then summarize based ONLY on the returned snapshot.

        NEVER draft slide content directly in your chat response.
        NEVER say you don't have access; if slide data is needed, call fetch_latest_slides().

        If the user is not asking about slides, do not call slide tools.
    """).strip()

def build_text_system_prompt(*, bot_name: str, bot_id: str, instruction: Optional[str] = None) -> str:
    safe_name = (bot_name or "Assistant").strip()
    safe_instr = strip_tags(instruction or "").strip()
    pre = (
        f"You are {safe_name} (bot_id={bot_id}). Be concise and specific.\n"
        "Prefer short bullets over long paragraphs."
    )
    if safe_instr:
        pre = f"{pre}\n{safe_instr}"

    rules = dedent("""
        Slides policy (runtime):
        - If a slides context is present, treat “CURRENT DECK — SOURCE OF TRUTH” as the ONLY factual source.
        - Never summarize “PREVIOUS DECK” unless the user asks for 'what changed', 'diff', 'previous', or 'history'.
        - If the current deck has no sections yet, say so explicitly (e.g., “Draft deck titled ‘X’ exists; no sections yet.”)
          Do NOT reuse previous content to describe the current deck.
        - When describing changes, ONLY reference `slides.diff` (fields, headers, ops). Do not infer or invent.
        - Keep answers short and useful.
    """).strip()

    return f"{pre}\n\n{rules}"
