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
# Sentence splitting
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
            if txt: lines.append(f"# {txt}")
        elif t == "paragraph":
            txt = (d.get("text") or "").strip()
            if txt: lines.append(txt)
        elif t == "list":
            items = d.get("items") or []
            for it in items[:8]:
                s = str(it).strip()
                if s: lines.append(f"- {s}")
    return lines

def build_slides_context(snap: Dict[str, Any]) -> str:
    """
    Produce a concise, LLM-friendly summary of the deck:
    - CURRENT only for facts
    - explicit DIFF vs previous revision
    - versions list (with keep_limit)
    """
    title = snap.get("title") or ""
    summary = snap.get("summary") or ""
    editorjs = snap.get("editorjs") or {}
    version = snap.get("version")
    updated_by = snap.get("updated_by") or ""
    updated_at = snap.get("updated_at") or ""
    keep_limit = int(snap.get("keep_limit") or 3)
    revs = snap.get("revisions") or []

    lines: List[str] = []
    lines.append("## CURRENT DECK — SOURCE OF TRUTH")
    lines.append(f"Meta: v{version} • by {updated_by} • at {updated_at}")
    lines.append(f"Title: {title or '—'}")
    if summary:
        lines.append(f"Summary: {summary}")
    flats = _flatten_editorjs_for_prompt(editorjs)
    lines.extend(flats if flats else ["[No sections]"])

    # Versions summary
    if revs:
        vs = ", ".join(f"v{r.get('version')}: { (r.get('title') or '—')[:60] }" for r in revs)
        lines.append(f"\n## VERSIONS (latest ≤ {keep_limit})")
        lines.append(vs)

    # DIFF (current vs previous)
    d = snap.get("diff") or {}
    if d:
        lines.append("\n## DIFF (CURRENT vs PREVIOUS)")
        fc = d.get("fields_changed", [])
        lines.append("Fields changed: " + (", ".join(fc) if fc else "none"))
        tdiff = d.get("title") or {}
        if tdiff.get("changed"):
            lines.append(f"Title: '{tdiff.get('from','')}' -> '{tdiff.get('to','')}'")
        sdiff = d.get("summary") or {}
        if sdiff.get("changed"):
            lines.append("Summary changed.")
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

    out = "\n".join(lines)
    return out[:8000] + ("\n…" if len(out) > 8000 else "")

# -----------------------------------------------------------------------------
# Knowledge context helpers (for grounding final text)
# -----------------------------------------------------------------------------
def build_knowledge_answer_context(kres: Dict[str, Any]) -> str:
    """
    Turn answer_from_knowledge result into a compact grounding block
    the text model can follow. Keep it short; include citations.
    PRIVACY: never surface internal UUIDs/keys to the user.
    """
    if not isinstance(kres, dict):
        return ""
    ans = (kres.get("answer") or "").strip()
    used = kres.get("used_assets") or []

    lines: List[str] = ["## KNOWLEDGE ANSWER — PRIMARY SOURCE"]
    if ans:
        lines.append(ans[:2000])
    if used:
        lines.append("\nSources:")
        for a in used[:8]:
            # Titles only; no UUIDs/keys fallback.
            title = (a.get("title") or a.get("original_name") or "").strip()
            if title:
                lines.append(f"- {title}")
    return "\n".join(lines)[:3200]

def build_knowledge_list_context(assets: List[Dict[str, Any]]) -> str:
    assets = assets or []
    lines: List[str] = ["## KNOWLEDGE LIST"]
    lines.append(f"Count: {len(assets)}")
    if not assets:
        lines.append("(No knowledge files found for this agent/user.)")
        return "\n".join(lines)
    for a in assets[:50]:
        title = (a.get("title") or a.get("original_name") or "").strip()
        mime  = (a.get("mimetype") or "").split(";")[0]
        lines.append(f"- {title} [{mime}]")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Prompts (routing and runtime policies)
# -----------------------------------------------------------------------------
def build_router_system_prompt() -> str:
    return (
        "You MUST decide tool calls. Follow these rules EXACTLY. "
        "Never answer directly when a rule applies—call the tool instead.\n\n"

        "SLIDES (TOOLS ONLY)\n"
        "- If the user asks to create/make/generate/update/edit/rewrite/expand slides/decks/presentations:\n"
        "  -> Call generate_or_update_slides with:\n"
        "     prompt = the user's latest message (verbatim)\n"
        "     context = a SHORT 1–2 sentence topic summary (from recent context)\n"
        "     ai_enrich = true, max_sections = 6\n"
        "- If the user asks to view/show/see/check/diff/what changed in slides:\n"
        "  -> Call fetch_latest_slides().\n"
        "- If the user asks to revert/undo/go back/previous OR names a version (e.g., 'v2'):\n"
        "  -> Call revert_slides(version=<number if specified, else omit>).\n\n"

        "KNOWLEDGE (TOOLS ONLY)\n"
        "- If the user asks to LIST, SHOW, CHECK, or SEE what knowledge/files/documents exist, "
        "  OR asks whether ANY files exist, OR asks to RETRIEVE/GET/ALL knowledge/files, "
        "  OR asks 'is there a file in my knowledge?' (or any paraphrase):\n"
        "  -> Call list_knowledge(query=\"\").\n"
        "- If the user asks ABOUT content of a specific file/report/document/PDF/knowledge (e.g., "
        "  'what does the quarterly report say?'):\n"
        "  -> Call answer_from_knowledge(question=<user message>)  # no slides by default\n"
        "- If the user EXPLICITLY asks for slides/deck/presentation about a file/topic:\n"
        "  -> EITHER call answer_from_knowledge(question=<user message>, make_slides=true)\n"
        "     OR call generate_or_update_slides (choose one).\n"
        "- If they ask what files exist with a search term:\n"
        "  -> Call list_knowledge(query=<term>).\n\n"

        "MANDATORY:\n"
        "- Do NOT answer questions about knowledge/files without calling a knowledge tool first.\n"
        "- Do NOT claim there are no files without calling list_knowledge.\n"
        "- If none of the above rules apply, respond normally (no tool).\n\n"

        "EXAMPLES:\n"
        "- 'is there a file in my knowledge?'                -> list_knowledge(query=\"\")\n"
        "- 'do I have any knowledge files?'                  -> list_knowledge(query=\"\")\n"
        "- 'can you retrieve all the knowledge?'             -> list_knowledge(query=\"\")\n"
        "- 'list knowledge files'                            -> list_knowledge(query=\"\")\n"
        "- 'what does the quarterly report say?'             -> answer_from_knowledge(question=...)\n"
        "- 'make slides from the quarterly report'           -> answer_from_knowledge(question=..., make_slides=true)\n"
        "- 'open slides' / 'what changed in the deck'        -> fetch_latest_slides()\n"
        "- 'revert the slides to v3'                         -> revert_slides(version=3)\n"
    ).strip()

def build_text_system_prompt(*, bot_name: str, bot_id: str, instruction: Optional[str] = None) -> str:
    safe_name = (bot_name or "Assistant").strip()
    safe_instr = strip_tags(instruction or "").strip()
    pre = f"You are {safe_name} (bot_id={bot_id}). Be concise and specific.\nPrefer short bullets over long paragraphs."
    if safe_instr:
        pre = f"{pre}\n{safe_instr}"

    rules = dedent("""
        Runtime policy:
        - If a SLIDES context is present, treat “CURRENT DECK — SOURCE OF TRUTH” as the ONLY factual source for slides.
        - Use the versions list to reference version numbers accurately. Do NOT invent them.
        - Do not reuse previous content as current facts; only use diffs to explain what changed.
        - If the current deck has no sections yet, say so explicitly (e.g., "Draft deck titled ‘X’ exists; no sections yet.")
        - If a KNOWLEDGE ANSWER or KNOWLEDGE LIST block is present, ground your answer on it; do not invent files or content.
        - Keep answers short and useful.
    """).strip()

    return f"{pre}\n\n{rules}"
