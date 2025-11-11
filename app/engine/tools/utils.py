from __future__ import annotations

import json
import logging
import os
import re
import time
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

from asgiref.sync import sync_to_async
from django.db import models
from langchain_openai import ChatOpenAI

from memory.models import Document, Session, _strip_ctrl

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Config & Context
# --------------------------------------------------------------------------------------
DB_PRIMARY = os.getenv("DB_PRIMARY_ALIAS", "default")
DEFAULT_TEXT_MODEL = os.getenv("RAG_TEXT_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_SECS = 25

MAX_EDITORJS_BLOCKS = 120
MAX_EXTRACTED_CHARS_PER_DOC = int(os.getenv("DOCS_MAX_CHARS", "16000"))
TOKENS_PER_DOC_DEFAULT = int(os.getenv("DOCS_TOKENS_PER_DOC", "8000"))
MAX_SOURCES_PER_ANALYSIS = 5

_TOOL_CTX: ContextVar[Dict[str, Any]] = ContextVar("TOOL_CTX", default={})

def set_tool_context(ctx: Dict[str, Any]) -> None:
    _TOOL_CTX.set(dict(ctx or {}))

def get_tool_context() -> Dict[str, Any]:
    return dict(_TOOL_CTX.get() or {})

def _now_ms() -> int:
    return int(time.time() * 1000)

# --------------------------------------------------------------------------------------
# Tokens + LLM
# --------------------------------------------------------------------------------------
try:
    from engine.tokens import truncate_string_to_token_limit
except Exception:  # fallback
    def truncate_string_to_token_limit(content: str, max_token_limit: int | None = None) -> str:
        if not content:
            return ""
        limit = max_token_limit or 10000
        b = content.encode("utf-8", errors="ignore")
        return b[:limit].decode("utf-8", errors="ignore")

def _llm(model: Optional[str] = None, temperature: float = 0.2, timeout: int = LLM_TIMEOUT_SECS) -> ChatOpenAI:
    return ChatOpenAI(model=(model or DEFAULT_TEXT_MODEL), temperature=temperature, timeout=timeout)

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _slug(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s

def _tokenize_query(q: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]{2,}", (q or "").lower()) if t]

def _file_basename(d: Document) -> str:
    try:
        return os.path.basename(d.file.name or "") or ""
    except Exception:
        return ""

def _name_similarity(q: str, d: Document) -> float:
    import difflib
    qn = re.sub(r"[^a-z0-9]+", "", (q or "").lower())
    cand = re.sub(
        r"[^a-z0-9]+",
        "",
        ((d.title or "") + " " + _file_basename(d) + " " + (d.normalized_name or "")).lower(),
    )
    return difflib.SequenceMatcher(a=qn, b=cand).ratio()

def _search_term_hits(q: str, d: Document) -> int:
    toks = set(_tokenize_query(q))
    try:
        terms = set(d.search_terms or [])
    except Exception:
        terms = set()
    return len(toks & terms)

def _rank_docs(docs: List[Document], query: str) -> List[Document]:
    return sorted(
        docs,
        key=lambda k: (_name_similarity(query, k) + 0.15 * _search_term_hits(query, k)),
        reverse=True,
    )

# -----------------------------
# Minimal Editor.js helpers
# -----------------------------
EditorJS = Dict[str, Any]
ALLOWED_BLOCK_TYPES = {"header", "paragraph", "list"}

def _coerce_blocks(blocks_any: Any) -> List[Dict[str, Any]]:
    if not isinstance(blocks_any, list):
        return []
    out: List[Dict[str, Any]] = []
    for b in blocks_any:
        if not isinstance(b, dict):
            continue
        t = (b.get("type") or "").lower()
        d = b.get("data") if isinstance(b.get("data"), dict) else None
        if t not in ALLOWED_BLOCK_TYPES or d is None:
            continue
        if t == "header":
            try:
                level = int(d.get("level", 2))
            except Exception:
                level = 2
            if level not in (1, 2, 3):
                level = 2
            d = {**d, "level": level, "text": _strip_ctrl(d.get("text", ""))}
        elif t == "paragraph":
            d = {**d, "text": _strip_ctrl(d.get("text", ""))}
        elif t == "list":
            items = d.get("items")
            if not isinstance(items, list):
                items = []
            d = {**d, "style": "unordered", "items": [_strip_ctrl(str(x)) for x in items if str(x).strip()]}
        out.append({"type": t, "data": d})
        if len(out) >= MAX_EDITORJS_BLOCKS:
            break
    return out

def normalize_editorjs(obj: Any) -> Optional[EditorJS]:
    if isinstance(obj, list):
        blocks = _coerce_blocks(obj)
        return {"time": _now_ms(), "blocks": blocks, "version": "2.x"} if blocks else None
    if isinstance(obj, dict) and "blocks" in obj:
        blocks = _coerce_blocks(obj.get("blocks"))
        if not blocks:
            return None
        time_val = obj.get("time")
        ver = obj.get("version") or "2.x"
        try:
            time_val = int(time_val) if time_val is not None else _now_ms()
        except Exception:
            time_val = _now_ms()
        return {"time": time_val, "blocks": blocks, "version": str(ver)}
    return None

def minimal_editorjs(title: str, summary: str = "") -> EditorJS:
    title = _strip_ctrl((title or "Untitled Deck").strip()[:140])
    summary = _strip_ctrl((summary or "").strip()[:300])
    blocks: List[Dict[str, Any]] = [{"type": "header", "data": {"text": title, "level": 2}}]
    if summary:
        blocks.append({"type": "paragraph", "data": {"text": summary}})
    return {"time": _now_ms(), "version": "2.x", "blocks": blocks}

def _ensure_single_top_header(blocks: List[Dict[str, Any]], title: Optional[str]) -> List[Dict[str, Any]]:
    """
    Ensure the first header is a single H2 (deck title) and demote any subsequent H2/H1 headers to H3.
    If no header is present, prepend an H2. If `title` is provided, set it on the first H2.
    """
    out: List[Dict[str, Any]] = []
    placed_top = False
    desired_title = _strip_ctrl(title) if title else None

    for b in blocks:
        t = (b.get("type") or "").lower()
        if t != "header":
            out.append(b)
            continue

        data = dict(b.get("data") or {})
        lvl = data.get("level", 2)
        try:
            lvl = int(lvl)
        except Exception:
            lvl = 2

        if not placed_top:
            # Make the very first header the H2 title
            out.append({"type": "header", "data": {"text": desired_title or _strip_ctrl(data.get("text", "")), "level": 2}})
            placed_top = True
        else:
            # Demote any subsequent H1/H2 to H3 to keep exactly one H2
            if lvl in (1, 2):
                lvl = 3
            out.append({"type": "header", "data": {"text": _strip_ctrl(data.get("text", "")), "level": lvl}})

    if not placed_top:
        out = [{"type": "header", "data": {"text": desired_title or "Untitled Deck", "level": 2}}] + out

    return out

# --------------------------------------------------------------------------------------
# Async-safe DB helpers (SESSION-SCOPED)
# --------------------------------------------------------------------------------------
@sync_to_async(thread_sensitive=True)
def _infer_user_from_session(session_id: int) -> Optional[int]:
    sess = Session.objects.using(DB_PRIMARY).filter(id=session_id).only("user_id").first()
    return getattr(sess, "user_id", None) if sess else None

async def _ensure_agent_user_from_ctx(ctx: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    # Kept for compatibility with other tools; not strictly needed for session-scoped docs.
    agent_id = ctx.get("agent_id")
    user_id = ctx.get("user_id") or ctx.get("owner_user_id") or ctx.get("user")
    if not user_id:
        sess_id = ctx.get("session_id")
        if sess_id:
            try:
                user_id = await _infer_user_from_session(int(sess_id))
            except Exception:
                user_id = None
    return (int(agent_id) if agent_id else None, int(user_id) if user_id else None)

@sync_to_async(thread_sensitive=True)
def _db_list_docs(*, session_id: int, query: str = "", limit: int = 20) -> List[Document]:
    qs = (
        Document.objects.using(DB_PRIMARY)
        .filter(session_id=session_id)
        .order_by("-created_at")
    )
    if query:
        q_text = query.strip()
        slug = _slug(q_text)
        toks = _tokenize_query(q_text)
        q_obj = (
            models.Q(title__icontains=q_text)
            | models.Q(normalized_name__icontains=slug)
            | models.Q(file__icontains=q_text)
            | models.Q(content__icontains=q_text)
        )
        for t in toks:
            q_obj |= models.Q(search_terms__contains=[t])
        qs = qs.filter(q_obj)
    return list(qs[: max(1, min(200, limit))])

# --------------------------------------------------------------------------------------
# Content + Meta helpers (NO extraction; content already on model)
# --------------------------------------------------------------------------------------
def _truncate_text_by_tokens(text: str, tokens: int = TOKENS_PER_DOC_DEFAULT) -> str:
    try:
        return truncate_string_to_token_limit(text, max_token_limit=tokens)
    except Exception:
        b = text.encode("utf-8", errors="ignore")
        return b[:tokens].decode("utf-8", errors="ignore")

def _doc_content_snippet(
    doc: Document,
    *,
    limit_chars: int = MAX_EXTRACTED_CHARS_PER_DOC,
    token_cap: int = TOKENS_PER_DOC_DEFAULT,
) -> str:
    text = _strip_ctrl(doc.content or "")
    if limit_chars and len(text) > limit_chars:
        text = text[:limit_chars]
    return _truncate_text_by_tokens(text, tokens=token_cap)

def _compact_meta_summary(meta: Dict[str, Any] | None) -> str:
    """Short summary for the tool 'summary' field (caller LLM)."""
    if not isinstance(meta, dict) or not meta:
        return "meta: {}"
    mime = meta.get("mime") or meta.get("mimetype")
    ext = meta.get("ext")
    enc = meta.get("encoding")
    size_b = meta.get("size_bytes") or (meta.get("stats") or {}).get("size_bytes")
    try:
        size_kb = f"{int(size_b)//1024:,} KB" if size_b else None
    except Exception:
        size_kb = None
    stats = meta.get("stats") or {}
    n_chars = stats.get("num_chars")
    n_words = stats.get("num_words")
    n_lines = stats.get("num_lines")
    kind = (meta.get("structure") or {}).get("kind") or (meta.get("kind"))
    bits = []
    if ext: bits.append(ext)
    if mime: bits.append(mime)
    if kind: bits.append(f"kind={kind}")
    if enc: bits.append(f"encoding={enc}")
    if size_kb: bits.append(size_kb)
    if n_words: bits.append(f"words={n_words}")
    if n_lines: bits.append(f"lines={n_lines}")
    return "meta: " + ", ".join(bits[:8])

def _pretty_meta_for_llm(meta: Dict[str, Any] | None, max_len: int = 2000) -> str:
    """Compact JSON (for analyzer LLM context)."""
    try:
        blob = json.dumps(meta or {}, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        blob = "{}"
    blob = blob[:max_len]
    return blob
