# tools.py
from __future__ import annotations

import json
import logging
import os
import re
import time
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple, cast

import difflib
from asgiref.sync import sync_to_async
from django.core.files.storage import default_storage
from django.db import models, transaction
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from memory.models import (
    Slides, SlidesRevision, SLIDES_KEEP_VERSIONS,
    Knowledge,
    Session,
    _strip_ctrl, _clean_json,
)

# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------
log = logging.getLogger(__name__)
DB_PRIMARY = os.getenv("DB_PRIMARY_ALIAS", "default")

# Shared tool context (session_id, bot_id, agent_id, user_id, etc.) set upstream
_TOOL_CTX: ContextVar[Dict[str, Any]] = ContextVar("TOOL_CTX", default={})

def set_tool_context(ctx: Dict[str, Any]) -> None:
    _TOOL_CTX.set(dict(ctx or {}))

def get_tool_context() -> Dict[str, Any]:
    return dict(_TOOL_CTX.get() or {})

# --------------------------------------------------------------------------------------
# Slug helper
# --------------------------------------------------------------------------------------
def _slug(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s

def _tokenize_query(q: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]{2,}", (q or "").lower()) if t]

# --------------------------------------------------------------------------------------
# Editor.js helpers (minimal, safe)
# --------------------------------------------------------------------------------------
EditorJS = Dict[str, Any]
ALLOWED_BLOCK_TYPES = {"header", "paragraph", "list"}
MAX_BLOCKS = 120

def _now_ms() -> int:
    return int(time.time() * 1000)

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

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
            level = int(d.get("level", 2))
            if level not in (1, 2, 3):
                level = 2
            d = {**d, "level": level, "text": _strip_ctrl(d.get("text", ""))}

        if t == "paragraph":
            d = {**d, "text": _strip_ctrl(d.get("text", ""))}

        if t == "list":
            items = d.get("items")
            if not isinstance(items, list):
                items = []
            d = {**d, "items": [_strip_ctrl(str(x)) for x in items if isinstance(x, (str, int, float))]}

        out.append({"type": t, "data": d})
        if len(out) >= MAX_BLOCKS:
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
    if not title:
        return blocks
    title = _strip_ctrl(title)
    out: List[Dict[str, Any]] = []
    put_title = False
    for b in blocks:
        if (b.get("type") or "").lower() == "header" and not put_title:
            out.append({"type": "header", "data": {"text": title, "level": 2}})
            put_title = True
        else:
            out.append(b)
    if not put_title:
        out = [{"type": "header", "data": {"text": title, "level": 2}}] + out
    return out

# --------------------------------------------------------------------------------------
# FULL DIFF (fields + Editor.js)
# --------------------------------------------------------------------------------------
def _canon_str(x: Any) -> str:
    return " ".join(_strip_ctrl(str(x or "")).split())

def _canon_items(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    return [_canon_str(v) for v in items if _canon_str(v)]

def _canon_block(b: Dict[str, Any]) -> Dict[str, Any]:
    t = (b.get("type") or "").lower()
    d = b.get("data") or {}
    if t == "header":
        return {"type": "header", "level": int(d.get("level", 2)), "text": _canon_str(d.get("text"))}
    if t == "paragraph":
        return {"type": "paragraph", "text": _canon_str(d.get("text"))}
    if t == "list":
        return {"type": "list", "style": (d.get("style") or "unordered"), "items": _canon_items(d.get("items"))}
    return {"type": t, "data": d}

def _block_signature(cb: Dict[str, Any]) -> str:
    t = cb.get("type")
    if t == "header":
        return f"header|{cb.get('level',2)}|{cb.get('text','')}"
    if t == "paragraph":
        return f"paragraph|{cb.get('text','')}"
    if t == "list":
        joined = "•".join(cb.get("items", []))
        return f"list|{cb.get('style','unordered')}|{joined}"
    return f"{t}|{json.dumps(cb, sort_keys=True)}"

def _diff_text(prev: str, cur: str) -> Dict[str, Any]:
    return {"changed": prev != cur, "from": prev, "to": cur}

def _diff_list(prev_items: List[str], cur_items: List[str]) -> Dict[str, Any]:
    prev_set, cur_set = set(prev_items), set(cur_items)
    added = [x for x in cur_items if x not in prev_set]
    removed = [x for x in prev_items if x not in cur_set]
    kept = [x for x in cur_items if x in prev_set]
    reordered = (kept != [x for x in prev_items if x in cur_set])
    return {"changed": bool(added or removed or reordered), "added": added, "removed": removed, "kept": kept, "reordered": reordered}

def _compare_blocks(pb: Dict[str, Any], cb: Dict[str, Any]) -> Dict[str, Any]:
    pt, ct = pb.get("type"), cb.get("type")
    if pt != ct:
        return {"type_changed": True, "from": pb, "to": cb}
    t = pt
    changes: Dict[str, Any] = {"type": t, "changed": False}
    if t == "header":
        level_diff = int(pb.get("level", 2)) != int(cb.get("level", 2))
        text_diff = _diff_text(pb.get("text",""), cb.get("text",""))
        changes.update({"level_changed": level_diff, "text": text_diff})
        changes["changed"] = level_diff or text_diff["changed"]
    elif t == "paragraph":
        text_diff = _diff_text(pb.get("text",""), cb.get("text",""))
        changes.update({"text": text_diff})
        changes["changed"] = text_diff["changed"]
    elif t == "list":
        style_diff = (pb.get("style","unordered") != cb.get("style","unordered"))
        items_diff = _diff_list(cast(List[str], pb.get("items", [])), cast(List[str], cb.get("items", [])))
        changes.update({"style_changed": style_diff, "items": items_diff})
        changes["changed"] = style_diff or items_diff["changed"]
    else:
        raw_changed = (pb != cb)
        changes.update({"raw_changed": raw_changed, "from": pb, "to": cb})
        changes["changed"] = raw_changed
    return changes

def _diff_editorjs(prev_ej: Dict[str, Any], cur_ej: Dict[str, Any]) -> Dict[str, Any]:
    prev_blocks = (prev_ej or {}).get("blocks") or []
    cur_blocks = (cur_ej or {}).get("blocks") or []
    P = [_canon_block(b) for b in prev_blocks]
    C = [_canon_block(b) for b in cur_blocks]
    Psig = [_block_signature(b) for b in P]
    Csig = [_block_signature(b) for b in C]
    sm = difflib.SequenceMatcher(a=Psig, b=Csig, autojunk=False)
    opcodes = sm.get_opcodes()

    ops: List[Dict[str, Any]] = []
    pending_deletes: List[Tuple[int, str, Dict[str, Any]]] = []
    pending_inserts: List[Tuple[int, str, Dict[str, Any]]] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i1, i2):
                ops.append({"op": "keep", "index": j1 + (k - i1), "type": C[j1 + (k - i1)]["type"]})
        elif tag == "delete":
            for k in range(i1, i2):
                pending_deletes.append((k, Psig[k], P[k]))
        elif tag == "insert":
            for k in range(j1, j2):
                pending_inserts.append((k, Csig[k], C[k]))
        elif tag == "replace":
            plen, clen = i2 - i1, j2 - j1
            n = min(plen, clen)
            for t in range(n):
                pb, cb = P[i1 + t], C[j1 + t]
                change = _compare_blocks(pb, cb)
                if change.get("changed"):
                    ops.append({"op": "update", "index": j1 + t, "type": cb.get("type"), "changes": change})
                else:
                    ops.append({"op": "keep", "index": j1 + t, "type": cb.get("type")})
            for k in range(i1 + n, i2):
                pending_deletes.append((k, Psig[k], P[k]))
            for k in range(j1 + n, j2):
                pending_inserts.append((k, Csig[k], C[k]))

    used_ins: set[int] = set()
    used_del: set[int] = set()
    for di, ds, db in pending_deletes:
        match_j = next((ji for (ji, js, _) in pending_inserts if js == ds and ji not in used_ins), None)
        if match_j is not None:
            used_del.add(di); used_ins.add(match_j)
            ops.append({"op": "move", "from": di, "to": match_j, "type": db.get("type"), "block": db})
    for di, ds, db in pending_deletes:
        if di not in used_del:
            ops.append({"op": "remove", "index": di, "type": db.get("type"), "block": db})
    for ji, js, jb in pending_inserts:
        if ji not in used_ins:
            ops.append({"op": "add", "index": ji, "type": jb, "block": jb})

    prev_headers = [b.get("text","") for b in P if b.get("type") == "header"]
    cur_headers  = [b.get("text","") for b in C if b.get("type") == "header"]
    headers = {
        "added":   [h for h in cur_headers if h not in prev_headers],
        "removed": [h for h in prev_headers if h not in cur_headers],
        "unchanged": [h for h in cur_headers if h in prev_headers],
        "renamed": [
            {"index": i, "from": prev_headers[i], "to": cur_headers[i]}
            for i in range(min(len(prev_headers), len(cur_headers)))
            if prev_headers[i] != cur_headers[i]
        ]
    }
    changed = any(op["op"] in ("add","remove","update","move") for op in ops)
    return {"changed": changed, "block_count": {"from": len(P), "to": len(C)}, "ops": ops, "headers": headers}

def compute_full_diff(*, cur_title: str, cur_summary: str, cur_ej: Dict[str, Any],
                      prev_title: str, prev_summary: str, prev_ej: Dict[str, Any]) -> Dict[str, Any]:
    title = _diff_text(_canon_str(prev_title), _canon_str(cur_title))
    summary = _diff_text(_canon_str(prev_summary), _canon_str(cur_summary))
    ej = _diff_editorjs(prev_ej or {}, cur_ej or {})
    fields_changed = []
    if title["changed"]: fields_changed.append("title")
    if summary["changed"]: fields_changed.append("summary")
    if ej["changed"]: fields_changed.append("editorjs")
    return {"fields_changed": fields_changed, "title": title, "summary": summary, "editorjs": ej}

# --------------------------------------------------------------------------------------
# LLM helpers
# --------------------------------------------------------------------------------------
def _extract_json_maybe(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    if not raw:
        return None
    fence = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
    m = fence.match(raw)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
    except Exception:
        pass
    return None

async def _ai_outline_from_prompt(*, prompt: str, context: str = "", max_sections: int = 6, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    model_name = (model or "gpt-4o-mini").strip()
    llm = ChatOpenAI(model=model_name, temperature=0.3, timeout=20)
    sys = (
        "Return STRICT JSON for a slide outline.\n"
        'Schema: {"title":"string","summary":"1-2 sentences","sections":[{"header":"string","bullets":["point",...]}]}\n'
        f"Keep sections <= {max_sections}. Avoid fluff/duplicates. Keep it relevant to the given context."
    )
    user = (("Context:\n" + context.strip() + "\n\n") if context.strip() else "") + ("Request:\n" + (prompt or "")[:4000])
    if not (prompt or context):
        return None
    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user)])
        data = _extract_json_maybe(getattr(msg, "content", "") or "")
        if not isinstance(data, dict):
            return None
        title = (data.get("title") or "").strip()[:140] or "Untitled Deck"
        summary = (data.get("summary") or "").strip()[:300]
        sections = data.get("sections") or []
        blocks: List[Dict[str, Any]] = [{"type": "header", "data": {"text": _strip_ctrl(title), "level": 2}}]
        if summary: blocks.append({"type": "paragraph", "data": {"text": _strip_ctrl(summary)}})
        for s in sections:
            hdr = _strip_ctrl((s.get("header") or "").strip())
            if not hdr: continue
            blocks.append({"type": "header", "data": {"text": hdr, "level": 3}})
            bullets = [_strip_ctrl(str(x).strip()) for x in (s.get("bullets") or []) if str(x).strip()]
            if bullets: blocks.append({"type": "list", "data": {"style": "unordered", "items": bullets[:8]}})
        ej = normalize_editorjs({"time": _now_ms(), "version": "2.x", "blocks": blocks})
        if not ej: return None
        return {"title": title, "summary": summary, "editorjs": ej}
    except Exception as e:
        log.warning("[slides:outline] synth failed: %s", e)
        return None

async def _llm_answer_with_sources(*, question: str, sources: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Ask the LLM to answer using ONLY the provided sources.
    `sources` = list of {id, title, snippet}.
    """
    llm = ChatOpenAI(model=model, temperature=0.2, timeout=30)
    sys = (
        "You are a precise analyst. Answer ONLY from the provided sources. "
        "If unsure, say you don't have enough information. "
        "Return STRICT JSON: {\"answer\": \"...\", \"citations\": [{\"id\":\"source-id\",\"title\":\"...\"}], \"bullets\": [\"key point\", ...]}"
    )
    src_lines: List[str] = []
    for i, s in enumerate(sources, 1):
        src_lines.append(f"[{i}] id={s['id']} title={_strip_ctrl(s['title'])}\n{_strip_ctrl(s['snippet'])}\n")
    user = "SOURCES:\n\n" + "\n".join(src_lines) + f"\nQUESTION:\n{_strip_ctrl(question.strip())}"
    msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user[:12000])])
    raw = (getattr(msg, "content", "") or "").strip()
    data = _extract_json_maybe(raw) or {}
    ans = _strip_ctrl((data.get("answer") or "").strip())
    cits = data.get("citations") or []
    bullets = [_strip_ctrl(b) for b in (data.get("bullets") or []) if isinstance(b, str) and b.strip()]
    return {"answer": ans, "citations": cits, "bullets": bullets, "raw": raw}

# --------------------------------------------------------------------------------------
# Knowledge helpers
# --------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _name_similarity(q: str, a: Knowledge) -> float:
    qn = _norm(q)
    cand = _norm((a.title or "") + " " + (a.original_name or ""))
    return difflib.SequenceMatcher(a=qn, b=cand).ratio()

@sync_to_async
def _infer_user_from_session(session_id: int) -> Optional[int]:
    sess = (
        Session.objects.using(DB_PRIMARY)
        .filter(id=session_id)
        .only("user_id")
        .first()
    )
    return getattr(sess, "user_id", None) if sess else None

@sync_to_async
def _db_list_assets(*, agent_id: int, user_id: int, query: str = "", limit: int = 20) -> List[Knowledge]:
    """
    High-recall name search that leverages Knowledge.search_terms (array),
    normalized_name, title, original_name, mimetype, and excerpt.
    """
    qs = (Knowledge.objects.using(DB_PRIMARY)
          .filter(agent_id=agent_id, user_id=user_id)
          .order_by("-created_at"))

    if query:
        q_text = query.strip()
        slug = _slug(q_text)
        toks = _tokenize_query(q_text)

        q_obj = (
            models.Q(original_name__icontains=q_text) |
            models.Q(title__icontains=q_text) |
            models.Q(mimetype__icontains=q_text) |
            models.Q(normalized_name__icontains=slug) |
            models.Q(excerpt__icontains=q_text)
        )

        # match any token via JSONField containment (Postgres @> on arrays)
        for t in toks:
            q_obj |= models.Q(search_terms__contains=[t])

        qs = qs.filter(q_obj)

    return list(qs[:max(1, min(200, limit))])

def _extract_asset_text(asset: Knowledge) -> str:
    """
    Robust text extraction for a Knowledge file.
    """
    from tempfile import NamedTemporaryFile
    try:
        from memory.extract import extract_text, detect_mime
    except Exception:
        from memory.extract import extract_text, detect_mime  # noqa

    storage_path = asset.file.name
    blob: bytes = b""
    try:
        with default_storage.open(storage_path, "rb") as f:
            blob = f.read()
    except Exception:
        return ""

    ext = os.path.splitext(asset.original_name or storage_path)[1].lower() or ".bin"
    tmp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(blob)
            tmp_path = tmp.name

        try:
            mime = detect_mime(tmp_path)
        except Exception:
            import mimetypes
            mime = mimetypes.guess_type(asset.original_name or storage_path)[0] or "application/octet-stream"

        text = extract_text(tmp_path, mime=mime)
        return _strip_ctrl(text or "")
    except Exception:
        try:
            return _strip_ctrl(blob.decode("utf-8", errors="replace"))
        except Exception:
            return _strip_ctrl(blob.decode("latin-1", errors="replace"))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

def _read_asset_chunks(asset: Knowledge, *, max_chars_per_chunk: int = 1600) -> List[Dict[str, Any]]:
    """
    Chunk the asset text; returns [{id, title, snippet, key, original_name}, ...].
    """
    text = _extract_asset_text(asset)
    name_line = f"[{(asset.title or asset.original_name or '').strip()}] "
    text = (name_line + re.sub(r"\s+", " ", text)).strip()
    if not text:
        return []

    # naive sentence-ish splitter
    parts: List[str] = []
    acc = ""
    for token in re.split(r"(?<=[\.\!\?\n])\s+", text):
        if len(acc) + len(token) + 1 > max_chars_per_chunk and acc:
            parts.append(acc.strip())
            acc = token
        else:
            acc = acc + " " + token if acc else token
    if acc.strip():
        parts.append(acc.strip())

    chunks: List[Dict[str, Any]] = []
    for i, p in enumerate(parts[:25]):  # cap chunks per asset
        chunks.append({
            "id": f"{asset.key}:{i}",
            "title": _strip_ctrl(asset.title or asset.original_name),
            "snippet": _strip_ctrl(p[:max_chars_per_chunk]),
            "key": str(asset.key),
            "original_name": asset.original_name,
        })
    return chunks

def _rank_chunks(question: str, chunks: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
    q = (question or "").lower()
    tokens = [t for t in re.findall(r"[a-z0-9]{3,}", q)]

    def score(c: Dict[str, Any]) -> int:
        s = (c.get("snippet") or "").lower()
        t = (c.get("title") or "").lower()
        content_hits = sum(s.count(tok) for tok in tokens)
        title_hits = 2 * sum(t.count(tok) for tok in tokens)
        lead_bonus = 1 if any(tok in s or tok in t for tok in tokens[:3]) else 0
        return content_hits + title_hits + lead_bonus

    scored = sorted(chunks, key=score, reverse=True)
    return scored[:max(1, min(12, top_k))]

# --------------------------------------------------------------------------------------
# DB helpers (ALWAYS primary; revisions-aware; re-read after write)
# --------------------------------------------------------------------------------------
@sync_to_async
def _db_fetch_latest_slides_snapshot(session_id: int) -> Optional[Dict[str, Any]]:
    s = (
        Slides.objects.using(DB_PRIMARY)
        .filter(session_id=session_id)
        .order_by("-updated_at", "-id")
        .first()
    )
    if not s:
        return None
    s.refresh_from_db(using=DB_PRIMARY)

    revs_qs = (
        SlidesRevision.objects.using(DB_PRIMARY)
        .filter(session_id=session_id)
        .order_by("-version", "-created_at")[:SLIDES_KEEP_VERSIONS]
    )
    revs = list(revs_qs)
    prev = revs[1] if len(revs) > 1 else None

    diff = compute_full_diff(
        cur_title=s.title or "",
        cur_summary=s.summary or "",
        cur_ej=s.editorjs or {},
        prev_title=(prev.title if prev else "") or "",
        prev_summary=(prev.summary if prev else "") or "",
        prev_ej=(prev.editorjs if prev else {}) or {},
    )

    return {
        "title": s.title,
        "summary": s.summary,
        "editorjs": s.editorjs,
        "version": s.version,
        "thread_id": s.session.thread_id,
        "updated_by": s.updated_by,
        "updated_at": s.updated_at.isoformat(),
        "keep_limit": SLIDES_KEEP_VERSIONS,
        "revisions": [
            {
                "version": r.version,
                "title": r.title,
                "summary": r.summary,
                "updated_by": r.updated_by,
                "created_at": r.created_at.isoformat(),
            } for r in revs
        ],
        "previous": (
            {
                "version": prev.version,
                "title": prev.title,
                "summary": prev.summary,
                "editorjs": prev.editorjs,
                "updated_by": prev.updated_by,
                "created_at": prev.created_at.isoformat(),
            } if prev else None
        ),
        "diff": diff,
    }

@sync_to_async
def _db_rotate_and_save_slides(session_id: int, payload: Dict[str, Any], updated_by: str) -> Dict[str, Any]:
    with transaction.atomic(using=DB_PRIMARY):
        s = (
            Slides.objects.using(DB_PRIMARY)
            .select_for_update()
            .filter(session_id=session_id)
            .first()
        )
        if not s:
            s = Slides.objects.using(DB_PRIMARY).create(session_id=session_id)
        title = (payload.get("title") or "Untitled Deck").strip()
        summary = (payload.get("summary") or "").strip()
        editorjs = payload.get("editorjs") or minimal_editorjs(title, summary)
        s.rotate_and_update(title=title, summary=summary, editorjs=editorjs, updated_by=updated_by)
    snap = _db_fetch_latest_slides_snapshot.__wrapped__(session_id)  # type: ignore[attr-defined]
    return cast(Dict[str, Any], snap)

@sync_to_async
def _db_revert_slides(session_id: int, target_version: Optional[int], updated_by: str) -> Dict[str, Any]:
    with transaction.atomic(using=DB_PRIMARY):
        s = (
            Slides.objects.using(DB_PRIMARY)
            .select_for_update()
            .filter(session_id=session_id)
            .first()
        )
        if not s:
            raise ValueError("No Slides row for session")
        if target_version is None:
            prev = (
                SlidesRevision.objects.using(DB_PRIMARY)
                .filter(session_id=session_id, version__lt=s.version)
                .order_by("-version")
                .first()
            )
            if not prev:
                raise ValueError("No previous version to revert to")
            target_version = prev.version
        s.revert_to_version(target_version=target_version, updated_by=updated_by or "tool:slides_revert")
    snap = _db_fetch_latest_slides_snapshot.__wrapped__(session_id)  # type: ignore[attr-defined]
    return cast(Dict[str, Any], snap)

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class WikipediaInput(BaseModel):
    query: str = Field(..., description="Topic to summarize in 1–2 sentences")

class SlideInput(BaseModel):
    prompt: Optional[str] = Field(None, description="Natural-language request for the deck")
    title: Optional[str] = Field(None, description="Deck title (optional)")
    summary: Optional[str] = Field(None, description="Short abstract (optional)")
    editorjs: Optional[EditorJS | List[Dict[str, Any]]] = Field(None, description="Editor.js doc or blocks[]")
    context: Optional[str] = Field(None, description="Short conversation/topic context to keep slides on-topic")
    ai_enrich: bool = Field(True, description="If true, expand sections with the LLM")
    max_sections: int = Field(6, ge=1, le=16, description="Max sections when enriching")

    @field_validator("editorjs")
    @classmethod
    def _validate_editorjs(cls, v):
        if v is None:
            return v
        ej = normalize_editorjs(v)
        if not ej:
            raise ValueError("editorjs must contain non-empty blocks")
        return ej

class RevertInput(BaseModel):
    version: Optional[int] = Field(None, description="Version to revert to. If omitted, revert to the previous version.")

class EmotionInput(BaseModel):
    text: str = Field(..., description="Text to classify into a single dominant emotion.")

class FetchSlidesInput(BaseModel):
    pass

# Knowledge schemas
class KnowledgeListInput(BaseModel):
    query: Optional[str] = Field(None, description="Filter by name/title/mime (fuzzy rank). No UUIDs are returned.")
    limit: int = Field(20, ge=1, le=200, description="Max number of assets to list")

class KnowledgeAnswerInput(BaseModel):
    question: str = Field(..., description="Your question about the report/knowledge")
    keys: Optional[List[str]] = Field(None, description="(Optional, internal) Specific Knowledge.key UUIDs")
    search_query: Optional[str] = Field(None, description="(Optional) Name/title filter; defaults to the question text")
    top_k_chunks: int = Field(8, ge=1, le=12, description="How many chunks to send to the LLM")
    make_slides: bool = Field(False, description="Generate/update a slide deck summarizing key insights")
    slides_max_sections: int = Field(6, ge=1, le=16, description="Max sections when creating the deck")

# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------
def _meta(status: str, *, session_id: Optional[int]) -> Dict[str, Any]:
    return {"status": status, "fetched_at": _now_iso(), "db": DB_PRIMARY, "session_id": session_id, "keep_limit": SLIDES_KEEP_VERSIONS}

@tool("search_wikipedia", args_schema=WikipediaInput, description="Brief 1–2 sentence summary from Wikipedia.")
async def search_wikipedia(query: str) -> Dict[str, Any]:
    import asyncio, wikipedia
    wikipedia.set_lang("en")

    async def _search_and_summary(q: str) -> Tuple[List[str], Optional[str]]:
        def _blocking() -> Tuple[List[str], Optional[str]]:
            hits = wikipedia.search(q, results=5)
            if not hits:
                return [], None
            title = next((h for h in hits if h.lower() == q.lower() or h.lower().startswith(q.lower())), hits[0])
            try:
                summary = wikipedia.summary(title, sentences=2)
            except Exception:
                summary = None
            return hits, summary
        return await asyncio.to_thread(_blocking)

    q = (query or "").strip()
    if not q:
        return {"status": "failed", "error": "empty_query"}

    try:
        hits, summary = await _search_and_summary(q)
        if not (hits and summary):
            return {"status": "failed", "error": "not_found", "meta": {"query": q}}
        return {"status": "ok", "data": {"title": hits[0], "summary": summary}}
    except Exception as e:
        log.exception("[tool:wikipedia] error")
        return {"status": "failed", "error": str(e)}

@tool("fetch_latest_slides", args_schema=FetchSlidesInput, description="Fetch the freshest slides snapshot (current + latest revisions) from the primary DB.")
async def fetch_latest_slides() -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "error": "missing_session_context"}
    try:
        snap = await _db_fetch_latest_slides_snapshot(int(session_id))
        if not snap:
            return {"status": "not_found", "meta": _meta("not_found", session_id=session_id)}
        return {"status": "ok", "slides": snap, "meta": _meta("ok", session_id=session_id)}
    except Exception as e:
        log.exception("[tool:fetch_latest_slides] db error")
        return {"status": "failed", "error": str(e), "meta": _meta("failed", session_id=session_id)}

@tool(
    "generate_or_update_slides",
    args_schema=SlideInput,
    description="Create/update slides. Infers missing fields from 'prompt' + 'context', persists, and returns a fresh snapshot.",
)
async def generate_or_update_slides(
    prompt: Optional[str] = None,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    editorjs: Optional[EditorJS] = None,
    context: Optional[str] = None,
    ai_enrich: bool = True,
    max_sections: int = 6,
) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "error": "missing_session_context"}

    prompt_clean  = _strip_ctrl((prompt or "").strip())
    title_clean   = _strip_ctrl((title or "").strip())
    summary_clean = _strip_ctrl((summary or "").strip())
    context_clean = _strip_ctrl((context or "").strip())
    ej = normalize_editorjs(editorjs) if editorjs is not None else None

    if not (title_clean or summary_clean or (ej and ej.get("blocks"))):
        outline = await _ai_outline_from_prompt(prompt=prompt_clean, context=context_clean, max_sections=max_sections) if (prompt_clean or context_clean) else None
        if outline:
            title_clean   = outline.get("title") or title_clean
            summary_clean = outline.get("summary") or summary_clean
            ej = outline.get("editorjs") or ej

    if not (ej and ej.get("blocks")):
        title_clean = title_clean or (prompt_clean[:80] if prompt_clean else "Untitled Deck")
        ej = minimal_editorjs(title_clean, summary_clean)

    if ai_enrich and (prompt_clean or context_clean):
        outline2 = await _ai_outline_from_prompt(prompt=prompt_clean, context=context_clean, max_sections=max_sections)
        if outline2 and (outline2.get("editorjs") or {}).get("blocks"):
            blocks = outline2["editorjs"]["blocks"]
            blocks = _ensure_single_top_header(blocks, title_clean or outline2.get("title"))
            ej = normalize_editorjs({"time": _now_ms(), "version": "2.x", "blocks": blocks}) or ej
            title_clean = title_clean or outline2.get("title") or title_clean
            summary_clean = summary_clean or outline2.get("summary") or summary_clean

    if ej and ej.get("blocks"):
        ej = normalize_editorjs({"time": _now_ms(), "version": "2.x", "blocks": _ensure_single_top_header(ej["blocks"], title_clean)}) or ej

    try:
        snap = await _db_rotate_and_save_slides(int(session_id), {
            "title": title_clean or "Untitled Deck",
            "summary": summary_clean or "",
            "editorjs": ej or minimal_editorjs(title_clean or "Untitled Deck", summary_clean or ""),
        }, updated_by="tool:slides")
        return {"status": "ok", "slides": snap, "meta": _meta("ok", session_id=session_id)}
    except Exception as e:
        log.exception("[tool:generate_or_update_slides] persist error")
        return {"status": "failed", "error": str(e), "meta": _meta("failed", session_id=session_id)}

@tool(
    "revert_slides",
    args_schema=RevertInput,
    description="Revert slides to a specific version (or previous if not provided). Returns the fresh snapshot.",
)
async def revert_slides(version: Optional[int] = None) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "error": "missing_session_context"}
    try:
        snap = await _db_revert_slides(int(session_id), version, updated_by="tool:slides_revert")
        return {"status": "ok", "slides": snap, "meta": _meta("ok", session_id=session_id)}
    except Exception as e:
        log.exception("[tool:revert_slides] db error")
        return {"status": "failed", "error": str(e), "meta": _meta("failed", session_id=session_id)}

@tool(
    "emotion_analyze",
    args_schema=EmotionInput,
    description="Classify a single dominant emotion (Joy/Anger/Sadness/Surprise) with intensity 1..3.",
)
async def emotion_analyze(text: str) -> Dict[str, Any]:
    model_name = "gpt-4o-mini"
    llm = ChatOpenAI(model=model_name, temperature=0.0, timeout=8)
    sys = (
        "Return a SINGLE dominant emotion.\n"
        "Allowed names: Joy, Anger, Sadness, Surprise. Intensity: integer 1..3.\n"
        'Return STRICT JSON ONLY like: {"name":"Joy","intensity":2}'
    )
    user = _strip_ctrl((text or "").strip()[:8000])
    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user)])
        raw = (getattr(msg, "content", "") or "").strip()
        data = json.loads(raw)
        name = str(data.get("name", "Joy")).title()
        inten = int(data.get("intensity", 1))
        if name not in ("Joy", "Anger", "Sadness", "Surprise"):
            name = "Joy"
        if inten < 1 or inten > 3:
            inten = 1
        return {"name": name, "intensity": inten}
    except Exception:
        return {"name": "Joy", "intensity": 1}

# -----------------------------
# Knowledge tools (RAG over Knowledge)
# -----------------------------
async def _ensure_agent_user_from_ctx(ctx: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
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

@tool(
    "list_knowledge",
    args_schema=KnowledgeListInput,
    description=(
        "List knowledge files for this agent/user. "
        "Use this for ANY request about whether files/knowledge exist, what files are available, "
        "to list/show/see/check knowledge/files, or to retrieve/get ALL knowledge. "
        "Return titles only (no UUIDs)."
    )
)
async def list_knowledge(query: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    ctx = get_tool_context()
    agent_id, user_id = await _ensure_agent_user_from_ctx(ctx)
    if not agent_id or not user_id:
        return {"status": "failed", "error": "missing_agent_or_user_context"}

    items = await _db_list_assets(agent_id=agent_id, user_id=user_id, query=(query or "").strip(), limit=limit)

    if query:
        items = sorted(items, key=lambda a: _name_similarity(query, a), reverse=True)

    # Public, user-facing fields only — NO keys.
    data = [{
        "title": (a.title or a.original_name),
        "mimetype": a.mimetype,
        "size_bytes": a.size_bytes,
        "created_at": a.created_at.isoformat(),
        "index_status": a.index_status,
    } for a in items[:limit]]

    return {"status": "ok", "assets": data, "count": len(data)}

@tool(
    "answer_from_knowledge",
    args_schema=KnowledgeAnswerInput,
    description=(
        "Answer a question using uploaded knowledge files (PDF/TXT/CSV/Word/Excel). "
        "Defaults to searching by the question text. "
        "Returns {answer, citations, used_assets, slides?}. No UUIDs are exposed."
    ),
)
async def answer_from_knowledge(
    question: str,
    keys: Optional[List[str]] = None,           # kept for internal calls
    search_query: Optional[str] = None,
    top_k_chunks: int = 8,
    make_slides: bool = False,
    slides_max_sections: int = 6,
) -> Dict[str, Any]:
    ctx = get_tool_context()
    agent_id, user_id = await _ensure_agent_user_from_ctx(ctx)
    session_id = ctx.get("session_id")
    if not agent_id or not user_id:
        return {"status": "failed", "error": "missing_agent_or_user_context"}

    # Use explicit keys if provided; otherwise use the search query, defaulting to the full question text.
    derived_query = (search_query or question or "").strip()

    # Enumerate candidate assets with name/title/slug/search_terms/excerpt filtering
    all_assets = await _db_list_assets(
        agent_id=agent_id,
        user_id=user_id,
        query=derived_query,
        limit=200,
    )

    # Restrict by keys if explicitly provided (internal use only)
    if keys:
        keyset = {k.lower() for k in keys}
        all_assets = [a for a in all_assets if str(a.key).lower() in keyset]
        if not all_assets:
            return {"status": "not_found", "error": "no_assets_with_keys"}

    # Prefer the closest filename/title match when using a query string
    if derived_query and not keys:
        all_assets = sorted(all_assets, key=lambda a: _name_similarity(derived_query, a), reverse=True)[:12]

    if not all_assets:
        return {"status": "not_found", "error": "no_assets", "meta": {"query": derived_query}}

    # Build chunks
    chunks: List[Dict[str, Any]] = []
    for a in all_assets[:12]:  # defensive cap
        try:
            chunks.extend(_read_asset_chunks(a))
        except Exception:
            log.exception("[tool:answer_from_knowledge] chunking failed for %s", a.original_name)

    if not chunks:
        return {"status": "failed", "error": "no_chunks_extracted"}

    # Rank + select
    top_chunks = _rank_chunks(question, chunks, top_k=top_k_chunks)

    # LLM answer constrained to sources
    answer_pack = await _llm_answer_with_sources(
        question=_strip_ctrl(question),
        sources=top_chunks,
        model=os.getenv("RAG_TEXT_MODEL", "gpt-4o-mini"),
    )
    answer_text = (answer_pack.get("answer") or "").strip()
    bullets = answer_pack.get("bullets") or []
    citations_llm = answer_pack.get("citations") or []

    # Map citations to titles only (no UUIDs)
    cited_titles: List[str] = []
    if citations_llm:
        id_to_title = {}
        for c in top_chunks:
            # id is "<uuid>:<chunkIndex>"
            id_to_title[c["id"]] = c["title"]
        for c in citations_llm:
            cid = (c or {}).get("id")
            if cid and cid in id_to_title:
                cited_titles.append(id_to_title[cid])

    if not cited_titles:
        # fall back to the titles of top chunk assets
        seen: set[str] = set()
        for c in top_chunks:
            t = c["title"]
            if t and t not in seen:
                seen.add(t)
                cited_titles.append(t)

    # Public response (no keys)
    out: Dict[str, Any] = {
        "status": "ok",
        "answer": answer_text or "I don't have enough information in the provided sources.",
        "citations": [{"title": t} for t in cited_titles[:8]],
        "used_assets": [{"title": (a.title or a.original_name), "mimetype": a.mimetype} 
                        for a in all_assets[:min(5, len(all_assets))]],
        "meta": _meta("ok", session_id=session_id),
    }

    # Optionally create/update slides summarizing key insights
    if make_slides and session_id:
        src_titles = ", ".join([d["title"] for d in out["used_assets"]][:5])
        slide_prompt = f"Create concise slides summarizing key insights for the question: {question}"
        context_lines = []
        if bullets:
            context_lines.append("Key points:\n- " + "\n- ".join(bullets[:10]))
        else:
            sample = " ".join(c["snippet"] for c in top_chunks[:3])
            context_lines.append("Context:\n" + _strip_ctrl(sample[:2000]))
        context_lines.append(f"Sources: {src_titles}")
        context = "\n\n".join(context_lines)

        try:
            slides_res = await generate_or_update_slides.ainvoke({  # type: ignore[attr-defined]
                "prompt": slide_prompt,
                "context": context,
                "ai_enrich": True,
                "max_sections": slides_max_sections,
            })
            if (slides_res or {}).get("status") == "ok":
                out["slides"] = (slides_res or {}).get("slides")
            else:
                out["slides_error"] = (slides_res or {}).get("error") or "unknown"
        except Exception:
            log.exception("[tool:answer_from_knowledge] slide generation failed")

    return out

# --------------------------------------------------------------------------------------
# Registry (exportable to the router)
# --------------------------------------------------------------------------------------
AGENT_TOOLS: Dict[str, Any] = {
    "search_wikipedia": search_wikipedia,
    "fetch_latest_slides": fetch_latest_slides,
    "generate_or_update_slides": generate_or_update_slides,
    "revert_slides": revert_slides,
    "emotion_analyze": emotion_analyze,

    # knowledge
    "list_knowledge": list_knowledge,
    "answer_from_knowledge": answer_from_knowledge,
}

TOOLS_SCHEMA = [convert_to_openai_tool(t) for t in AGENT_TOOLS.values()]
