# app/memory/tools/documents.py
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from memory.models import _strip_ctrl

from .utils import (
    MAX_EXTRACTED_CHARS_PER_DOC,
    TOKENS_PER_DOC_DEFAULT,
    MAX_SOURCES_PER_ANALYSIS,
    get_tool_context,
    _db_list_docs,
    _rank_docs,
    _doc_content_snippet,
    _compact_meta_summary,
    _pretty_meta_for_llm,
    _llm,
)

log = logging.getLogger(__name__)

# ======================================================================================
# Input Schemas (documents)
# ======================================================================================
class DocumentsListInput(BaseModel):
    query: Optional[str] = Field(None, description="Filter by title/file name (fuzzy).")
    limit: int = Field(20, ge=1, le=200, description="Max number of documents to list (per session)")

class DocumentsFetchInput(BaseModel):
    query: str = Field(..., description="Title/file-name/keywords to fetch the closest document(s)")
    limit: int = Field(1, ge=1, le=5, description="How many docs to return")
    include_content: bool = Field(True, description="Include truncated content (already extracted)")
    sample_chars_per_doc: int = Field(MAX_EXTRACTED_CHARS_PER_DOC, ge=1000, le=100000, description="Character cap per doc")
    tokens_per_doc: int = Field(TOKENS_PER_DOC_DEFAULT, ge=2000, le=14000, description="Token cap per doc")

class DocumentsAnalyzeInput(BaseModel):
    question: str = Field(..., description="Your question about the documents")
    search_query: Optional[str] = Field(None, description="Name/title filter; leave empty to analyze recent files in session")
    limit_files: int = Field(3, ge=1, le=MAX_SOURCES_PER_ANALYSIS, description="How many files to analyze")
    sample_chars_per_doc: int = Field(MAX_EXTRACTED_CHARS_PER_DOC, ge=2000, le=100000, description="Content char cap per doc")
    tokens_per_doc: int = Field(TOKENS_PER_DOC_DEFAULT, ge=2000, le=14000, description="Token cap per doc")
    make_slides: bool = Field(False, description="If true, also generate slides grounded on these sources")
    slides_max_sections: int = Field(6, ge=3, le=16, description="Max sections when creating slides")

class DocumentsSlidesInput(BaseModel):
    query: str = Field(..., description="Title/file-name/keywords to find doc(s) to turn into slides")
    limit_files: int = Field(3, ge=1, le=MAX_SOURCES_PER_ANALYSIS, description="Max docs to pull into slides")
    sample_chars_per_doc: int = Field(MAX_EXTRACTED_CHARS_PER_DOC, ge=2000, le=100000, description="Content char cap per doc")
    tokens_per_doc: int = Field(TOKENS_PER_DOC_DEFAULT, ge=2000, le=14000, description="Token cap per doc")
    slides_max_sections: int = Field(6, ge=3, le=16, description="Max sections in the deck")
    title_override: Optional[str] = Field(None, description="Optional deck title to force")

# ======================================================================================
# LLM helper (documents analysis with meta)
# ======================================================================================
async def _llm_analyze_sources(
    *, question: str, sources: List[Dict[str, Any]], model: Optional[str] = None
) -> Dict[str, Any]:
    """
    sources item: {title, file_name, updated_at, meta, content}
    We inject BOTH content and compact meta JSON for each source into the LLM prompt.
    """
    llm = _llm(model=model, temperature=0.2)
    sys = (
        "You are a precise analyst. Use ONLY the provided sources.\n"
        "Return STRICT JSON: "
        '{"answer":"...","bullets":["..."],"citations":[{"title":"...","meta":{}}]}'
    )

    src_lines: List[str] = []
    for i, s in enumerate(sources, 1):
        title = _strip_ctrl(s.get("title") or s.get("file_name") or f"Source {i}")
        # Keep meta compact to keep prompt small
        meta_json = _pretty_meta_for_llm(s.get("meta"), max_len=1200)
        snippet = _strip_ctrl((s.get("content") or "")[:3500])
        src_lines.append(f"[{i}] {title}\nMETA: {meta_json}\nCONTENT:\n{snippet}\n")

    user = "SOURCES:\n\n" + "\n".join(src_lines) + f"\nQUESTION:\n{_strip_ctrl(question.strip())}\n"

    try:
        msg = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=user[:12000])])
        raw = (getattr(msg, "content", "") or "").strip()
        import json as _json
        data = _json.loads(raw if raw.startswith("{") else raw[raw.find("{"):raw.rfind("}") + 1])

        # sanitize
        data["answer"] = _strip_ctrl((data.get("answer") or "").strip())
        data["bullets"] = [
            _strip_ctrl(b) for b in (data.get("bullets") or []) if isinstance(b, str) and b.strip()
        ][:8]
        cits = data.get("citations") or []
        new_cits = []
        for c in cits:
            title = _strip_ctrl((c or {}).get("title") or "")
            meta = c.get("meta") if isinstance(c.get("meta"), dict) else {}
            new_cits.append({"title": title, "meta": meta})
        data["citations"] = new_cits
        return data
    except Exception:
        return {"answer": "I don't have enough information in the provided sources.", "bullets": [], "citations": []}

# ======================================================================================
# Internal helpers
# ======================================================================================
def _mk_assets_payload(docs: List[Any], include_content: bool = False,
                       sample_chars: int = MAX_EXTRACTED_CHARS_PER_DOC,
                       token_cap: int = TOKENS_PER_DOC_DEFAULT) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for d in docs:
        payload: Dict[str, Any] = {
            "title": (d.title or os.path.basename(d.file.name)),
            "file_name": os.path.basename(d.file.name),
            "updated_at": d.updated_at.isoformat(),
            "index_status": d.index_status,
            "sha256": d.sha256,
            "meta": d.meta or {},
        }
        if include_content:
            payload["content"] = _doc_content_snippet(d, limit_chars=sample_chars, token_cap=token_cap)
        assets.append(payload)
    return assets

def _bulleted(lines: List[str]) -> str:
    # Always multiline bullets; trim empties and cap to 8 lines
    clean = [f"- {_strip_ctrl(x).strip()}" for x in lines if _strip_ctrl(x).strip()]
    return "\n".join(clean[:8]) or "- (no details)"

def _pack_slides_for_data(slides_obj) -> Dict[str, Any]:
    """
    Build slides payload in the same shape used by slides tools: {version,title,summary,editorjs,updated_at}.
    Import lazily from slides to avoid circulars.
    """
    if not slides_obj:
        return {}
    from .slides import _ensure_editorjs  # lazy import
    ej = _ensure_editorjs(getattr(slides_obj, "editorjs", {}) or {})
    return {
        "version": slides_obj.version,
        "title": slides_obj.title or "",
        "summary": slides_obj.summary or "",
        "editorjs": ej,
        "updated_at": slides_obj.updated_at.isoformat(),
    }

# ======================================================================================
# Tools (SESSION-SCOPED)
# ======================================================================================
@tool(
    "documents_list",
    args_schema=DocumentsListInput,
    description="List session documents. Returns title, file_name, updated_at, and meta (session-scoped).",
)
async def documents_list(query: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "summary": "- Missing session context.", "data": {}}

    items = await _db_list_docs(session_id=int(session_id), query=(query or "").strip(), limit=limit)
    ranked = _rank_docs(items, query) if query else items

    assets = _mk_assets_payload(ranked[:limit], include_content=False)

    if not assets:
        return {"status": "not_found", "summary": "- No documents in this session.", "data": {"assets": []}}

    # Caller-LLM-friendly summary (bulleted with compact meta bits)
    parts = [f"{a['title']} ({_compact_meta_summary(a.get('meta'))})" for a in assets[:5]]
    summary_out = _bulleted([
        f"Found {len(assets)} document(s).",
        *parts
    ])
    return {"status": "ok", "summary": summary_out, "data": {"assets": assets}}

@tool(
    "documents_fetch",
    args_schema=DocumentsFetchInput,
    description="Fetch closest session document(s) by fuzzy title/file/keywords; optionally include content.",
)
async def documents_fetch(
    query: str,
    limit: int = 1,
    include_content: bool = True,
    sample_chars_per_doc: int = MAX_EXTRACTED_CHARS_PER_DOC,
    tokens_per_doc: int = TOKENS_PER_DOC_DEFAULT,
) -> Dict[str, Any]:
    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "summary": "- Missing session context.", "data": {}}

    items = await _db_list_docs(session_id=int(session_id), query=(query or "").strip(), limit=200)
    if not items:
        return {"status": "not_found", "summary": f"- No documents found for “{_strip_ctrl(query)}”.", "data": {}}

    ranked = _rank_docs(items, query)[:limit]
    assets = _mk_assets_payload(
        ranked, include_content=include_content,
        sample_chars=sample_chars_per_doc, token_cap=tokens_per_doc
    )

    parts = [f"{a['title']} ({_compact_meta_summary(a.get('meta'))})" for a in assets]
    summary_out = _bulleted([
        f"Fetched {len(assets)} document(s) for “{_strip_ctrl(query)}”.",
        *parts
    ])
    return {"status": "ok", "summary": summary_out, "data": {"assets": assets}}

@tool(
    "documents_analyze",
    args_schema=DocumentsAnalyzeInput,
    description=("Answer a question using session documents. Returns analysis + sources. "
                 "Set make_slides=true to also generate slides grounded on these sources."),
)
async def documents_analyze(
    question: str,
    search_query: Optional[str] = None,
    limit_files: int = 3,
    sample_chars_per_doc: int = MAX_EXTRACTED_CHARS_PER_DOC,
    tokens_per_doc: int = TOKENS_PER_DOC_DEFAULT,
    make_slides: bool = False,
    slides_max_sections: int = 6,
) -> Dict[str, Any]:
    # Lazy imports to avoid circular at module load
    from .slides import (
        _llm_outline_to_editorjs,
        _slides_upsert_sync,
        _ensure_editorjs,
    )

    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "summary": "- Missing session context.", "data": {}}

    derived_query = (search_query or "").strip()
    all_docs = await _db_list_docs(session_id=int(session_id), query=derived_query, limit=200)
    if not all_docs:
        return {"status": "not_found", "summary": "- No matching documents found in this session.", "data": {}}

    ranked = _rank_docs(all_docs, derived_query)[:limit_files] if derived_query else all_docs[:limit_files]

    sources: List[Dict[str, Any]] = []
    for d in ranked:
        content = _doc_content_snippet(d, limit_chars=sample_chars_per_doc, token_cap=tokens_per_doc)
        sources.append(
            {
                "title": (d.title or os.path.basename(d.file.name)),
                "file_name": os.path.basename(d.file.name),
                "updated_at": d.updated_at.isoformat(),
                "meta": d.meta or {},
                "content": content,
            }
        )

    analysis = await _llm_analyze_sources(question=_strip_ctrl(question), sources=sources)

    # Summary (bulleted)
    titles = ", ".join([s["title"] for s in sources])
    meta_bits = "; ".join([_compact_meta_summary(s.get("meta")) for s in sources])
    bullets = analysis.get("bullets") or []
    summary_lines = [
        f"Analyzed {len(sources)} document(s): {titles}.",
        meta_bits if meta_bits else "",
        *bullets[:5],
    ]
    summary_out = _bulleted(summary_lines)

    out_data: Dict[str, Any] = {"analysis": analysis, "sources": sources}

    if make_slides:
        # Ground slides on the same sources (content + compact meta)
        context_text = "\n\n".join(
            f"(META){_pretty_meta_for_llm(s['meta'], 600)}\n{_strip_ctrl((s.get('content') or '')[:2000])}"
            for s in sources
        )[:8000]
        prompt = f"Create concise slides summarizing key insights to answer: {question}"
        ej = await _llm_outline_to_editorjs(prompt=prompt, context=context_text, max_sections=slides_max_sections)
        ej = _ensure_editorjs(ej)
        try:
            deck_title = (sources[0]["title"] if sources else "Document Summary")[:140]
            s, _before = await _slides_upsert_sync(
                int(session_id), title=deck_title, summary="", editorjs=ej, updated_by="tool:documents_analyze"
            )
            out_data["slides"] = _pack_slides_for_data(s)
            summary_out = _bulleted([
                *summary_lines,
                f"Created/updated slides v{s.version}."
            ])
        except Exception as e:
            log.exception("[documents_analyze] slide generation failed")
            summary_out = _bulleted([
                *summary_lines,
                f"Failed to create slides: {e}"
            ])

    return {"status": "ok", "summary": summary_out, "data": out_data}

@tool(
    "documents_generate_slides",
    args_schema=DocumentsSlidesInput,
    description="Generate/update slides from session document(s) matching a fuzzy query. Returns deck + sources.",
)
async def documents_generate_slides(
    query: str,
    limit_files: int = 3,
    sample_chars_per_doc: int = MAX_EXTRACTED_CHARS_PER_DOC,
    tokens_per_doc: int = TOKENS_PER_DOC_DEFAULT,
    slides_max_sections: int = 6,
    title_override: Optional[str] = None,
) -> Dict[str, Any]:
    # Lazy imports to avoid circular at module load
    from .slides import (
        _llm_outline_to_editorjs,
        _slides_upsert_sync,
        _ensure_editorjs,
    )

    ctx = get_tool_context()
    session_id = ctx.get("session_id")
    if not session_id:
        return {"status": "failed", "summary": "- Missing session context.", "data": {}}

    items = await _db_list_docs(session_id=int(session_id), query=(query or "").strip(), limit=200)
    if not items:
        return {"status": "not_found", "summary": f"- No documents found for “{_strip_ctrl(query)}”.", "data": {}}

    ranked = _rank_docs(items, query)[:limit_files]

    sources: List[Dict[str, Any]] = []
    for d in ranked:
        content = _doc_content_snippet(d, limit_chars=sample_chars_per_doc, token_cap=tokens_per_doc)
        sources.append(
            {
                "title": (d.title or os.path.basename(d.file.name)),
                "file_name": os.path.basename(d.file.name),
                "updated_at": d.updated_at.isoformat(),
                "meta": d.meta or {},
                "content": content,
            }
        )

    # Generation context (content + compact meta)
    context_text = "\n\n".join(
        f"(META){_pretty_meta_for_llm(s['meta'], 600)}\n{_strip_ctrl((s.get('content') or '')[:2000])}"
        for s in sources
    )[:8000]
    prompt = f"Create concise slides summarizing the key ideas from the selected document(s) about: {query}"
    ej = await _llm_outline_to_editorjs(prompt=prompt, context=context_text, max_sections=slides_max_sections)
    ej = _ensure_editorjs(ej)

    try:
        deck_title = (title_override or (sources[0]["title"] if sources else "Slides"))[:140]
        s, _before = await _slides_upsert_sync(
            int(session_id), title=deck_title, summary="", editorjs=ej, updated_by="tool:documents_generate_slides"
        )
        slides_payload = _pack_slides_for_data(s)
        names = ", ".join([src["title"] for src in sources])
        meta_bits = "; ".join([_compact_meta_summary(src.get("meta")) for src in sources])
        summary_out = _bulleted([
            f"Built slides v{s.version} from {len(sources)} document(s): {names}.",
            meta_bits if meta_bits else "",
        ])
        return {"status": "ok", "summary": summary_out, "data": {"slides": slides_payload, "sources": sources}}
    except Exception as e:
        log.exception("[documents_generate_slides] persist error")
        return {"status": "failed", "summary": f"- Failed to generate slides: {e}", "data": {}}
