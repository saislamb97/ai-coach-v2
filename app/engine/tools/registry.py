from __future__ import annotations

from typing import Any, Dict
from langchain_core.utils.function_calling import convert_to_openai_tool

# expose context helpers to callers
from .utils import set_tool_context, get_tool_context

# Slides tools (current canonical names)
from .slides import (
    slides_fetch,             # fetch latest
    slides_generate_or_update,
    slides_list_versions,
    slides_diff,
    slides_revert,
    slides_add_sections,
    slides_remove_sections,
    slides_edit,
)

# Documents tools
from .documents import (
    documents_list,
    documents_fetch,
    documents_analyze,
    documents_generate_slides,
)

# General tools
from .general import (
    emotion_detect,
    search_wikipedia,
)

AGENT_TOOLS: Dict[str, Any] = {
    # Slides
    "slides_fetch": slides_fetch,
    "slides_generate_or_update": slides_generate_or_update,
    "slides_list_versions": slides_list_versions,
    "slides_diff": slides_diff,
    "slides_revert": slides_revert,
    "slides_add_sections": slides_add_sections,
    "slides_remove_sections": slides_remove_sections,
    "slides_edit": slides_edit,
    # Documents
    "documents_list": documents_list,
    "documents_fetch": documents_fetch,
    "documents_analyze": documents_analyze,
    "documents_generate_slides": documents_generate_slides,
    # General
    "emotion_detect": emotion_detect,
    "search_wikipedia": search_wikipedia,
}

TOOLS_SCHEMA = [convert_to_openai_tool(t) for t in AGENT_TOOLS.values()]

__all__ = [
    "AGENT_TOOLS",
    "TOOLS_SCHEMA",
    "set_tool_context",
    "get_tool_context",
]
