# engine/tokens.py
from __future__ import annotations

import os, logging
from typing import List, Union
from dataclasses import dataclass
from core.selectors import get_llm_config

logger = logging.getLogger(__name__)

config = get_llm_config()
OPENAI_MODEL = (config.text_model if config else os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

@dataclass(frozen=True)
class TokenLimits:
    token_limit: int   # hard ceiling for model (input + output tokens)
    context_limit: int # soft target for prompt/system/context
    history_limit: int # soft cap for chat history portion

DEFAULT_LIMITS = TokenLimits(token_limit=18000, context_limit=10000, history_limit=5000)

def get_token_limits() -> TokenLimits:
    """
    Pull from dynamic model config if present, else fallback.
    """
    try:
        config = get_llm_config()
        return TokenLimits(
            token_limit = getattr(config, "token_limit",    DEFAULT_LIMITS.token_limit) or DEFAULT_LIMITS.token_limit,
            context_limit= getattr(config, "context_limit", DEFAULT_LIMITS.context_limit) or DEFAULT_LIMITS.context_limit,
            history_limit= getattr(config, "history_limit", DEFAULT_LIMITS.history_limit) or DEFAULT_LIMITS.history_limit,
        )
    except Exception as e:
        logger.warning(f"[TokenLimits] Fallback to defaults: {e}")
        return DEFAULT_LIMITS

class _FallbackEncoding:
    """
    Minimal stand-in for tiktoken's Encoding with encode()/decode().
    - encode(text) -> List[int] of UTF-8 bytes
    - decode(List[int]) -> str from UTF-8 bytes
    Only the *length* of encode() and truncating via decode() are used by this module,
    so this approximation is sufficient for safe behavior.
    """
    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            text = str(text)
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, tokens: List[int] | bytes | bytearray) -> str:
        if isinstance(tokens, (bytes, bytearray)):
            b = bytes(tokens)
        else:
            b = bytes(int(t) & 0xFF for t in tokens)
        return b.decode("utf-8", errors="ignore")

# Try to initialize a tiktoken encoding; fall back gracefully on any error.
try:
    import tiktoken  # type: ignore

    try:
        ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception as e1:
        try:
            # Second attempt: pick by model (works for many OpenAI models)
            ENCODING = tiktoken.encoding_for_model(OPENAI_MODEL)
        except Exception as e2:
            logger.warning(
                "[TokenLimits] tiktoken available but encoding init failed; "
                "falling back to UTF-8 byte tokenizer. e1=%s e2=%s",
                e1, e2
            )
            ENCODING = _FallbackEncoding()
except Exception as e0:
    logger.warning(
        "[TokenLimits] tiktoken not available; using UTF-8 byte tokenizer fallback. e=%s", e0
    )
    ENCODING = _FallbackEncoding()

# ---------------------------------------------------
# Token Counting
# ---------------------------------------------------
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

def count_tokens(message: Union[dict, BaseMessage], tokens_per_message: int = 3, tokens_per_name: int = 1) -> int:
    """
    Approximate token count for a single message.
    We keep the simple heuristic (+3 per message) which works well enough
    across OpenAI chat models.
    """
    tokens = tokens_per_message
    if isinstance(message, dict):
        for key, value in message.items():
            try:
                tokens += len(ENCODING.encode(str(value)))
            except Exception:
                # Extremely defensive: if the encoder trips, approximate by byte length.
                tokens += len(str(value).encode("utf-8", errors="ignore"))
            if key == "name":
                tokens += tokens_per_name
    elif isinstance(message, BaseMessage):
        try:
            tokens += len(ENCODING.encode(getattr(message, "content", "") or ""))
        except Exception:
            tokens += len((getattr(message, "content", "") or "").encode("utf-8", errors="ignore"))
        if getattr(message, "name", None):
            tokens += tokens_per_name
    else:
        raise ValueError("Unsupported message type for token counting")
    return tokens

# ---------------------------------------------------
# Message Trimming
# ---------------------------------------------------
def trim_messages_to_token_limit(messages: List[BaseMessage], reserved_tokens: int = 3000) -> List[BaseMessage]:
    """
    Trim long system/history while preserving structure:
    - Merge all SystemMessages into one (dedup + normalize)
    - Respect history soft/hard caps
    - Keep any other messages (e.g., tool planner scaffolding) intact
    """
    limits = get_token_limits()
    allowed_total = max(1024, limits.token_limit - reserved_tokens)

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    history_msgs = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    other_msgs = [m for m in messages if m not in system_msgs + history_msgs]

    # Merge system messages, trim to context_limit
    merged_content = "\n\n".join(m.content.strip() for m in system_msgs if getattr(m, "content", None)) \
                     or "You are a helpful assistant."
    try:
        sys_tokens = ENCODING.encode(merged_content)
    except Exception:
        sys_tokens = list(merged_content.encode("utf-8", errors="ignore"))

    if len(sys_tokens) > limits.context_limit:
        logger.warning("[Tokens] System message too long; trimming to context_limit.")
        try:
            merged_content = ENCODING.decode(sys_tokens[:limits.context_limit])
        except Exception:
            merged_content = bytes(sys_tokens[:limits.context_limit]).decode("utf-8", errors="ignore")
    preserved = [SystemMessage(content=merged_content)]

    # First pass: enforce history_limit for history-only part
    def hist_token_sum(msgs: List[BaseMessage]) -> int:
        return sum(count_tokens(m) for m in msgs)

    while hist_token_sum(history_msgs) > limits.history_limit and len(history_msgs) > 1:
        history_msgs.pop(0)

    # Second pass: ensure total <= allowed_total by dropping oldest history
    def total_tokens() -> int:
        return sum(count_tokens(m) for m in preserved + history_msgs + other_msgs) + 3  # end-of-seq fudge

    while total_tokens() > allowed_total and history_msgs:
        history_msgs.pop(0)

    return preserved + history_msgs + other_msgs

# ---------------------------------------------------
# String Truncation
# ---------------------------------------------------
def truncate_string_to_token_limit(content: str, max_token_limit: int | None = None) -> str:
    if not content:
        return ""
    max_token_limit = max_token_limit or get_token_limits().context_limit
    try:
        tokens = ENCODING.encode(content)
        return ENCODING.decode(tokens[:max_token_limit]) if len(tokens) > max_token_limit else content
    except Exception:
        # Ultra-defensive fallback if encoder/decoder raise unexpectedly
        data = content.encode("utf-8", errors="ignore")
        return data[:max_token_limit].decode("utf-8", errors="ignore") if len(data) > max_token_limit else content
