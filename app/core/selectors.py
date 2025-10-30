# core/selectors.py
from __future__ import annotations

from typing import Optional
import logging

from django.conf import settings
from django.core.cache import cache

from .models import LLMConfig

log = logging.getLogger(__name__)
CACHE_TTL: int = int(getattr(settings, "CACHE_TTL", 600))


def get_llm_config() -> Optional[LLMConfig]:
    """
    Return the most recently updated active LLMConfig.
    Uses cache for speed.
    If nothing active exists, returns None.
    """
    cache_key = "llm_config:active_latest"

    cached = cache.get(cache_key)
    if cached:
        # If it was cached and it's still active, return it. If it's become inactive, ignore it.
        return cached if cached.is_active else None

    try:
        obj = (
            LLMConfig.objects
            .filter(is_active=True)
            .order_by("-updated_at")
            .first()
        )
    except Exception as e:
        # Stay defensive: DB might not be migrated yet, etc.
        log.debug("[core.selectors] get_llm_config failed: %s", e)
        return None

    if obj:
        cache.set(cache_key, obj, timeout=CACHE_TTL)
    return obj
