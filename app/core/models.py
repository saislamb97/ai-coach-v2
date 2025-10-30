from __future__ import annotations

import logging
from typing import Optional

from django.conf import settings
from django.core.cache import cache
from django.core.validators import MinValueValidator
from django.db import models
from django.utils import timezone

from core.constants import (
    DEFAULT_TEXT_MODEL,
    DEFAULT_EMBED_MODEL,
    TEXT_MODEL_CHOICES,
    TEXT_LIMITS,
    EMBED_MODEL_CHOICES,
    EMBED_LIMITS,
    DEFAULT_TEXT_LIMIT,
    DEFAULT_EMBED_LIMIT,
    SERVICE_CHOICES,
)

logger = logging.getLogger(__name__)
CACHE_TTL = int(getattr(settings, "CACHE_TTL", 600))


# -----------------------------
# Cache Helpers
# -----------------------------
def _ck(*parts: Optional[object]) -> str:
    """Generate a cache key from non-null parts."""
    return ":".join(str(p) for p in parts if p is not None)


class CacheMixin(models.Model):
    """Abstract base mixin providing simple caching for model instances."""

    class Meta:
        abstract = True

    def cache_write(self) -> None:
        cache.set(self.cache_key, self, timeout=CACHE_TTL)

    def cache_delete(self) -> None:
        cache.delete(self.cache_key)

    def save(self, *args, **kwargs):
        result = super().save(*args, **kwargs)
        self.cache_write()
        return result

    def delete(self, *args, **kwargs):
        self.cache_delete()
        return super().delete(*args, **kwargs)


# -----------------------------
# LLMConfig
# -----------------------------
class LLMConfig(CacheMixin, models.Model):
    """
    Configuration for text generation and embeddings (model choice, limits, etc.).
    Not enforced as a singleton. Multiple configs can coexist and one (or more)
    can be marked active.
    """

    name = models.CharField(max_length=100, unique=True, help_text="Config name (e.g. default-openai).")
    llm_service = models.CharField(max_length=50, choices=SERVICE_CHOICES, default="openai", help_text="Provider for chat completions (e.g., openai, google).")
    text_model = models.CharField(max_length=100, choices=TEXT_MODEL_CHOICES, default=DEFAULT_TEXT_MODEL)
    embed_model = models.CharField(max_length=100, choices=EMBED_MODEL_CHOICES, default=DEFAULT_EMBED_MODEL)

    is_active = models.BooleanField(default=True, db_index=True)

    token_limit = models.PositiveIntegerField(default=0, validators=[MinValueValidator(0)])
    context_limit = models.PositiveIntegerField(default=0, validators=[MinValueValidator(0)])
    history_limit = models.PositiveIntegerField(default=0, validators=[MinValueValidator(0)])
    embed_limit = models.PositiveIntegerField(default=0, validators=[MinValueValidator(0)])

    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [models.Index(fields=["is_active", "updated_at"])]
        verbose_name = "LLM Config"
        verbose_name_plural = "LLM Configs"

    def __str__(self) -> str:
        return self.name

    @property
    def cache_key(self) -> str:
        return _ck("llm_config", str(self.pk))

    def clean(self):
        """
        Ensure that all derived numeric limits are populated
        and consistent with the chosen models.
        """
        token_cap = int(TEXT_LIMITS.get(self.text_model, DEFAULT_TEXT_LIMIT))
        self.token_limit = token_cap
        self.context_limit = int(token_cap * 0.75)
        self.history_limit = int(self.context_limit * 0.5)

        self.embed_limit = int(EMBED_LIMITS.get(self.embed_model, DEFAULT_EMBED_LIMIT))

    def save(self, *args, **kwargs):
        self.clean()
        return super().save(*args, **kwargs)
