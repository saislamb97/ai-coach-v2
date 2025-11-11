# agent/models.py
from __future__ import annotations

import uuid
import logging
from typing import Optional

from django.conf import settings
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone

from agent.utils import agent_upload_path, voice_upload_path
from core.constants import SERVICE_CHOICES, GENDER_CHOICES, ALL_VOICE_CHOICES
from user.models import User

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
# Voice
# -----------------------------
class Voice(CacheMixin, models.Model):
    """Represents a voice configuration for a text-to-speech provider."""

    name = models.CharField(max_length=255, unique=True)
    service = models.CharField(max_length=150, choices=SERVICE_CHOICES, default="gtts", help_text="TTS provider (e.g., gtts, elevenlabs).")
    voice_id = models.CharField(max_length=255, choices=ALL_VOICE_CHOICES, default="gtts::en@com", help_text="Voice ID with service prefix (e.g., gtts::en@com, openai::alloy).")
    gender = models.CharField(max_length=150, choices=GENDER_CHOICES, default="female", help_text="Voice gender.")
    preview = models.FileField(upload_to=voice_upload_path, max_length=250, blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["created_at", "pk"]
        indexes = [models.Index(fields=["service", "gender"])]
        verbose_name = "Voice"
        verbose_name_plural = "Voices"

    def __str__(self) -> str:
        return self.name

    @property
    def cache_key(self) -> str:
        return _ck("voice", self.pk)


# -----------------------------
# Agent
# -----------------------------
class Agent(CacheMixin, models.Model):
    """Represents an AI agent with a specific voice and configuration."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="agents")
    voice = models.ForeignKey(Voice, on_delete=models.SET_NULL, null=True, related_name="agents")
    bot_id = models.UUIDField(default=uuid.uuid4, unique=True, db_index=True)

    name = models.CharField(max_length=250)
    description = models.TextField(max_length=500, blank=True)
    persona = models.TextField(max_length=500, blank=True)
    age = models.IntegerField(blank=True, null=True)
    max_tokens = models.IntegerField(default=300)

    glb = models.URLField(max_length=500, blank=True, null=True)
    avatar = models.ImageField(upload_to=agent_upload_path, max_length=250, blank=True, null=True)

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "is_active"]),
            models.Index(fields=["created_at"]),
        ]
        verbose_name = "Agent"
        verbose_name_plural = "Agents"

    def __str__(self) -> str:
        return self.name

    @property
    def cache_key(self) -> str:
        return _ck("agent", str(self.bot_id))

    def clean(self):
        """Ensure that the agent always has a valid voice assigned."""
        if not self.voice:
            first_voice = Voice.objects.first()
            if first_voice:
                self.voice = first_voice
            else:
                raise ValidationError("No Voice available. Please create a Voice before creating an Agent.")

    def save(self, *args, **kwargs):
        self.clean()
        return super().save(*args, **kwargs)
