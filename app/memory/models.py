from __future__ import annotations

import uuid
from typing import Optional

from django.db import models
from django.utils import timezone

from agent.models import Agent
from user.models import User


# -----------------------------
# Utilities
# -----------------------------
def generate_thread_id() -> str:
    """Generate external-facing thread_id like 'user_<16hex>' or 'user_<16hex>'."""
    return f"user_{uuid.uuid4().hex[:16]}"


# -----------------------------
# Base mixins
# -----------------------------
class TimeStampedModel(models.Model):
    """Abstract base with created/updated timestamps."""
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class TenantModel(models.Model):
    """
    Associates rows with a specific (user, agent) pair.
    This is your multitenancy boundary.
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="%(class)ss")
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name="%(class)ss")

    class Meta:
        abstract = True

    @property
    def tenant(self) -> tuple[int, int | None]:
        return (self.user_id, self.agent_id)


# -----------------------------
# Session
# -----------------------------
class Session(TenantModel, TimeStampedModel):
    """
    A conversational session between a user and an agent.
    We expose thread_id as the stable external ID for this session.
    """

    thread_id = models.CharField(max_length=100, unique=True, db_index=True, default=generate_thread_id)

    title = models.CharField(max_length=255, blank=True, default="")
    summary = models.TextField(blank=True, default="")
    is_active = models.BooleanField(default=True)

    class Meta:
        indexes = [models.Index(fields=["user", "agent", "created_at"])]
        ordering = ["-created_at"]
        verbose_name = "Session"
        verbose_name_plural = "Sessions"

    def __str__(self) -> str:
        return f"Session<{self.thread_id}>"


# -----------------------------
# Chat
# -----------------------------
class Chat(TimeStampedModel):
    """
    A single chat turn within a session.

    Fields:
    - query: user's message
    - response: assistant reply
    - emotion: JSON blob with { "name": "joy" | "anger" | "sadness", "intensity": 1-3 }
    - viseme: JSON blob describing the TTS/viseme frames for the assistant response
      Example:
      {
        "frame_ms": 100,
        "frames": [
          [0.0,0.1,0.0,0.3,...],
          [0.05,0.15,0.05,0.35,...]
        ],
        "audio_ref": "s3://bucket/agent/clip.mp3"
      }
    - meta: misc runtime/diagnostic info
    """

    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="chats", db_index=True)

    query = models.TextField(help_text="User message / prompt.")
    response = models.TextField(blank=True, default="", help_text="Assistant final reply text/markdown.")

    emotion = models.JSONField(blank=True, default=dict, help_text='e.g. {"name":"joy","intensity":2}')
    viseme = models.JSONField(blank=True, default=dict, help_text="Viseme timeline (+ optional audio ref) for this assistant response.")
    meta = models.JSONField(blank=True, default=dict, help_text="Debug/runtime metadata, timings, routing info, etc.")

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["session", "created_at"]),
            models.Index(fields=["created_at"]),
        ]
        verbose_name = "Chat"
        verbose_name_plural = "Chats"

    def __str__(self) -> str:
        return f"Chat<{self.pk}> thread={self.session.thread_id} agent={self.session.agent.bot_id}"


# -----------------------------
# Slides (current deck for this session)
# -----------------------------
class Slides(TimeStampedModel):
    """
    The current slide deck (editor.js format) associated with a session.

    Notes:
    - There is at most one Slides row per Session (OneToOne).
    - We no longer track historical versions or run_id.
    - 'editorjs' stores the canonical editor.js doc.
    - 'summary' gives a compact bullet/outline version for RAG / preview.
    - 'title' is a short human-readable label for the deck.

    This supports:
      POST /api/slides/update/?session_id=...
      GET /api/slides/latest/?session_id=...

    Implementation detail:
      - "latest" is just this row.
      - If you overwrite slides, you UPDATE this same row.
    """

    session = models.OneToOneField(Session, on_delete=models.CASCADE, related_name="slides", db_index=True)

    title = models.CharField(max_length=255, blank=True, default="", help_text="Short display name for this slide deck.")
    summary = models.TextField(blank=True, default="", help_text="Short outline / summary of the deck for fast context use.")
    editorjs = models.JSONField(blank=True, default=dict, help_text="Full editor.js JSON for the current slide deck.")

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Slides"
        verbose_name_plural = "Slides"

    def __str__(self) -> str:
        return f"Slides<thread={self.session.thread_id} title={self.title!r}>"
