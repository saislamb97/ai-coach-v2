from __future__ import annotations

import json
from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import mark_safe

from .models import Session, Chat, Slides


# ==========================================================
# Utilities
# ==========================================================
class PrettyMixin:
    """Helpers for HTML previews & JSON pretty-print in readonly admin fields."""

    @staticmethod
    def _preview(text: str | None, limit: int = 300):
        if not text:
            return mark_safe("<em>—</em>")
        t = (text or "").strip()
        if len(t) > limit:
            t = t[:limit].rstrip() + "…"
        # format_html() protects against unsafe HTML in t
        return format_html("{}", t)

    @staticmethod
    def _json_pretty(data, max_height: int = 320):
        if not data:
            return mark_safe("<em>—</em>")
        try:
            body = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            body = str(data)
        return mark_safe(
            f"<pre style='white-space:pre-wrap;max-height:{max_height}px;overflow:auto;margin:0'>{body}</pre>"
        )


# ==========================================================
# Session
# ==========================================================
@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    """
    Admin for conversation sessions. Replaces VisitorAdmin.
    """

    date_hierarchy = "created_at"
    ordering = ("-updated_at",)
    list_select_related = ("user", "agent")
    list_per_page = 25

    list_display = (
        "thread_id",
        "user",
        "agent",
        "title",
        "is_active",
        "updated_at",
    )
    list_filter = ("agent", "user", "is_active", "created_at")
    search_fields = ("thread_id", "title", "summary")

    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        (None, {"fields": ("user", "agent", "thread_id", "is_active")}),
        ("Details", {"fields": ("title", "summary")}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def __str__(self):
        return self.thread_id


# ==========================================================
# Chat
# ==========================================================
@admin.register(Chat)
class ChatAdmin(PrettyMixin, admin.ModelAdmin):
    """
    Admin for individual chat turns inside a session.
    """

    date_hierarchy = "created_at"
    ordering = ("-created_at",)
    list_select_related = ("session", "session__user", "session__agent")
    list_per_page = 25

    list_display = (
        "session_thread",
        "agent_display",
        "user_display",
        "query_preview",
        "created_at",
    )
    list_filter = ("session__user", "session__agent", "created_at")
    search_fields = (
        "query",
        "response",
        "session__thread_id",
        "session__title",
        "session__summary",
    )

    readonly_fields = (
        "created_at",
        "updated_at",
        "session",
        "agent_display",
        "user_display",
        "query_full",
        "response_full",
        "emotion_pretty",
        "viseme_pretty",
        "meta_pretty",
    )

    fieldsets = (
        (None, {"fields": ("session", "agent_display", "user_display")}),
        ("Query / Response", {"fields": ("query_full", "response_full")}),
        ("Emotion", {"fields": ("emotion_pretty",)}),
        ("Viseme / Audio", {"fields": ("viseme_pretty",)}),
        ("Meta / Debug", {"fields": ("meta_pretty",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    # ---- virtual columns / readonly renderers ----
    @admin.display(description="Thread")
    def session_thread(self, obj: Chat):
        return getattr(obj.session, "thread_id", "—")

    @admin.display(description="Agent")
    def agent_display(self, obj: Chat):
        return getattr(obj.session, "agent", "—")

    @admin.display(description="User")
    def user_display(self, obj: Chat):
        return getattr(obj.session, "user", "—")

    @admin.display(description="Query")
    def query_preview(self, obj: Chat):
        return self._preview(obj.query, 120)

    @admin.display(description="Query (full)")
    def query_full(self, obj: Chat):
        return self._preview(obj.query, 2000)

    @admin.display(description="Response (full)")
    def response_full(self, obj: Chat):
        return self._preview(obj.response, 4000)

    @admin.display(description="Emotion")
    def emotion_pretty(self, obj: Chat):
        return self._json_pretty(obj.emotion, max_height=200)

    @admin.display(description="Viseme")
    def viseme_pretty(self, obj: Chat):
        return self._json_pretty(obj.viseme, max_height=240)

    @admin.display(description="Meta")
    def meta_pretty(self, obj: Chat):
        return self._json_pretty(obj.meta, max_height=240)


# ==========================================================
# Slides
# ==========================================================
@admin.register(Slides)
class SlidesAdmin(PrettyMixin, admin.ModelAdmin):
    """
    Admin for the current slide deck attached to a session.
    One row per session (OneToOne).
    """

    date_hierarchy = "updated_at"
    ordering = ("-updated_at",)
    list_select_related = ("session", "session__user", "session__agent")
    list_per_page = 25

    list_display = (
        "session_thread",
        "agent_display",
        "user_display",
        "title",
        "updated_at",
    )
    list_filter = ("session__agent", "session__user", "updated_at", "created_at")
    search_fields = ("session__thread_id", "title", "summary")

    readonly_fields = ("created_at", "updated_at", "session_pretty", "editorjs_pretty")

    fieldsets = (
        ("Session", {"fields": ("session_pretty",)}),
        ("Slides Info", {"fields": ("title", "summary")}),
        ("editor.js Document", {"fields": ("editorjs_pretty",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    # ---- virtuals / readonly renders ----
    @admin.display(description="Thread")
    def session_thread(self, obj: Slides):
        return getattr(obj.session, "thread_id", "—")

    @admin.display(description="Agent")
    def agent_display(self, obj: Slides):
        return getattr(obj.session, "agent", "—")

    @admin.display(description="User")
    def user_display(self, obj: Slides):
        return getattr(obj.session, "user", "—")

    @admin.display(description="Session")
    def session_pretty(self, obj: Slides):
        sess = obj.session
        if not sess:
            return mark_safe("<em>—</em>")
        return format_html(
            "<div>"
            "<div><b>Thread:</b> {thread}</div>"
            "<div><b>User:</b> {user}</div>"
            "<div><b>Agent:</b> {agent}</div>"
            "<div><b>Title:</b> {title}</div>"
            "<div><b>Summary:</b> {summary}</div>"
            "</div>",
            thread=sess.thread_id,
            user=sess.user,
            agent=sess.agent,
            title=(sess.title or "—"),
            summary=(sess.summary or "—"),
        )

    @admin.display(description="editor.js (pretty)")
    def editorjs_pretty(self, obj: Slides):
        return self._json_pretty(obj.editorjs, max_height=360)
