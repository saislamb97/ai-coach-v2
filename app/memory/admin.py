from __future__ import annotations

import json
from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import mark_safe

from .models import (
    Session, Chat,
    Slides, SlidesRevision,
    Knowledge,
)

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
        return format_html("{}", t)

    @staticmethod
    def _json_pretty(data, max_height: int = 320):
        if data in (None, {}, [], ""):
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
    """Admin for conversation sessions."""
    date_hierarchy = "created_at"
    ordering = ("-updated_at",)
    list_select_related = ("user", "agent")
    list_per_page = 25

    list_display = ("thread_id", "user", "agent", "title", "is_active", "updated_at")
    list_filter = ("agent", "user", "is_active", "created_at")
    search_fields = ("thread_id", "title", "summary")

    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        (None, {"fields": ("user", "agent", "thread_id", "is_active")}),
        ("Details", {"fields": ("title", "summary")}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )


# ==========================================================
# Chat
# ==========================================================
@admin.register(Chat)
class ChatAdmin(PrettyMixin, admin.ModelAdmin):
    """Admin for individual chat turns inside a session."""
    date_hierarchy = "created_at"
    ordering = ("-created_at",)
    list_select_related = ("session", "session__user", "session__agent")
    list_per_page = 25

    list_display = ("session_thread", "agent_display", "user_display", "query_preview", "created_at")
    list_filter = ("session__user", "session__agent", "created_at")
    search_fields = ("query", "response", "session__thread_id", "session__title", "session__summary")

    readonly_fields = (
        "created_at", "updated_at", "session",
        "agent_display", "user_display",
        "query_full", "response_full",
        "emotion_pretty", "viseme_pretty", "meta_pretty",
    )

    fieldsets = (
        (None, {"fields": ("session", "agent_display", "user_display")}),
        ("Query / Response", {"fields": ("query_full", "response_full")}),
        ("Emotion", {"fields": ("emotion_pretty",)}),
        ("Viseme / Audio", {"fields": ("viseme_pretty",)}),
        ("Meta / Debug", {"fields": ("meta_pretty",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

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
# Slides + Revisions
# ==========================================================
class SlidesRevisionInline(PrettyMixin, admin.StackedInline):
    """Read-only inline showing the latest 3 snapshots for quick eye-balling."""
    model = SlidesRevision
    extra = 0
    can_delete = False
    verbose_name = "Recent revision"
    verbose_name_plural = "Recent revisions (latest 3)"
    fields = ("version", "updated_by", "created_at", "title", "summary", "editorjs_pretty")
    readonly_fields = fields

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.order_by("-version")[:3]

    @admin.display(description="editor.js (pretty)")
    def editorjs_pretty(self, obj: SlidesRevision):
        return self._json_pretty(obj.editorjs, max_height=260)


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

    list_display = ("session_thread", "agent_display", "user_display", "title", "version", "updated_by", "updated_at")
    list_filter = ("session__agent", "session__user", "updated_at", "created_at")
    search_fields = ("session__thread_id", "title", "summary")

    readonly_fields = (
        "created_at", "updated_at",
        "session_pretty",
        "version",
        "editorjs_pretty",
        "history_latest_html",
    )

    fieldsets = (
        ("Session", {"fields": ("session_pretty",)}),
        ("Slides Info", {"fields": ("title", "summary", "version", "updated_by")}),
        ("Current editor.js", {"fields": ("editorjs_pretty",)}),
        ("History (latest 3)", {"fields": ("history_latest_html",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    inlines = [SlidesRevisionInline]

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
            "<div><b>Session Title:</b> {title}</div>"
            "<div><b>Session Summary:</b> {summary}</div>"
            "</div>",
            thread=sess.thread_id,
            user=sess.user,
            agent=sess.agent,
            title=(sess.title or "—"),
            summary=(sess.summary or "—"),
        )

    @admin.display(description="editor.js (pretty) — current")
    def editorjs_pretty(self, obj: Slides):
        return self._json_pretty(obj.editorjs, max_height=360)

    @admin.display(description="History (latest 3)")
    def history_latest_html(self, obj: Slides):
        revs = obj.session.slide_revisions.order_by("-version")[:3]
        if not revs:
            return mark_safe("<em>No revisions yet</em>")
        items = []
        for r in revs:
            items.append(
                f"<li><b>v{r.version}</b> • {r.updated_by or '—'} • {r.created_at:%Y-%m-%d %H:%M}</li>"
            )
        return mark_safe("<ul style='margin:0;padding-left:18px'>" + "".join(items) + "</ul>")


@admin.register(SlidesRevision)
class SlidesRevisionAdmin(PrettyMixin, admin.ModelAdmin):
    """Browse & revert specific revisions."""
    date_hierarchy = "created_at"
    ordering = ("-version", "-created_at")
    list_select_related = ("session", "session__user", "session__agent")
    list_per_page = 25

    list_display = ("session_thread", "version", "updated_by", "created_at")
    list_filter = ("session__agent", "session__user", "updated_by")
    search_fields = ("session__thread_id", "title", "summary")

    readonly_fields = ("created_at", "updated_at", "session", "version", "title", "summary", "editorjs_pretty", "updated_by")

    fieldsets = (
        ("Session / Version", {"fields": ("session", "version", "updated_by", "created_at", "updated_at")}),
        ("Title / Summary", {"fields": ("title", "summary")}),
        ("editor.js", {"fields": ("editorjs_pretty",)}),
    )

    actions = ("revert_to_selected",)

    @admin.display(description="Thread")
    def session_thread(self, obj: SlidesRevision):
        return getattr(obj.session, "thread_id", "—")

    @admin.display(description="editor.js (pretty)")
    def editorjs_pretty(self, obj: SlidesRevision):
        return self._json_pretty(obj.editorjs, max_height=360)

    @admin.action(description="Revert slides to the selected revision(s)")
    def revert_to_selected(self, request, queryset):
        count = 0
        for rev in queryset:
            slides = getattr(rev.session, "slides", None)
            if slides is None:
                slides = Slides.objects.create(session=rev.session, version=0, title="", summary="", editorjs={})
            slides.revert_to_version(target_version=rev.version, updated_by=f"admin:{request.user}")
            count += 1
        self.message_user(request, f"Reverted {count} revision(s). A new current version was created for each revert.")


# ==========================================================
# Knowledge
# ==========================================================
@admin.register(Knowledge)
class KnowledgeAdmin(PrettyMixin, admin.ModelAdmin):
    date_hierarchy = "created_at"
    ordering = ("-created_at",)
    list_select_related = ("user", "agent")
    list_per_page = 25

    list_display = (
        "short_name", "agent", "user",
        "key", "mimetype", "size_kb",
        "index_status", "created_at",
    )
    list_filter = ("agent", "user", "index_status", "mimetype", "created_at")
    search_fields = ("original_name", "title", "key", "normalized_name", "search_terms")

    readonly_fields = (
        "created_at", "updated_at",
        "file_link",
        "mimetype", "size_bytes", "sha256", "ext",
        "pages", "rows", "cols",
        "excerpt_preview", "meta_pretty", "index_meta_pretty",
        "normalized_name", "search_terms",
    )

    fields = (
        ("user", "agent"),
        ("title", "original_name"),
        "file",
        ("mimetype", "ext", "size_bytes"),
        ("pages", "rows", "cols"),
        "excerpt",
        ("index_status",),
        ("created_at", "updated_at"),
        "file_link",
        "normalized_name", "search_terms",
        "meta_pretty",
        "index_meta_pretty",
    )

    actions = ("action_extract_and_index", "mark_unindexed")

    @admin.display(description="Name")
    def short_name(self, obj: Knowledge):
        return obj.display_name

    @admin.display(description="Size (KB)")
    def size_kb(self, obj: Knowledge):
        try:
            return f"{obj.size_bytes // 1024:,}"
        except Exception:
            return "—"

    @admin.display(description="File")
    def file_link(self, obj: Knowledge):
        if not obj.file:
            return mark_safe("<em>—</em>")
        url = obj.file.url
        return mark_safe(f"<a href='{url}' target='_blank' rel='noopener'>download</a>")

    @admin.display(description="Excerpt (preview)")
    def excerpt_preview(self, obj: Knowledge):
        return self._preview(obj.excerpt, 600)

    @admin.display(description="Meta")
    def meta_pretty(self, obj: Knowledge):
        return self._json_pretty(obj.meta, max_height=240)

    @admin.display(description="Index Meta")
    def index_meta_pretty(self, obj: Knowledge):
        return self._json_pretty(obj.index_meta, max_height=240)

    @admin.action(description="Extract & index selected")
    def action_extract_and_index(self, request, queryset):
        # optional admin reindex; Celery already does this on create
        ok = 0
        for asset in queryset:
            try:
                asset.extract_and_index()
                asset.refresh_search_terms()
                asset.save(update_fields=[
                    "size_bytes","mimetype","sha256","ext",
                    "pages","rows","cols","excerpt","meta",
                    "index_status","index_meta","normalized_name","search_terms","updated_at"
                ])
                ok += 1
            except Exception:
                pass
        self.message_user(request, f"Processed {ok} item(s).")

    @admin.action(description="Mark as unindexed")
    def mark_unindexed(self, request, queryset):
        n = queryset.update(index_status=Knowledge.IndexStatus.UNINDEXED)
        self.message_user(request, f"Marked {n} item(s) as unindexed.")
