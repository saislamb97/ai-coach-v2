# agent/admin.py
from django.contrib import admin, messages
from django.utils.html import format_html, mark_safe

from .models import Voice, Agent


# -----------------------------
# Voice Admin
# -----------------------------
@admin.register(Voice)
class VoiceAdmin(admin.ModelAdmin):
    list_display = ("name", "voice_id", "service", "gender", "preview_player", "created_at")
    search_fields = ("name", "voice_id")
    list_filter = ("service", "gender", "created_at")
    readonly_fields = ("created_at", "preview_player")

    fieldsets = (
        (None, {"fields": ("name", "voice_id", "service", "gender")}),
        ("Preview", {"fields": ("preview", "preview_player")}),
        ("Timestamps", {"fields": ("created_at",)}),
    )

    def preview_player(self, obj):
        if not obj.preview:
            return "—"
        return mark_safe(
            f'<audio controls style="max-width:260px;">'
            f'  <source src="{obj.preview.url}" type="audio/mpeg">'
            f'  Your browser does not support the audio element.'
            f'</audio>'
        )

    preview_player.short_description = "Preview"


# -----------------------------
# Agent Admin
# -----------------------------
@admin.register(Agent)
class AgentAdmin(admin.ModelAdmin):
    actions = ["activate_selected", "deactivate_selected", "bulk_resave"]

    list_display = ("name", "user", "voice", "is_active", "bot_id", "created_at", "preview")
    search_fields = ("name", "bot_id", "user__username", "user__email")
    list_filter = ("is_active", "user", "created_at")
    readonly_fields = ("created_at", "updated_at", "preview")

    fieldsets = (
        ("Identity", {"fields": ("user", "name", "bot_id", "is_active")}),
        ("Behavior", {"fields": ("description", "persona", "age", "max_tokens")}),
        ("Assets", {"fields": ("voice", "avatar", "glb")}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    list_select_related = ("user", "voice")

    def preview(self, obj):
        """
        Render avatar inline if available.
        Falls back to — if no image.
        """
        img_field = getattr(obj, "avatar", None)
        url = getattr(img_field, "url", None) if img_field else None
        if not url:
            return "—"
        return format_html('<img src="{}" style="max-height:80px;border-radius:10px;" />', url)

    preview.short_description = "Preview"

    def activate_selected(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(request, f"✅ Activated {updated} Agent(s).", level=messages.SUCCESS)

    activate_selected.short_description = "✔️ Activate Selected"

    def deactivate_selected(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f"📴 Deactivated {updated} Agent(s).", level=messages.INFO)

    deactivate_selected.short_description = "📴 Deactivate Selected"

    def bulk_resave(self, request, queryset):
        for obj in queryset:
            obj.save()
        self.message_user(request, f"💾 Re-saved {queryset.count()} Agent(s).", level=messages.SUCCESS)

    bulk_resave.short_description = "💾 Bulk Save Selected"
