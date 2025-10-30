# core/admin.py
from __future__ import annotations

from django.contrib import admin, messages
from django.utils import timezone
from django.core.cache import cache

from .models import LLMConfig


@admin.register(LLMConfig)
class LLMConfigAdmin(admin.ModelAdmin):
    """
    Admin for LLMConfig.
    - You can create multiple rows.
    - You can toggle is_active on any of them.
    - Includes utility actions for bulk activate/deactivate/re-save and cache clear.
    """

    actions = ["activate_selected", "deactivate_selected", "bulk_resave", "clear_cache"]

    list_display = (
        "name",
        "llm_service",
        "text_model",
        "embed_model",
        "token_limit",
        "context_limit",
        "history_limit",
        "embed_limit",
        "is_active",
        "updated_at",
    )

    search_fields = ("name", "llm_service", "text_model", "embed_model")
    list_filter = ("is_active", "llm_service", "text_model", "embed_model", "updated_at")
    readonly_fields = ("created_at", "updated_at", "token_limit", "context_limit", "history_limit", "embed_limit")

    fieldsets = (
        ("Identity", {"fields": ("name", "is_active")}),
        ("Models", {"fields": ("llm_service", "text_model", "embed_model")}),
        ("Computed Limits (read-only)", {
            "fields": ("token_limit", "context_limit", "history_limit", "embed_limit"),
        }),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    # --------------- Actions ---------------

    def activate_selected(self, request, queryset):
        """
        Mark selected configs as active.
        Does NOT auto-deactivate others (because LLMConfig is not singleton).
        """
        updated = queryset.update(is_active=True, updated_at=timezone.now())
        self.message_user(request, f"✅ Activated {updated} configuration(s).", level=messages.SUCCESS)

    activate_selected.short_description = "✔️ Activate selected"

    def deactivate_selected(self, request, queryset):
        updated = queryset.update(is_active=False, updated_at=timezone.now())
        self.message_user(request, f"📴 Deactivated {updated} configuration(s).", level=messages.INFO)

    deactivate_selected.short_description = "📴 Deactivate selected"

    def bulk_resave(self, request, queryset):
        """
        Re-run clean() + save() to recompute numeric limits on each row.
        """
        count = 0
        for obj in queryset:
            obj.save()
            count += 1
        self.message_user(request, f"💾 Re-saved {count} row(s).", level=messages.SUCCESS)

    bulk_resave.short_description = "💾 Bulk Save (recompute derived limits)"

    def clear_cache(self, request, queryset):
        """
        Clear any cached active config reference used by selectors.get_llm_config().
        We'll just delete the shared key and let it repopulate later.
        """
        cache_key = "llm_config:active_latest"
        cache.delete(cache_key)
        self.message_user(request, "🧹 Cache cleared for active LLMConfig.", level=messages.SUCCESS)

    clear_cache.short_description = "🧹 Clear selector cache"


# Customize admin branding
admin.site.site_header = "AI Coach Admin Panel"
admin.site.site_title = "AI Coach Admin Portal"
admin.site.index_title = "Welcome to AI Coach Admin Dashboard"
