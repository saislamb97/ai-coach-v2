from __future__ import annotations

import secrets
import json

from django.contrib import admin, messages
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.admin.models import LogEntry
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.urls import path, reverse
from django.shortcuts import redirect

from .models import User, ApiAuth


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
# 👤 User Admin
# ==========================================================
@admin.register(User)
class CustomUserAdmin(BaseUserAdmin):
    """
    Admin for the custom User model.
    - Login is by email.
    - username is auto-generated / display only.
    """

    model = User
    ordering = ("-date_joined",)
    list_display = ("email", "username", "first_name", "last_name", "is_staff", "is_active")
    list_filter = ("is_staff", "is_superuser", "is_active", "country")
    search_fields = ("email", "username", "first_name", "last_name")

    readonly_fields = ("last_login", "date_joined",)

    fieldsets = (
        (None, {"fields": ("email", "username", "password")}),
        (
            "Personal Information",
            {
                "fields": (
                    "first_name",
                    "last_name",
                    "profile",
                    "country",
                    "company",
                    "postcode",
                    "address",
                    "website",
                )
            },
        ),
        (
            "Roles & Permissions",
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                )
            },
        ),
        ("Important Dates", {"fields": ("last_login", "date_joined")}),
    )

    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": ("email", "password1", "password2"),
            },
        ),
    )


# ==========================================================
# 🔑 ApiAuth Admin
# ==========================================================
@admin.register(ApiAuth)
class ApiAuthAdmin(admin.ModelAdmin):
    """
    Manage API auth records:
    - api_key
    - allowed_origins
    - active status

    Includes inline secure-ish regeneration of api_key.
    """

    list_display = (
        "name",
        "user",
        "public_id",
        "short_api_key",
        "is_active",
        "created_at",
        "updated_at",
    )
    list_filter = ("is_active", "created_at", "user")
    search_fields = ("name", "user__email", "user__username", "api_key", "public_id")
    ordering = ("-created_at",)
    list_select_related = ("user",)
    list_per_page = 25

    readonly_fields = (
        "api_key_display",
        "public_id",
        "created_at",
        "updated_at",
    )

    fieldsets = (
        (None, {"fields": ("user", "name", "description", "is_active")}),
        ("Access Control", {"fields": ("allowed_origins",)}),
        ("Keys / Metadata", {"fields": ("api_key_display", "public_id", "created_at", "updated_at")}),
    )

    #
    # ----- Display helpers -----
    #

    @admin.display(description="API Key")
    def short_api_key(self, obj: ApiAuth):
        """
        Show a shortened API key in list view, to avoid leaking the whole secret.
        """
        if not obj.api_key:
            return "—"
        if len(obj.api_key) <= 10:
            return obj.api_key
        return f"{obj.api_key[:6]}...{obj.api_key[-4:]}"

    @admin.display(description="API Key (Full / Rotate)")
    def api_key_display(self, obj: ApiAuth):
        """
        Show full API key and a regenerate button.
        NOTE: This still exposes the full key in admin. If you later
        move to hashed keys, this should change accordingly.
        """
        if not obj or not obj.pk:
            return "—"

        regenerate_url = reverse("admin:user_apiauth_regenerate_key", args=[obj.pk])
        return format_html(
            "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:8px;'>"
            "<input type='text' value='{}' readonly "
            "style='width:380px;font-family:monospace;border-radius:4px;padding:4px;' />"
            "<a class='button' href='{}' "
            "style='background:#2e6da4;color:#fff;padding:5px 10px;border-radius:4px;'>Regenerate</a>"
            "</div>",
            obj.api_key,
            regenerate_url,
        )

    #
    # ----- Custom URL for key rotation -----
    #

    def get_urls(self):
        """
        Add a custom route to regenerate an API key from the admin.
        """
        urls = super().get_urls()
        custom_urls = [
            path(
                "<int:auth_id>/regenerate-key/",
                self.admin_site.admin_view(self.regenerate_key_view),
                name="user_apiauth_regenerate_key",
            ),
        ]
        return custom_urls + urls

    def regenerate_key_view(self, request, auth_id, *args, **kwargs):
        """
        Rotate the API key in-place and show the new value in a success message.
        """
        apiauth = self.get_object(request, auth_id)
        if not apiauth:
            self.message_user(request, "API Auth not found.", level=messages.ERROR)
            return redirect("..")

        new_key = secrets.token_urlsafe(32)
        apiauth.api_key = new_key
        apiauth.save(update_fields=["api_key"])

        self.message_user(
            request,
            format_html(
                "✅ API key for <b>{}</b> regenerated successfully.<br>"
                "<b>New key:</b> <code style='font-family:monospace;'>{}</code>",
                apiauth.name,
                new_key,
            ),
            level=messages.SUCCESS,
        )
        return redirect(reverse("admin:user_apiauth_change", args=[apiauth.id]))

    #
    # ----- Permissions -----
    #

    def has_add_permission(self, request):
        # Allow adding new ApiAuth rows.
        return True

    def has_delete_permission(self, request, obj=None):
        # Only superusers can delete ApiAuth rows.
        return request.user.is_superuser


# ==========================================================
# 🕵️ LogEntry Admin (read-only)
# ==========================================================
@admin.register(LogEntry)
class LogEntryAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "content_type",
        "object_repr",
        "action_flag",
        "change_message",
        "action_time",
    )
    list_filter = ("action_flag", "content_type")
    search_fields = ("object_repr", "change_message", "user__email")
    ordering = ("-action_time",)
    readonly_fields = [f.name for f in LogEntry._meta.get_fields() if f.concrete]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        # Only superusers can purge logs.
        return request.user.is_superuser
