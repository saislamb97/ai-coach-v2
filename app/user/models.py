from __future__ import annotations

import os
import uuid
import secrets
import logging

from django.conf import settings
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models

from core.constants import COUNTRY_CHOICES

logger = logging.getLogger(__name__)


# -----------------------------
# Utilities
# -----------------------------
def profile_upload_path(instance, filename):
    """
    Build a stable-ish per-user upload path for profile images.
    e.g. profiles/alex/1f2e9c..._avatar.png
    """
    base = (instance.username or instance.email.split("@")[0]).lower().replace(" ", "_")
    return os.path.join(f"profiles/{base}/", f"{uuid.uuid4().hex}_{filename}")


def create_unique_username(email: str) -> str:
    """
    Derive a unique username based on the local part of the email.
    Falls back to appending a counter if needed.
    """
    base = email.split("@")[0]
    candidate = base
    counter = 1
    while User.objects.filter(username=candidate).exists():
        candidate = f"{base}{counter}"
        counter += 1
    return candidate


def default_allowed_origins():
    """
    Default allowlist for API usage (CORS / Host / Referer checks).
    These reflect dev/staging defaults, plus BASE_URL if present.
    """
    base_url = getattr(settings, "BASE_URL", "http://localhost:8002")
    return [
        "http://127.0.0.1:8002",
        "http://localhost:8002",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        base_url,
    ]


# -----------------------------
# Custom User Manager
# -----------------------------
class UserManager(BaseUserManager):
    """
    Email is the primary login identifier.
    """

    use_in_migrations = True

    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("An email address is required.")

        email = self.normalize_email(email)
        extra_fields.setdefault("is_active", True)

        user = self.model(email=email, **extra_fields)
        if password:
            user.set_password(password)
        else:
            user.set_unusable_password()

        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        """
        Create an admin/superuser account.
        """
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if not extra_fields.get("is_staff") or not extra_fields.get("is_superuser"):
            raise ValueError("Superuser must have is_staff=True and is_superuser=True.")

        return self.create_user(email, password, **extra_fields)


# -----------------------------
# User model
# -----------------------------
class User(AbstractUser):
    """
    Primary user model.
    - Auth: email is the login field (USERNAME_FIELD = 'email').
    - username is still stored to keep Django admin happy and for display purposes,
      but it's auto-generated and not relied on for auth.
    """

    email = models.EmailField(unique=True)
    username = models.CharField(max_length=150, unique=True, null=True, blank=True)
    profile = models.ImageField(upload_to=profile_upload_path, blank=True, null=True)

    country = models.CharField(max_length=2, choices=COUNTRY_CHOICES, default="MY", blank=True, null=True)
    company = models.CharField(max_length=255, blank=True, default="")
    postcode = models.CharField(max_length=20, blank=True, default="")
    address = models.CharField(max_length=255, blank=True, default="")
    website = models.URLField(blank=True, default="")

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS: list[str] = []

    objects = UserManager()

    class Meta:
        indexes = [models.Index(fields=["email"])]
        verbose_name = "User"
        verbose_name_plural = "Users"

    def __str__(self) -> str:
        return self.email

    def save(self, *args, **kwargs):
        """
        Ensure username is always populated/unique even though login is email.
        """
        if not self.username and self.email:
            self.username = create_unique_username(self.email)
        super().save(*args, **kwargs)


# -----------------------------
# ApiAuth model
# -----------------------------
class ApiAuth(models.Model):
    """
    API key auth / allowlist for frontend and WS access.

    NOTE: For production hardening you almost certainly want to store a hash
    instead of the raw api_key. In that model you'd keep api_key_hash here,
    and only show the plaintext key once on creation.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="api_auths",
        db_index=True,
    )

    name = models.CharField(max_length=150, help_text="Human label for this key (per client/app).")
    description = models.TextField(blank=True, default="")
    api_key = models.CharField(max_length=100, unique=True, editable=False, db_index=True)

    allowed_origins = models.JSONField(default=default_allowed_origins, blank=True, help_text="Allowed Origin/Host/IP patterns for this key.")
    public_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["api_key"]),
        ]
        ordering = ["-created_at"]
        verbose_name = "API Auth"
        verbose_name_plural = "API Auths"
        unique_together = ("user", "name")

    def __str__(self) -> str:
        return f"ApiAuth<{self.name}>"

    def save(self, *args, **kwargs):
        """
        Auto-generate api_key if missing.
        Currently this stores plaintext. To upgrade later:
        - generate plaintext_key = secrets.token_urlsafe(32)
        - hash it before save (e.g. sha256) into api_key_hash
        - return plaintext once to caller.
        """
        if not self.api_key:
            self.api_key = secrets.token_urlsafe(32)
        super().save(*args, **kwargs)
