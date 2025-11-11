# app/memory/models.py
from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Set, Tuple

from django.core.files.storage import default_storage
from django.core.validators import FileExtensionValidator, RegexValidator
from django.db import models, transaction
from django.db.models import F, Q, UniqueConstraint
from django.utils import timezone

from agent.models import Agent
from user.models import User
from memory.extract import SUPPORTED_EXTS, extract

logger = logging.getLogger(__name__)

# =============================================================================
# Utilities
# =============================================================================

# allow \t \n, strip other control chars
_CTRL_RE = re.compile(r"[\x00\x01-\x08\x0B\x0C\x0E-\x1F\x7F]")


def generate_thread_id() -> str:
    """Generate external-facing thread_id like 'user_<16hex>'."""
    return f"user_{uuid.uuid4().hex[:16]}"


def _slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").strip().lower())
    return re.sub(r"-{2,}", "-", s).strip("-") or "agent"


def _ext_of(name: str) -> str:
    return os.path.splitext(name or "")[1].lower()


def _strip_ctrl(s: Any) -> str:
    """Remove control chars except \\t, \\n; normalize newlines."""
    s = str(s or "").replace("\r\n", "\n").replace("\r", "\n")
    return _CTRL_RE.sub("", s)


def _clean_json(obj: Any) -> Any:
    """Recursively remove control chars from dict/list values."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, str):
        return _strip_ctrl(obj)
    return obj


# =============================================================================
# Abstract Base Mixins
# =============================================================================

class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class TenantModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="%(class)ss")
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name="%(class)ss")

    class Meta:
        abstract = True

    @property
    def tenant(self) -> Tuple[int, int]:
        return self.user_id, self.agent_id


# =============================================================================
# Session
# =============================================================================

class Session(TenantModel, TimeStampedModel):
    thread_id = models.CharField(
        max_length=100,
        unique=True,
        db_index=True,
        default=generate_thread_id,
        validators=[
            RegexValidator(
                regex=r"^[a-z0-9_][a-z0-9_\-]{3,99}$",
                message="thread_id must be 4–100 chars of lowercase letters, digits, '_' or '-'.",
            )
        ],
        help_text="Stable external-facing conversation identifier.",
    )
    title = models.CharField(max_length=255, blank=True, default="")
    summary = models.TextField(blank=True, default="")
    is_active = models.BooleanField(default=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "agent", "created_at"]),
        ]
        ordering = ["-created_at"]
        verbose_name = "Session"
        verbose_name_plural = "Sessions"

    def __str__(self) -> str:
        return f"Session<{self.thread_id}>"

    def save(self, *args, **kwargs) -> None:
        self.title = _strip_ctrl(self.title)
        self.summary = _strip_ctrl(self.summary)
        super().save(*args, **kwargs)


# =============================================================================
# Chat
# =============================================================================

class Chat(TimeStampedModel):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="chats")
    query = models.TextField(help_text="User message / prompt.")
    response = models.TextField(blank=True, default="", help_text="Assistant reply text/markdown.")
    emotion = models.JSONField(blank=True, default=dict)
    viseme = models.JSONField(blank=True, default=dict)
    meta = models.JSONField(blank=True, default=dict)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["session", "created_at"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self) -> str:
        return f"Chat<{self.pk}> thread={self.session.thread_id}"

    def save(self, *args, **kwargs) -> None:
        self.query = _strip_ctrl(self.query)
        self.response = _strip_ctrl(self.response)
        self.emotion = _clean_json(self.emotion)
        self.viseme = _clean_json(self.viseme)
        self.meta = _clean_json(self.meta)
        super().save(*args, **kwargs)


# =============================================================================
# Slides (Deck + Versions)
# =============================================================================

SLIDES_KEEP_VERSIONS = 3


class Slides(TimeStampedModel):
    session = models.OneToOneField(Session, on_delete=models.CASCADE, related_name="slides")
    version = models.PositiveIntegerField(default=0)
    title = models.CharField(max_length=255, blank=True, default="")
    summary = models.TextField(blank=True, default="")
    editorjs = models.JSONField(blank=True, default=dict)
    updated_by = models.CharField(max_length=64, blank=True, default="user")

    class Meta:
        ordering = ["-updated_at"]
        constraints = [
            # defensive: ensure version never negative (PositiveIntegerField already enforces, but explicit is nice)
            models.CheckConstraint(check=Q(version__gte=0), name="slides_version_nonnegative"),
        ]

    def __str__(self) -> str:
        return f"Slides<thread={self.session.thread_id} v={self.version}>"

    def _write_revision(self) -> "SlidesRevision":
        """Persist a version snapshot and prune older revisions."""
        rev = SlidesRevision.objects.create(
            session=self.session,
            slides=self,
            version=self.version,
            title=self.title,
            summary=self.summary,
            editorjs=self.editorjs,
            updated_by=self.updated_by.strip(),
        )

        # Keep only latest N revisions per session
        stale_ids = (
            SlidesRevision.objects.filter(session_id=self.session_id)
            .order_by("-version")
            .values_list("id", flat=True)[SLIDES_KEEP_VERSIONS:]
        )
        if stale_ids:
            SlidesRevision.objects.filter(id__in=stale_ids).delete()

        return rev

    @transaction.atomic
    def rotate_and_update(
        self, *, title: str, summary: str, editorjs: dict, updated_by: str = ""
    ) -> "SlidesRevision":
        """
        Update content and create a new revision.
        Uses SELECT ... FOR UPDATE to avoid version race conditions.
        """
        # Lock this row for the duration of the version bump
        Slides.objects.select_for_update().get(pk=self.pk)

        self.title = _strip_ctrl(title)
        self.summary = _strip_ctrl(summary)
        self.editorjs = _clean_json(editorjs)
        self.updated_by = _strip_ctrl(updated_by)

        # atomic increment
        Slides.objects.filter(pk=self.pk).update(version=F("version") + 1)
        # refresh instance's version
        self.refresh_from_db(fields=["version"])
        self.save(update_fields=["title", "summary", "editorjs", "updated_by", "updated_at"])
        return self._write_revision()

    @transaction.atomic
    def revert_to_version(self, *, target_version: int, updated_by: str = "") -> "SlidesRevision":
        """Revert to a previous revision and create a new version."""
        # Lock row to serialize version bump
        Slides.objects.select_for_update().get(pk=self.pk)

        rev = SlidesRevision.objects.filter(session=self.session, version=target_version).first()
        if not rev:
            raise ValueError(f"Version {target_version} not found for session {self.session_id}")

        self.title, self.summary, self.editorjs = rev.title, rev.summary, rev.editorjs
        self.updated_by = _strip_ctrl(updated_by) or "tool:slides_revert"

        Slides.objects.filter(pk=self.pk).update(version=F("version") + 1)
        self.refresh_from_db(fields=["version"])
        self.save(update_fields=["title", "summary", "editorjs", "updated_by", "updated_at"])
        return self._write_revision()


class SlidesRevision(TimeStampedModel):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="slide_revisions")
    slides = models.ForeignKey(Slides, on_delete=models.CASCADE, related_name="revisions")
    version = models.PositiveIntegerField(db_index=True)
    title = models.CharField(max_length=255, blank=True, default="")
    summary = models.TextField(blank=True, default="")
    editorjs = models.JSONField(blank=True, default=dict)
    updated_by = models.CharField(max_length=64, blank=True, default="")

    class Meta:
        constraints = [
            UniqueConstraint(fields=["session", "version"], name="unique_slide_version_per_session")
        ]
        ordering = ["-version", "-created_at"]

    def __str__(self) -> str:
        return f"SlidesRevision<thread={self.session.thread_id} v={self.version}>"


# =============================================================================
# Document
# =============================================================================

def document_upload_path(instance: "Document", filename: str) -> str:
    """
    Keep original basename but segregate by session thread_id.
    """
    thread_id = _slugify(getattr(instance.session, "thread_id", "thread"))
    base = os.path.basename(filename or "")
    return f"documents/{thread_id}/{base}"


class Document(TimeStampedModel):
    class IndexStatus(models.TextChoices):
        UNINDEXED = "unindexed", "Unindexed"
        INDEXED = "indexed", "Indexed"
        FAILED = "failed", "Failed"

    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=255, blank=True, default="")
    file = models.FileField(
        upload_to=document_upload_path,
        max_length=500,
        validators=[FileExtensionValidator(allowed_extensions=[e.lstrip(".") for e in sorted(SUPPORTED_EXTS)])],
    )
    content = models.TextField(blank=True, default="")
    meta = models.JSONField(blank=True, default=dict)
    index_status = models.CharField(max_length=16, choices=IndexStatus.choices, default=IndexStatus.UNINDEXED)
    search_terms = models.JSONField(blank=True, default=list)
    normalized_name = models.CharField(max_length=300, db_index=True, blank=True, default="")
    sha256 = models.CharField(max_length=64, db_index=True, blank=True, default="")

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["session", "created_at"]),
            models.Index(fields=["normalized_name"]),
            models.Index(fields=["sha256"]),
            models.Index(fields=["session", "normalized_name"]),
        ]
        constraints = [
            # Only enforce uniqueness when we actually have a hash value
            UniqueConstraint(
                fields=["session", "sha256"],
                name="unique_document_hash_per_session",
                condition=~models.Q(sha256=""),
            ),
        ]

    def __str__(self) -> str:
        return f"Document<{self.display_name} session={self.session_id}>"

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @property
    def display_name(self) -> str:
        return self.title.strip() or os.path.basename(self.file.name or "")

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return [t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t]

    @staticmethod
    def _variants(name: str) -> Set[str]:
        base = (name or "").strip()
        dotless = re.sub(r"\.[A-Za-z0-9]+$", "", base)
        return {base.lower(), _slugify(base), dotless.lower(), _slugify(dotless)}

    def refresh_search_terms(self) -> None:
        base_names = {self.title or "", os.path.basename(self.file.name or "")}
        variants: Set[str] = set().union(*(self._variants(n) | set(self._tokenize(n)) for n in base_names))
        # trim size to keep index small/deterministic
        self.normalized_name = _slugify(self.display_name)[:300]
        self.search_terms = sorted(filter(None, variants))[:100]

    def _reset_extraction_fields(self) -> None:
        self.index_status = self.IndexStatus.UNINDEXED
        self.content = ""
        self.meta = {}
        self.sha256 = ""

    # -------------------------------------------------------------------------
    # Extraction / Indexing
    # -------------------------------------------------------------------------
    def extract_and_index(self) -> None:
        """
        Extracts content and metadata from uploaded file.
        - Streams file to a temporary path (calculates sha256 on the fly)
        - Runs memory.extract.extract
        - Populates content/meta/title/hash/index_status
        """
        storage_path = self.file.name
        tmp_path = None

        try:
            # stream to temp file and hash
            with default_storage.open(storage_path, "rb") as src, NamedTemporaryFile(
                delete=False, suffix=_ext_of(storage_path) or ".bin"
            ) as tmp:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    tmp.write(chunk)
                    hasher.update(chunk)
                tmp_path = tmp.name
                computed_sha = hasher.hexdigest()

            # run extractor
            result: Dict[str, Any] = extract(tmp_path, original_filename=os.path.basename(storage_path))
            result_sha = str(result.get("sha256") or computed_sha)

            # content/meta
            self.content = _strip_ctrl(result.get("content", ""))
            meta = {k: v for k, v in result.items() if k != "content"}
            self.meta = _clean_json(meta)

            # title if missing
            if not self.title.strip():
                self.title = _strip_ctrl(result.get("file_name") or os.path.basename(storage_path))

            self.sha256 = result_sha
            self.index_status = self.IndexStatus.INDEXED

        except Exception as e:
            logger.exception("document.extract_and_index failed: %s", e)
            # best-effort preview for diagnostics
            try:
                with default_storage.open(storage_path, "rb") as f:
                    preview = f.read(4000)
                self.content = _strip_ctrl(preview.decode("utf-8", errors="replace"))
            except Exception:
                self.content = ""
            self.meta = _clean_json({"error": str(e)})
            self.index_status = self.IndexStatus.FAILED

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def save(self, *args, **kwargs) -> None:
        """
        Detect file changes and reset extraction fields if necessary.
        Always sanitize major text/JSON fields before save.
        """
        file_changed = not self.pk
        if self.pk:
            try:
                old_name = Document.objects.only("file").get(pk=self.pk).file.name
                file_changed = old_name != getattr(self.file, "name", old_name)
            except Document.DoesNotExist:
                file_changed = True

        if file_changed:
            self._reset_extraction_fields()

        self.refresh_search_terms()
        self.title = _strip_ctrl(self.title)
        self.content = _strip_ctrl(self.content)
        self.meta = _clean_json(self.meta)
        super().save(*args, **kwargs)
