# app/memory/models.py
from __future__ import annotations

import hashlib
import os
import re
import uuid
from typing import List, Set, Tuple

from django.core.files.storage import default_storage
from django.core.validators import FileExtensionValidator
from django.db import models
from django.utils import timezone

from agent.models import Agent
from user.models import User

from memory.extract import SUPPORTED_EXTS

import logging
logger = logging.getLogger(__name__)

# -----------------------------
# Utilities
# -----------------------------
def generate_thread_id() -> str:
    """Generate external-facing thread_id like 'user_<16hex>'."""
    return f"user_{uuid.uuid4().hex[:16]}"

def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "agent"

def _ext_of(name: str) -> str:
    _, ext = os.path.splitext(name or "")
    return ext.lower()

# ---- control-char scrubbers (keep \t \n \r; drop NUL and other C0) ----
_CTRL_RE = re.compile(r"[\x00\x01-\x08\x0B\x0C\x0E-\x1F\x7F]")

def _strip_ctrl(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return _CTRL_RE.sub("", s)

def _clean_json(obj):
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, str):
        return _strip_ctrl(obj)
    return obj

# -----------------------------
# Base mixins
# -----------------------------
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
    def tenant(self) -> Tuple[int, int | None]:
        return (self.user_id, self.agent_id)

# -----------------------------
# Session
# -----------------------------
class Session(TenantModel, TimeStampedModel):
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
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="chats", db_index=True)
    query = models.TextField(help_text="User message / prompt.")
    response = models.TextField(blank=True, default="", help_text="Assistant final reply text/markdown.")
    emotion = models.JSONField(blank=True, default=dict, help_text='e.g. {"name":"joy","intensity":2}')
    viseme = models.JSONField(blank=True, default=dict, help_text="Viseme timeline (+ optional audio ref).")
    meta = models.JSONField(blank=True, default=dict, help_text="Debug/runtime metadata, timings, routing info, etc.")
    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["session", "created_at"]), models.Index(fields=["created_at"])]
        verbose_name = "Chat"
        verbose_name_plural = "Chats"
    def __str__(self) -> str:
        return f"Chat<{self.pk}> thread={self.session.thread_id} agent={self.session.agent.bot_id}"

# -----------------------------
# Slides (current deck + revisions)
# -----------------------------
SLIDES_KEEP_VERSIONS = 3

class Slides(TimeStampedModel):
    session = models.OneToOneField(Session, on_delete=models.CASCADE, related_name="slides", db_index=True)
    version = models.PositiveIntegerField(default=0, help_text="Monotonic version; increments on each change.")
    title = models.CharField(max_length=255, blank=True, default="")
    summary = models.TextField(blank=True, default="")
    editorjs = models.JSONField(blank=True, default=dict, help_text="Editor.js JSON for the current deck.")
    updated_by = models.CharField(max_length=64, blank=True, default="", help_text="who updated this: ai | user | tool:<name>")
    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Slides"
        verbose_name_plural = "Slides"
    def __str__(self) -> str:
        return f"Slides<thread={self.session.thread_id} v={self.version} title={self.title!r}>"
    def _write_revision(self) -> "SlidesRevision":
        rev = SlidesRevision.objects.create(
            session=self.session,
            slides=self,
            version=self.version,
            title=self.title or "",
            summary=self.summary or "",
            editorjs=self.editorjs or {},
            updated_by=(self.updated_by or "").strip(),
        )
        ids = list(
            SlidesRevision.objects
            .filter(session_id=self.session_id)
            .order_by("-version")
            .values_list("id", flat=True)
        )
        for stale_id in ids[SLIDES_KEEP_VERSIONS:]:
            SlidesRevision.objects.filter(id=stale_id).delete()
        return rev
    def rotate_and_update(self, *, title: str, summary: str, editorjs: dict, updated_by: str = "") -> "SlidesRevision":
        self.title = _strip_ctrl(title or "")
        self.summary = _strip_ctrl(summary or "")
        self.editorjs = _clean_json(editorjs or {})
        self.updated_by = _strip_ctrl(updated_by or "")
        self.version = (self.version or 0) + 1
        self.save(update_fields=["title", "summary", "editorjs", "updated_by", "version", "updated_at"])
        return self._write_revision()
    def revert_to_version(self, *, target_version: int, updated_by: str = "") -> "SlidesRevision":
        rev = SlidesRevision.objects.filter(session=self.session, version=target_version).first()
        if not rev:
            raise ValueError(f"version {target_version} not found for session {self.session_id}")
        self.title = rev.title or ""
        self.summary = rev.summary or ""
        self.editorjs = rev.editorjs or {}
        self.updated_by = (updated_by or "tool:slides_revert").strip()
        self.version = (self.version or 0) + 1
        self.save(update_fields=["title", "summary", "editorjs", "updated_by", "version", "updated_at"])
        return self._write_revision()

class SlidesRevision(TimeStampedModel):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="slide_revisions", db_index=True)
    slides = models.ForeignKey(Slides, on_delete=models.CASCADE, related_name="slide_revisions", db_index=True)
    version = models.PositiveIntegerField(db_index=True)
    title = models.CharField(max_length=255, blank=True, default="")
    summary = models.TextField(blank=True, default="")
    editorjs = models.JSONField(blank=True, default=dict)
    updated_by = models.CharField(max_length=64, blank=True, default="")
    class Meta:
        unique_together = [("session", "version")]
        ordering = ["-version", "-created_at"]
        indexes = [models.Index(fields=["session", "version"])]
    def __str__(self) -> str:
        return f"SlidesRevision<thread={self.session.thread_id} v={self.version}>"

# -----------------------------
# Knowledge (uploads per agent)
# -----------------------------
def knowledge_upload_to(instance: "Knowledge", filename: str) -> str:
    """
    Build storage path AND remember the client-supplied filename on the instance,
    so we never rely on admin input for file_name.
    """
    # Remember client name for later persistence
    try:
        instance._uploaded_client_name = os.path.basename(filename or "").strip() or "file"
    except Exception:
        instance._uploaded_client_name = "file"

    agent_slug = _slugify(getattr(instance.agent, "name", "") or "agent")
    stem, ext = os.path.splitext(os.path.basename(filename or "file"))
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", (stem or "file")).strip("._")
    ext = (ext or _ext_of(filename)).lower()
    base = f"knowledge/{agent_slug}/{stem}{ext}"
    try:
        return default_storage.get_available_name(base)
    except Exception:
        return f"knowledge/{agent_slug}/{stem}__dup{uuid.uuid4().hex[:6]}{ext}"

class Knowledge(TenantModel, TimeStampedModel):
    key = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    title = models.CharField(max_length=255, blank=True, default="", help_text="Optional human-readable title.")
    file_name = models.CharField(max_length=255, blank=True, editable=False)
    file = models.FileField(
        upload_to=knowledge_upload_to,
        max_length=500,
        validators=[FileExtensionValidator(allowed_extensions=[e.lstrip(".") for e in sorted(SUPPORTED_EXTS)])],
    )

    size_bytes = models.BigIntegerField(default=0)
    mimetype = models.CharField(max_length=100, blank=True, default="")
    sha256 = models.CharField(max_length=64, blank=True, default="")
    ext = models.CharField(max_length=16, blank=True, default="")
    pages = models.PositiveIntegerField(null=True, blank=True)
    rows = models.PositiveIntegerField(null=True, blank=True)
    cols = models.PositiveIntegerField(null=True, blank=True)

    excerpt = models.TextField(blank=True, default="", help_text="First ~4k chars of extracted text for quick search.")
    meta = models.JSONField(blank=True, default=dict)

    class IndexStatus(models.TextChoices):
        UNINDEXED = "unindexed", "Unindexed"
        INDEXED = "indexed", "Indexed"
        FAILED = "failed", "Failed"

    index_status = models.CharField(max_length=16, choices=IndexStatus.choices, default=IndexStatus.UNINDEXED)
    index_meta = models.JSONField(blank=True, default=dict)

    normalized_name = models.CharField(max_length=300, db_index=True, blank=True, default="")
    search_terms = models.JSONField(blank=True, default=list, help_text="Lowercased variants, tokens, slugs, aliases.")

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["agent", "created_at"]),
            models.Index(fields=["mimetype"]),
            models.Index(fields=["normalized_name"]),
        ]
        verbose_name = "Knowledge"
        verbose_name_plural = "Knowledge"

    def __str__(self) -> str:
        return f"Knowledge<{self.display_name} • {self.key}>"

    @property
    def display_name(self) -> str:
        return (self.title or "").strip() or (self.file_name or "").strip() or self.file.name

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        s = (s or "").strip().lower()
        toks = re.split(r"[^a-z0-9]+", s)
        return [t for t in toks if t]

    @staticmethod
    def _variants(name: str) -> Set[str]:
        name = (name or "").strip()
        low = name.lower()
        slug = _slugify(name)
        dotless = re.sub(r"\.[A-Za-z0-9]+$", "", name)
        return {low, slug, dotless.lower(), _slugify(dotless)}

    def refresh_search_terms(self) -> None:
        base_names = {self.title or "", self.file_name or "", os.path.basename(self.file.name or "")}
        variants: Set[str] = set()
        for n in base_names:
            variants |= self._variants(n)
            variants |= set(self._tokenize(n))
        self.normalized_name = _slugify(self.display_name)[:300]
        self.search_terms = sorted({v for v in variants if v})[:80]

    # ---------- helpers ----------
    def _mark_unindexed_due_to_new_file(self):
        """Reset indexing fields when file is swapped/re-uploaded."""
        self.index_status = self.IndexStatus.UNINDEXED
        self.sha256 = ""
        self.mimetype = ""
        self.ext = ""
        self.pages = None
        self.rows = None
        self.cols = None
        self.excerpt = ""
        self.meta = {}
        self.index_meta = {}

    # ---------- Extraction / indexing ----------
    def extract_and_index(self) -> None:
        """
        Compute sha256/size/mime and extract rich content via memory.extract.extract_rich.
        Always scrubs control chars to avoid DB NUL errors.
        """
        from tempfile import NamedTemporaryFile
        from memory.extract import extract_rich, detect_mime  # ← rich extractor

        storage_path = self.file.name
        try:
            with default_storage.open(storage_path, "rb") as f:
                blob = f.read()
        except Exception as e:
            self.index_status = self.IndexStatus.FAILED
            self.index_meta = _clean_json({"error": "storage_open_failed", "path": storage_path, "detail": str(e)})
            return

        self.size_bytes = len(blob)

        # Use the captured client file name; fall back to storage basename
        if not self.file_name:
            self.file_name = getattr(self, "_uploaded_client_name", "") or os.path.basename(storage_path)

        self.ext = _ext_of(self.file_name or storage_path)

        tmp_path = None
        try:
            # Materialize to temp path for libraries/CLIs that need it
            with NamedTemporaryFile(delete=False, suffix=self.ext or ".bin") as tmp:
                tmp.write(blob)
                tmp_path = tmp.name

            if not self.mimetype:
                try:
                    self.mimetype = detect_mime(tmp_path)
                except Exception:
                    import mimetypes
                    self.mimetype = mimetypes.guess_type(self.file_name or storage_path)[0] or ""

            self.sha256 = hashlib.sha256(blob).hexdigest()

            # ---- rich extraction ----
            result = extract_rich(tmp_path)  # ExtractionResult
            text = result.text or ""

            # excerpt
            self.excerpt = _strip_ctrl(text)[:4000]

            # pages/rows/cols mapping from structure
            pages = rows = cols = None
            kind = (result.structure or {}).get("kind")
            if kind == "pdf":
                pages = (result.structure or {}).get("page_count")
            elif kind == "csv":
                rows = (result.structure or {}).get("rows")
                cols = (result.structure or {}).get("cols")
            elif kind == "excel":
                sheets = (result.structure or {}).get("sheets", []) or []
                rows = sum(int(s.get("rows", 0) or 0) for s in sheets) if sheets else None
                try:
                    cols = max(int(s.get("cols", 0) or 0) for s in sheets) if sheets else None
                except ValueError:
                    cols = None
            elif kind == "json":
                shape = (result.structure or {}).get("shape") or {}
                if shape.get("type") == "array":
                    rows = shape.get("length")
                if "columns" in shape:
                    try:
                        cols = len(shape.get("columns") or [])
                    except Exception:
                        cols = None
            # docx kind could map paragraphs/tables via meta only; leave pages/rows/cols as None

            self.pages = pages
            self.rows = rows
            self.cols = cols

            # meta: store everything useful (safe JSON)
            self.meta = _clean_json({
                "mime": result.mime or self.mimetype,
                "ext": result.ext or self.ext,
                "encoding": result.encoding,
                "stats": {
                    "size_bytes": result.stats.size_bytes,
                    "num_chars": result.stats.num_chars,
                    "num_words": result.stats.num_words,
                    "num_lines": result.stats.num_lines,
                },
                "structure": result.structure or {},
                "file_name": self.file_name,
            })

            self.index_status = self.IndexStatus.INDEXED
            self.index_meta = _clean_json({"ok": True})

        except Exception as e:
            # Best-effort fallback (preview)
            try:
                preview = blob[:4000]
                try:
                    txt = preview.decode("utf-8", errors="replace")
                except Exception:
                    txt = preview.decode("latin-1", errors="replace")
                self.excerpt = _strip_ctrl(txt)
                self.meta = _clean_json({"mime": self.mimetype, "ext": self.ext, "fallback": True})
                self.index_status = self.IndexStatus.FAILED  # mark failed (extraction didn’t complete)
                self.index_meta = _clean_json({"error": str(e)})
            except Exception as e2:
                self.index_status = self.IndexStatus.FAILED
                self.index_meta = _clean_json({"error": f"extract_failed:{e2!s}"})
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # Save hook: capture client name, detect file changes, and scrub fields.
    def save(self, *args, **kwargs):
        # Detect file change by comparing current DB value
        file_changed = False
        if self.pk:
            try:
                old = Knowledge.objects.only("file", "size_bytes", "sha256").get(pk=self.pk)
                file_changed = bool(old.file.name != getattr(self.file, "name", old.file.name))
            except Knowledge.DoesNotExist:
                file_changed = True
        else:
            file_changed = True  # new row

        # If file changed, auto-fill file_name from upload hook (or storage basename)
        if file_changed:
            if not self.file_name:
                self.file_name = getattr(self, "_uploaded_client_name", "") or os.path.basename(getattr(self.file, "name", "") or "")
            self._mark_unindexed_due_to_new_file()

        self.refresh_search_terms()

        # scrub
        self.title = _strip_ctrl(self.title or "")
        self.file_name = _strip_ctrl(self.file_name or "")
        self.mimetype = _strip_ctrl(self.mimetype or "")
        self.ext = _strip_ctrl(self.ext or "")
        self.excerpt = _strip_ctrl(self.excerpt or "")
        self.meta = _clean_json(self.meta or {})
        self.index_meta = _clean_json(self.index_meta or {})

        super().save(*args, **kwargs)