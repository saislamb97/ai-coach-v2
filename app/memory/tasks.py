# app/memory/tasks.py
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional, Dict, Any

from celery import shared_task
from django.core.cache import cache
from django.db import transaction
from django.db.utils import IntegrityError, OperationalError, DatabaseError
from django.db.models import Count
from django.utils import timezone

from .models import Document, Session, _clean_json, _strip_ctrl

log = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _mark_duplicate(obj: Document, *, winner_id: Optional[int], sha: Optional[str]) -> Dict[str, Any]:
    """
    Persist the document as a duplicate of winner_id.
    Clear sha256 to respect the partial unique constraint on (session, sha256).
    """
    meta = dict(obj.meta or {})
    meta["duplicate_of_id"] = winner_id
    if sha:
        meta["duplicate_sha256"] = sha

    obj.index_status = Document.IndexStatus.DUPLICATE
    obj.meta = _clean_json(meta)
    obj.sha256 = ""  # important: keep the partial-unique constraint satisfied

    with transaction.atomic():
        obj.save(
            update_fields=[
                "title",
                "content",
                "meta",
                "index_status",
                "normalized_name",
                "search_terms",
                "updated_at",
            ]
        )

    log.info(
        "[memory.index_document] duplicate id=%s -> existing_id=%s",
        obj.id,
        winner_id,
    )
    return {"status": str(Document.IndexStatus.DUPLICATE), "id": obj.id, "duplicate_of": winner_id}


# -------------------------------------------------------------------
# Document Indexing
# -------------------------------------------------------------------
@shared_task(bind=True, max_retries=2, default_retry_delay=5)
def index_document(self, document_id: int) -> dict:
    """
    Full extraction + indexing for a Document, with:
      - Early duplicate detection (same (session, sha256))
      - Race-safe fallback (IntegrityError -> mark DUPLICATE)
      - Clean handling of missing files / gone objects
    Returns a small dict suitable for logs/telemetry.
    """
    try:
        obj = Document.objects.get(id=document_id)
    except Document.DoesNotExist:
        log.info("[memory.index_document] document %s no longer exists", document_id)
        return {"status": "gone", "id": document_id}

    path = getattr(obj.file, "name", "")
    log.info("[memory.index_document] start id=%s path=%s", obj.id, path)

    # If file field is empty or storage missing, don't blow up Celery with retries
    if not path:
        log.warning("[memory.index_document] no file for id=%s", obj.id)
        with transaction.atomic():
            obj.index_status = Document.IndexStatus.FAILED
            meta = dict(obj.meta or {})
            meta["error"] = "no_file"
            obj.meta = _clean_json(meta)
            obj.save(update_fields=["index_status", "meta", "updated_at"])
        return {"status": "error", "id": obj.id, "error": "no_file"}

    try:
        # 1) Extract (fills title/content/meta and sets obj.sha256 on success)
        obj.extract_and_index()
        obj.refresh_search_terms()

        # Scrub text/JSON
        obj.title = _strip_ctrl(obj.title or "")
        obj.content = _strip_ctrl(obj.content or "")
        obj.meta = _clean_json(obj.meta or {})

        # If extraction failed, index_status will be FAILED and sha may be empty; persist and return.
        if obj.index_status == Document.IndexStatus.FAILED:
            with transaction.atomic():
                obj.save(
                    update_fields=[
                        "title",
                        "content",
                        "meta",
                        "index_status",
                        "normalized_name",
                        "search_terms",
                        "updated_at",
                    ]
                )
            log.info("[memory.index_document] completed id=%s status=%s", obj.id, obj.index_status)
            return {"status": str(obj.index_status), "id": obj.id, "session_id": obj.session_id}

        # 2) Early duplicate check (same session & sha, different id)
        if obj.sha256:
            dup = (
                Document.objects.filter(session_id=obj.session_id, sha256=obj.sha256)
                .exclude(id=obj.id)
                .only("id")
                .first()
            )
            if dup:
                return _mark_duplicate(obj, winner_id=dup.id, sha=obj.sha256)

        # 3) Save with sha256 (normal path). Race-safe against other workers.
        try:
            with transaction.atomic():
                obj.save(
                    update_fields=[
                        "title",
                        "content",
                        "meta",
                        "index_status",
                        "normalized_name",
                        "search_terms",
                        "sha256",
                        "updated_at",
                    ]
                )
        except IntegrityError:
            # Another worker won the race with the same (session, sha256)
            winner = (
                Document.objects.filter(session_id=obj.session_id, sha256=obj.sha256)
                .exclude(id=obj.id)
                .only("id")
                .first()
            )
            return _mark_duplicate(obj, winner_id=getattr(winner, "id", None), sha=obj.sha256)

        log.info("[memory.index_document] completed id=%s status=%s", obj.id, obj.index_status)
        return {"status": str(obj.index_status), "id": obj.id, "session_id": obj.session_id}

    except (OperationalError, DatabaseError) as e:
        # Transient DB issues: retry a couple of times
        log.exception("[memory.index_document] db error id=%s err=%s", obj.id, e)
        raise self.retry(exc=e)
    except Exception as e:
        # Non-transient extraction/storage issues: record and return error
        log.exception("[memory.index_document] failed id=%s err=%s", obj.id, e)
        return {"status": "error", "id": obj.id, "error": str(e)}


# -------------------------------------------------------------------
# Janitor: prune idle Sessions (default 30 minutes)
# -------------------------------------------------------------------
LOCK_KEY_PRUNE = "memory:prune_idle_sessions:lock"
COOLDOWN_KEY_PRUNE = "memory:prune_idle_sessions:recent"
LOCK_EXPIRE_S_PRUNE = 30
COOLDOWN_SECONDS_PRUNE = 30


def _acquire_lock(key: str, ttl: int) -> bool:
    try:
        return cache.add(key, True, timeout=ttl)  # atomic in most backends
    except Exception:
        log.exception("[memory.prune_idle_sessions] lock acquire error; proceeding without lock")
        return True


def _cooldown_ok(key: str, cooldown_s: int) -> bool:
    try:
        if cache.get(key):
            return False
        cache.set(key, True, timeout=cooldown_s)
        return True
    except Exception:
        # If cache is down, don't block the janitor entirely
        return True


@shared_task(bind=True, max_retries=3, default_retry_delay=10, ignore_result=True)
def prune_idle_sessions(self, minutes: int = 30) -> None:
    """
    Delete sessions that have *no chats at all* and are older than `minutes`.
    If you want 'inactive regardless of having chats', switch to updated_at__lt=cutoff.
    """
    if not _cooldown_ok(COOLDOWN_KEY_PRUNE, COOLDOWN_SECONDS_PRUNE):
        log.info("[memory.prune_idle_sessions] skipped: recent run")
        return
    if not _acquire_lock(LOCK_KEY_PRUNE, LOCK_EXPIRE_S_PRUNE):
        log.info("[memory.prune_idle_sessions] skipped: already running")
        return

    try:
        cutoff = timezone.now() - timedelta(minutes=minutes)
        qs = (
            Session.objects.filter(created_at__lt=cutoff)
            .annotate(num_chats=Count("chats"))
            .filter(num_chats=0)
        )
        deleted, _ = qs.delete()
        log.info("[memory.prune_idle_sessions] deleted=%s older_than=%smin", deleted, minutes)
    except Exception as e:
        log.exception("[memory.prune_idle_sessions] failed: %s", e)
    finally:
        cache.delete(LOCK_KEY_PRUNE)
