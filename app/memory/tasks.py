# app/memory/tasks.py
from __future__ import annotations
import logging
from datetime import timedelta

from celery import shared_task
from django.db import transaction
from django.utils import timezone
from django.core.cache import cache
from django.db.models import Count

from .models import Document, Session, _clean_json, _strip_ctrl

log = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Document Indexing
# -------------------------------------------------------------------
@shared_task
def index_document(document_id: int) -> dict:
    """
    Run full extraction + indexing for a given Document.
    Handles missing file paths, retries, and robust logging.
    """
    try:
        obj = Document.objects.get(id=document_id)
    except Document.DoesNotExist:
        log.warning("[memory.index_document] document %s no longer exists", document_id)
        return {"status": "gone", "id": document_id}

    log.info("[memory.index_document] start id=%s path=%s", obj.id, getattr(obj.file, "name", ""))

    try:
        obj.extract_and_index()
        obj.refresh_search_terms()

        # Defensive scrubbing before partial update
        obj.title = _strip_ctrl(obj.title or "")
        obj.content = _strip_ctrl(obj.content or "")
        obj.meta = _clean_json(obj.meta or {})

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

        status = obj.index_status
        log.info("[memory.index_document] completed id=%s status=%s", obj.id, status)
        return {"status": status, "id": obj.id, "session_id": obj.session_id}

    except Exception as e:
        log.exception("[memory.index_document] failed id=%s err=%s", document_id, e)
        return {"status": "error", "id": document_id, "error": str(e)}


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
