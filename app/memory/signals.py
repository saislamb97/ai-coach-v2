# app/memory/signals.py
from __future__ import annotations
import logging
from django.db import transaction
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Document
from .tasks import index_document

log = logging.getLogger(__name__)


@receiver(post_save, sender=Document)
def enqueue_index_on_save(sender, instance: Document, created: bool, **kwargs):
    """
    Enqueue document indexing whenever a Document is saved.

    - Runs for both create and update.
    - Skips if the document is already indexed or has no file.
    - Defers the Celery task until after DB commit (safe for admin/uploads).
    """

    try:
        # Skip if no file exists or Celery shouldn’t re-index
        if not instance.file:
            log.warning("document.enqueue: skipped (no file) id=%s", instance.id)
            return

        should_enqueue = created or instance.index_status == Document.IndexStatus.UNINDEXED
        if not should_enqueue:
            log.debug("document.enqueue: skipped id=%s (index_status=%s)", instance.id, instance.index_status)
            return

        def _enqueue():
            log.info(
                "document.enqueue: triggering index_document for id=%s session=%s path=%s",
                instance.id,
                instance.session_id,
                getattr(instance.file, "name", ""),
            )
            try:
                index_document.delay(instance.id)
            except Exception as e:
                log.exception("document.enqueue: Celery dispatch failed for id=%s: %s", instance.id, e)

        # Run only after transaction commit (safe for admin + file upload)
        transaction.on_commit(_enqueue)

    except Exception:
        log.exception(
            "document.enqueue: failed for id=%s session=%s",
            getattr(instance, "id", None),
            getattr(instance, "session_id", None),
        )
