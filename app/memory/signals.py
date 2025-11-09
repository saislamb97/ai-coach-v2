# app/memory/signals.py (unchanged logic; shown for clarity)
from __future__ import annotations
import logging

from django.db import transaction
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Knowledge

log = logging.getLogger(__name__)

@receiver(post_save, sender=Knowledge)
def enqueue_index_on_upload(sender, instance: Knowledge, created: bool, **kwargs):
    """
    Enqueue background extraction/indexing when a Knowledge row is created
    or explicitly marked UNINDEXED again (e.g., file replaced).
    """
    try:
        should_enqueue = created or instance.index_status == Knowledge.IndexStatus.UNINDEXED
        if not should_enqueue:
            return

        from .tasks import index_knowledge

        def _enqueue():
            log.info(
                "knowledge.enqueue: index_knowledge id=%s key=%s path=%s",
                instance.id, instance.key, getattr(instance.file, "name", "")
            )
            index_knowledge.delay(instance.id)

        transaction.on_commit(_enqueue)
    except Exception:
        log.exception("knowledge.enqueue: failed for id=%s key=%s", getattr(instance, "id", None), getattr(instance, "key", None))
