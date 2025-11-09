# app/memory/tasks.py
from __future__ import annotations

from celery import shared_task
from django.db import transaction

from .models import Knowledge, _clean_json, _strip_ctrl

@shared_task(name="memory.index_knowledge")
def index_knowledge(knowledge_id: int) -> dict:
    try:
        obj = Knowledge.objects.get(id=knowledge_id)
    except Knowledge.DoesNotExist:
        return {"status": "gone", "id": knowledge_id}

    obj.extract_and_index()
    obj.refresh_search_terms()

    # extra scrub before partial update
    obj.title = _strip_ctrl(obj.title or "")
    obj.original_name = _strip_ctrl(obj.original_name or "")
    obj.mimetype = _strip_ctrl(obj.mimetype or "")
    obj.ext = _strip_ctrl(obj.ext or "")
    obj.excerpt = _strip_ctrl(obj.excerpt or "")
    obj.meta = _clean_json(obj.meta or {})
    obj.index_meta = _clean_json(obj.index_meta or {})

    with transaction.atomic():
        obj.save(update_fields=[
            "size_bytes","mimetype","sha256","ext",
            "pages","rows","cols","excerpt","meta",
            "index_status","index_meta","normalized_name","search_terms","updated_at"
        ])
    return {"status": "ok", "id": obj.id, "key": str(obj.key), "index_status": obj.index_status}
