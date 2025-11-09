# aicoach/celery.py
from __future__ import annotations
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aicoach.settings")

app = Celery("aicoach")

# Load all CELERY_* keys from Django settings
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks
app.autodiscover_tasks()
