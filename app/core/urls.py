# core/urls.py
from django.urls import path
from core.views import apiDoc

app_name = "core"

urlpatterns = [
    path("", apiDoc, name="api-doc"),
]
