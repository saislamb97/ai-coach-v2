from __future__ import annotations

from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    VoiceViewSet,
    AgentViewSet,
    SessionViewSet,
    ChatViewSet,
    SlidesViewSet,
    KnowledgeViewSet,
    docView,
    testView,
)

router = DefaultRouter()
router.register(r"voices", VoiceViewSet, basename="voice")
router.register(r"agents", AgentViewSet, basename="agent")
router.register(r"sessions", SessionViewSet, basename="session")
router.register(r"chats", ChatViewSet, basename="chat")
router.register(r"slides", SlidesViewSet, basename="slides")
router.register(r"knowledge", KnowledgeViewSet, basename="knowledge")  # ← NEW

app_name = "api"

urlpatterns = [
    path("", include(router.urls)),
    path("test/", testView, name="test"),
    path("doc/", docView, name="doc"),
]