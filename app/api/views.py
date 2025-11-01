# api/views.py
from __future__ import annotations
from django.shortcuts import render, get_object_or_404
from rest_framework import filters as drf_filters, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet
from drf_spectacular.utils import extend_schema, extend_schema_view, OpenApiParameter

from agent.models import Voice, Agent
from memory.models import Session, Chat, Slides

from api.auth import ApiKeyAuthentication
from api.permissions import HasValidAPIKeyAndAllowedOrigin
from api.mixins import TenantScopedQuerysetMixin
from api.filters import VoiceFilter, AgentFilter, SessionFilter, ChatFilter, SlidesFilter
from api.serializers import (
    VoiceReadSerializer, VoiceWriteSerializer,
    AgentReadSerializer, AgentWriteSerializer,
    SessionReadSerializer, SessionWriteSerializer,
    ChatReadSerializer, ChatWriteSerializer,
    SlidesReadSerializer, SlidesWriteSerializer,
)

# ------------------- Swagger shared params -------------------
USER_SCOPE_PARAMS = [
    OpenApiParameter(name="user_id", location=OpenApiParameter.QUERY, required=False, description="Admin-only: scope to user id"),
    OpenApiParameter(name="username", location=OpenApiParameter.QUERY, required=False, description="Admin-only: scope to username"),
    OpenApiParameter(name="email", location=OpenApiParameter.QUERY, required=False, description="Admin-only: scope to email"),
]

# ------------------- Voices -------------------
@extend_schema_view(
    list=extend_schema(summary="List voices"),
    retrieve=extend_schema(summary="Retrieve a voice"),
    create=extend_schema(summary="Create a voice"),
    update=extend_schema(summary="Update a voice"),
    partial_update=extend_schema(summary="Partially update a voice"),
    destroy=extend_schema(summary="Delete a voice"),
)
@extend_schema(tags=["Voice"])
class VoiceViewSet(ModelViewSet):
    queryset = Voice.objects.all().order_by("created_at", "pk")
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]
    filterset_class = VoiceFilter
    search_fields = ["name", "service", "gender"]
    ordering_fields = ["created_at", "name"]

    def get_serializer_class(self):
        return VoiceWriteSerializer if self.action in ("create", "update", "partial_update") else VoiceReadSerializer

# ------------------- Agents -------------------
@extend_schema_view(
    list=extend_schema(summary="List agents", parameters=USER_SCOPE_PARAMS),
    retrieve=extend_schema(summary="Retrieve an agent by bot_id"),
    create=extend_schema(summary="Create an agent"),
    update=extend_schema(summary="Update an agent"),
    partial_update=extend_schema(summary="Partially update an agent"),
    destroy=extend_schema(summary="Delete an agent"),
)
@extend_schema(tags=["Agent"])
class AgentViewSet(TenantScopedQuerysetMixin, ModelViewSet):
    # lookup by bot_id as requested
    lookup_field = "bot_id"
    lookup_value_regex = r"[0-9a-fA-F-]{36}"

    queryset = Agent.objects.select_related("user", "voice").all()
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]
    filterset_class = AgentFilter
    search_fields = ["name", "persona", "user__username", "user__email", "voice__name"]
    ordering_fields = ["created_at", "name", "is_active"]

    user_lookup_field = "user"

    def get_queryset(self):
        qs = super().get_queryset()
        qs = self.scope_to_tenant(qs)  # default: authenticated user
        return qs

    def get_serializer_class(self):
        return AgentWriteSerializer if self.action in ("create", "update", "partial_update") else AgentReadSerializer

# ------------------- Sessions -------------------
@extend_schema_view(
    list=extend_schema(summary="List sessions", parameters=USER_SCOPE_PARAMS),
    retrieve=extend_schema(summary="Retrieve a session by thread_id"),
    create=extend_schema(summary="Create a session"),
    update=extend_schema(summary="Update a session"),
    partial_update=extend_schema(summary="Partially update a session"),
    destroy=extend_schema(summary="Delete a session"),
)
@extend_schema(tags=["Session"])
class SessionViewSet(TenantScopedQuerysetMixin, ModelViewSet):
    lookup_field = "thread_id"

    queryset = Session.objects.select_related("user", "agent").all()
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]
    filterset_class = SessionFilter
    search_fields = ["thread_id", "title", "summary", "agent__name", "agent__bot_id"]
    ordering_fields = ["created_at", "updated_at", "title"]
    user_lookup_field = "user"

    def get_queryset(self):
        return self.scope_to_tenant(super().get_queryset())

    def get_serializer_class(self):
        return SessionWriteSerializer if self.action in ("create", "update", "partial_update") else SessionReadSerializer

# ------------------- Chats -------------------
@extend_schema_view(
    list=extend_schema(summary="List chat messages", parameters=USER_SCOPE_PARAMS),
    retrieve=extend_schema(summary="Retrieve a chat message"),
    create=extend_schema(summary="Create a chat message"),
    update=extend_schema(summary="Update a chat message"),
    partial_update=extend_schema(summary="Partially update a chat message"),
    destroy=extend_schema(summary="Delete a chat message"),
)
@extend_schema(tags=["Chat"])
class ChatViewSet(TenantScopedQuerysetMixin, ModelViewSet):
    queryset = Chat.objects.select_related("session", "session__user", "session__agent").all()
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]
    filterset_class = ChatFilter
    search_fields = ["query", "response", "session__thread_id"]
    ordering_fields = ["created_at", "id"]
    user_lookup_field = "session__user"

    def get_queryset(self):
        return self.scope_to_tenant(super().get_queryset())

    def get_serializer_class(self):
        return ChatWriteSerializer if self.action in ("create", "update", "partial_update") else ChatReadSerializer

# ------------------- Slides -------------------
@extend_schema_view(
    list=extend_schema(summary="List slides", parameters=USER_SCOPE_PARAMS),
    retrieve=extend_schema(summary="Retrieve slides by id"),
    create=extend_schema(summary="Create/attach slides to a session"),
    update=extend_schema(summary="Update slides"),
    partial_update=extend_schema(summary="Partially update slides"),
    destroy=extend_schema(summary="Delete slides"),
)
@extend_schema(tags=["Slides"])
class SlidesViewSet(TenantScopedQuerysetMixin, ModelViewSet):
    queryset = Slides.objects.select_related("session", "session__user", "session__agent").all()
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]
    filterset_class = SlidesFilter
    search_fields = ["title", "summary", "session__thread_id"]
    ordering_fields = ["updated_at", "created_at"]
    user_lookup_field = "session__user"

    def get_queryset(self):
        return self.scope_to_tenant(super().get_queryset())

    def get_serializer_class(self):
        return SlidesWriteSerializer if self.action in ("create", "update", "partial_update") else SlidesReadSerializer

# ------------------- Simple dev test endpoint -------------------
def testView(request):
    return render(request, "test.html", {})
