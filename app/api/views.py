# app/api/views.py
from __future__ import annotations

from django.shortcuts import render
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet
from rest_framework.exceptions import ValidationError
from drf_spectacular.utils import extend_schema, extend_schema_view, OpenApiParameter

from agent.models import Voice, Agent
from memory.models import Session, Chat, Slides, Document

from api.auth import ApiKeyAuthentication
from api.permissions import HasValidAPIKeyAndAllowedOrigin
from api.mixins import TenantScopedQuerysetMixin
from api.filters import (
    VoiceFilter, AgentFilter, SessionFilter, ChatFilter, SlidesFilter, DocumentFilter
)
from api.serializers import (
    VoiceReadSerializer, VoiceWriteSerializer,
    AgentReadSerializer, AgentWriteSerializer,
    SessionReadSerializer, SessionWriteSerializer,
    ChatReadSerializer, ChatWriteSerializer,
    SlidesReadSerializer, SlidesWriteSerializer,
    DocumentReadSerializer, DocumentWriteSerializer,
)


# ---------- Helper mixin: always return READ serializer on writes ----------
class ReturnReadOnWriteMixin:
    read_serializer_class = None
    write_serializer_class = None

    def get_serializer_class(self):
        if self.action in ("create", "update", "partial_update"):
            return self.write_serializer_class or super().get_serializer_class()
        return self.read_serializer_class or super().get_serializer_class()

    def _serialize_read(self, instance):
        serializer = self.read_serializer_class(instance, context=self.get_serializer_context())
        return serializer.data

    def perform_create(self, serializer):
        serializer.save()

    def perform_update(self, serializer):
        serializer.save()

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        instance = serializer.instance
        headers = self.get_success_headers(serializer.data)
        return Response(self._serialize_read(instance), status=status.HTTP_201_CREATED, headers=headers)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop("partial", False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        instance = serializer.instance
        return Response(self._serialize_read(instance), status=status.HTTP_200_OK)

    def partial_update(self, request, *args, **kwargs):
        kwargs["partial"] = True
        return self.update(request, *args, **kwargs)


# ------------------- Swagger shared params -------------------
USER_SCOPE_PARAMS = [
    OpenApiParameter(
        name="user_id",
        location=OpenApiParameter.QUERY,
        required=False,
        description="Admin-only: scope to user id",
    ),
]


# ------------------- Voices -------------------
@extend_schema_view(
    list=extend_schema(summary="List voices"),
    retrieve=extend_schema(summary="Retrieve a voice"),
    create=extend_schema(summary="Create a voice (preview_b64 supported)"),
    update=extend_schema(summary="Update a voice (preview_b64 supported)"),
    partial_update=extend_schema(summary="Partially update a voice"),
    destroy=extend_schema(summary="Delete a voice"),
)
@extend_schema(tags=["Voice"])
class VoiceViewSet(ReturnReadOnWriteMixin, ModelViewSet):
    queryset = Voice.objects.all().order_by("created_at", "pk")
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]
    filterset_class = VoiceFilter
    search_fields = ["name", "service", "gender"]
    ordering_fields = ["created_at", "name"]

    read_serializer_class = VoiceReadSerializer
    write_serializer_class = VoiceWriteSerializer


# ------------------- Agents -------------------
@extend_schema_view(
    list=extend_schema(summary="List agents", parameters=USER_SCOPE_PARAMS),
    retrieve=extend_schema(summary="Retrieve an agent by bot_id"),
    create=extend_schema(summary="Create an agent (avatar_b64 supported)"),
    update=extend_schema(summary="Update an agent (avatar_b64 supported)"),
    partial_update=extend_schema(summary="Partially update an agent"),
    destroy=extend_schema(summary="Delete an agent"),
)
@extend_schema(tags=["Agent"])
class AgentViewSet(ReturnReadOnWriteMixin, TenantScopedQuerysetMixin, ModelViewSet):
    lookup_field = "bot_id"
    lookup_value_regex = r"[0-9a-fA-F-]{36}"

    queryset = Agent.objects.select_related("user", "voice").all()
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]
    filterset_class = AgentFilter
    search_fields = ["name", "description", "persona", "voice__name"]
    ordering_fields = ["created_at", "updated_at", "name", "is_active"]
    user_lookup_field = "user"

    read_serializer_class = AgentReadSerializer
    write_serializer_class = AgentWriteSerializer

    def get_queryset(self):
        return self.scope_to_tenant(super().get_queryset())

    def perform_create(self, serializer):
        user = self._resolve_auth_user()
        if not user:
            raise ValidationError("API key is not linked to a valid user.")
        serializer.save(user=user)


# ------------------- Sessions -------------------
@extend_schema_view(
    list=extend_schema(summary="List sessions (filterable by bot_id)", parameters=USER_SCOPE_PARAMS),
    retrieve=extend_schema(summary="Retrieve a session by thread_id"),
    create=extend_schema(summary="Create a session (send bot_id, not agent)"),
    update=extend_schema(summary="Update a session"),
    partial_update=extend_schema(summary="Partially update a session"),
    destroy=extend_schema(summary="Delete a session"),
)
@extend_schema(tags=["Session"])
class SessionViewSet(ReturnReadOnWriteMixin, TenantScopedQuerysetMixin, ModelViewSet):
    lookup_field = "thread_id"

    queryset = Session.objects.select_related("user", "agent").all()
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]
    filterset_class = SessionFilter
    search_fields = ["thread_id", "title", "summary", "agent__name", "agent__bot_id"]
    ordering_fields = ["created_at", "updated_at"]
    user_lookup_field = "user"

    read_serializer_class = SessionReadSerializer
    write_serializer_class = SessionWriteSerializer

    def get_queryset(self):
        return self.scope_to_tenant(super().get_queryset())

    def perform_create(self, serializer):
        user = self._resolve_auth_user()
        if not user:
            raise ValidationError("API key is not linked to a valid user.")
        serializer.save(user=user)


# ------------------- Chats -------------------
@extend_schema_view(
    list=extend_schema(summary="List chat messages", parameters=USER_SCOPE_PARAMS),
    retrieve=extend_schema(summary="Retrieve a chat message"),
    create=extend_schema(summary="Create a chat message (send thread_id, not session)"),
    update=extend_schema(summary="Update a chat message"),
    partial_update=extend_schema(summary="Partially update a chat message"),
    destroy=extend_schema(summary="Delete a chat message"),
)
@extend_schema(tags=["Chat"])
class ChatViewSet(ReturnReadOnWriteMixin, TenantScopedQuerysetMixin, ModelViewSet):
    queryset = Chat.objects.select_related("session", "session__user", "session__agent").all()
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]
    filterset_class = ChatFilter
    search_fields = ["query", "response", "session__thread_id"]
    ordering_fields = ["created_at", "id"]
    user_lookup_field = "session__user"

    read_serializer_class = ChatReadSerializer
    write_serializer_class = ChatWriteSerializer

    def get_queryset(self):
        return self.scope_to_tenant(super().get_queryset())


# ------------------- Slides -------------------
@extend_schema_view(
    list=extend_schema(summary="List slides", parameters=USER_SCOPE_PARAMS),
    retrieve=extend_schema(summary="Retrieve slides by id"),
    create=extend_schema(summary="Create/attach slides (send thread_id, not session)"),
    update=extend_schema(summary="Update slides"),
    partial_update=extend_schema(summary="Partially update slides"),
    destroy=extend_schema(summary="Delete slides"),
)
@extend_schema(tags=["Slides"])
class SlidesViewSet(ReturnReadOnWriteMixin, TenantScopedQuerysetMixin, ModelViewSet):
    queryset = Slides.objects.select_related("session", "session__user", "session__agent").all()
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]
    filterset_class = SlidesFilter
    search_fields = ["title", "summary", "session__thread_id"]
    ordering_fields = ["updated_at", "created_at"]
    user_lookup_field = "session__user"

    read_serializer_class = SlidesReadSerializer
    write_serializer_class = SlidesWriteSerializer

    def get_queryset(self):
        return self.scope_to_tenant(super().get_queryset())


# ------------------- Documents -------------------
@extend_schema_view(
    list=extend_schema(summary="List documents (filterable by title/thread_id)", parameters=USER_SCOPE_PARAMS),
    retrieve=extend_schema(summary="Retrieve a document"),
    create=extend_schema(
        summary="Upload document (JSON: thread_id, title, file_b64, title)",
        description="Uploads a file to a session; extraction runs async via signals.",
    ),
    update=extend_schema(
        summary="Update document (optionally re-upload)",
        description="You may resend name+file_b64 to replace the file; you may switch session via thread_id.",
    ),
    partial_update=extend_schema(summary="Partially update document"),
    destroy=extend_schema(summary="Delete document"),
)
@extend_schema(tags=["Document"])
class DocumentViewSet(ReturnReadOnWriteMixin, TenantScopedQuerysetMixin, ModelViewSet):
    queryset = Document.objects.select_related("session", "session__user", "session__agent").all()
    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]
    filterset_class = DocumentFilter
    search_fields = ["title", "session__thread_id", "normalized_name", "sha256"]
    ordering_fields = ["created_at", "updated_at", "index_status"]
    user_lookup_field = "session__user"

    read_serializer_class = DocumentReadSerializer
    write_serializer_class = DocumentWriteSerializer

    def get_queryset(self):
        return self.scope_to_tenant(super().get_queryset())


def docView(request):
    """
    Public api doc page for REST APIs and WebSocket.
    Pulls no dynamic data; safe for DEBUG.
    """
    return render(request, "doc.html", {})


# ------------------- Simple dev test endpoint -------------------
def testView(request):
    return render(request, "test.html", {})
