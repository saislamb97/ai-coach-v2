from __future__ import annotations

from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema, extend_schema_view

from django.shortcuts import render
from agent.models import Voice, Agent
from memory.models import Session, Chat, Slides
from .serializers import (
    VoiceSerializer,
    AgentSerializer,
    SessionSerializer,
    ChatSerializer,
    SlidesSerializer,
)

from api.auth import ApiKeyAuthentication
from api.permissions import HasValidAPIKeyAndAllowedOrigin


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
    serializer_class = VoiceSerializer

    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]


@extend_schema_view(
    list=extend_schema(summary="List agents"),
    retrieve=extend_schema(summary="Retrieve an agent"),
    create=extend_schema(summary="Create an agent"),
    update=extend_schema(summary="Update an agent"),
    partial_update=extend_schema(summary="Partially update an agent"),
    destroy=extend_schema(summary="Delete an agent"),
)
@extend_schema(tags=["Agent"])
class AgentViewSet(ModelViewSet):
    queryset = Agent.objects.all().order_by("-created_at")
    serializer_class = AgentSerializer

    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]


@extend_schema_view(
    list=extend_schema(summary="List sessions"),
    retrieve=extend_schema(summary="Retrieve a session"),
    create=extend_schema(summary="Create a session"),
    update=extend_schema(summary="Update a session"),
    partial_update=extend_schema(summary="Partially update a session"),
    destroy=extend_schema(summary="Delete a session"),
)
@extend_schema(tags=["Session"])
class SessionViewSet(ModelViewSet):
    queryset = Session.objects.select_related("user", "agent").order_by("-created_at")
    serializer_class = SessionSerializer

    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]


@extend_schema_view(
    list=extend_schema(summary="List chat messages"),
    retrieve=extend_schema(summary="Retrieve a chat message"),
    create=extend_schema(summary="Create a chat message"),
    update=extend_schema(summary="Update a chat message"),
    partial_update=extend_schema(summary="Partially update a chat message"),
    destroy=extend_schema(summary="Delete a chat message"),
)
@extend_schema(tags=["Chat"])
class ChatViewSet(ModelViewSet):
    queryset = (
        Chat.objects
        .select_related("session", "session__user", "session__agent")
        .order_by("-created_at")
    )
    serializer_class = ChatSerializer

    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]


@extend_schema_view(
    list=extend_schema(summary="List slides"),
    retrieve=extend_schema(summary="Retrieve slides for a session"),
    create=extend_schema(summary="Create or attach slides to a session"),
    update=extend_schema(summary="Update slides"),
    partial_update=extend_schema(summary="Partially update slides"),
    destroy=extend_schema(summary="Delete slides for a session"),
)
@extend_schema(tags=["Slides"])
class SlidesViewSet(ModelViewSet):
    queryset = (
        Slides.objects
        .select_related("session", "session__user", "session__agent")
        .order_by("-updated_at")
    )
    serializer_class = SlidesSerializer

    authentication_classes = [ApiKeyAuthentication]
    permission_classes = [IsAuthenticated, HasValidAPIKeyAndAllowedOrigin]


# Simple dev test endpoint
def testView(request):
    return render(request, "test.html", {})
