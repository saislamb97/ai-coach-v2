# api/filters.py
from __future__ import annotations
import django_filters as df

from agent.models import Voice, Agent
from memory.models import Session, Chat, Slides, Document


class VoiceFilter(df.FilterSet):
    name = df.CharFilter(lookup_expr="icontains")
    service = df.CharFilter()
    gender = df.CharFilter()

    class Meta:
        model = Voice
        fields = ["name", "service", "gender"]


class AgentFilter(df.FilterSet):
    bot_id = df.UUIDFilter()
    is_active = df.BooleanFilter()
    name = df.CharFilter(lookup_expr="icontains")

    class Meta:
        model = Agent
        fields = ["bot_id", "is_active", "name"]


class SessionFilter(df.FilterSet):
    thread_id = df.CharFilter()
    bot_id = df.UUIDFilter(field_name="agent__bot_id")
    is_active = df.BooleanFilter()

    class Meta:
        model = Session
        fields = ["thread_id", "bot_id", "is_active"]


class ChatFilter(df.FilterSet):
    thread_id = df.CharFilter(field_name="session__thread_id")

    class Meta:
        model = Chat
        fields = ["thread_id"]


class SlidesFilter(df.FilterSet):
    thread_id = df.CharFilter(field_name="session__thread_id")
    title = df.CharFilter(lookup_expr="icontains")

    class Meta:
        model = Slides
        fields = ["thread_id", "title"]


class DocumentFilter(df.FilterSet):
    thread_id = df.CharFilter(field_name="session__thread_id")
    title = df.CharFilter(lookup_expr="icontains")
    index_status = df.ChoiceFilter(choices=Document.IndexStatus.choices)

    class Meta:
        model = Document
        fields = ["thread_id", "title", "index_status"]