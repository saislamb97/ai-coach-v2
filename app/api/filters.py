# api/filters.py
from __future__ import annotations
import django_filters as df
from agent.models import Voice, Agent
from memory.models import Session, Chat, Slides

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

    class Meta:
        model = Agent
        fields = ["bot_id", "is_active"]

class SessionFilter(df.FilterSet):
    thread_id = df.CharFilter()
    # renamed: allow filtering by the Agent's bot_id using 'bot_id'
    bot_id = df.UUIDFilter(field_name="agent__bot_id")
    is_active = df.BooleanFilter()

    class Meta:
        model = Session
        fields = ["thread_id", "bot_id", "is_active"]

class ChatFilter(df.FilterSet):
    session = df.NumberFilter()
    thread_id = df.CharFilter(field_name="session__thread_id")

    class Meta:
        model = Chat
        fields = ["session", "thread_id"]

class SlidesFilter(df.FilterSet):
    session = df.NumberFilter()
    thread_id = df.CharFilter(field_name="session__thread_id")
    title = df.CharFilter(lookup_expr="icontains")

    class Meta:
        model = Slides
        fields = ["session", "thread_id", "title"]
