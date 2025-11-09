from __future__ import annotations
import django_filters as df
from agent.models import Voice, Agent
from memory.models import Session, Chat, Slides, Knowledge

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
    # expose Agent's bot_id as 'bot_id'
    bot_id = df.UUIDFilter(field_name="agent__bot_id")
    is_active = df.BooleanFilter()

    class Meta:
        model = Session
        fields = ["thread_id", "bot_id", "is_active"]


class ChatFilter(df.FilterSet):
    # DO NOT expose 'session' in requests; use thread_id only
    thread_id = df.CharFilter(field_name="session__thread_id")

    class Meta:
        model = Chat
        fields = ["thread_id"]


class SlidesFilter(df.FilterSet):
    # DO NOT expose 'session' in requests; use thread_id only
    thread_id = df.CharFilter(field_name="session__thread_id")
    title = df.CharFilter(lookup_expr="icontains")

    class Meta:
        model = Slides
        fields = ["thread_id", "title"]


class KnowledgeFilter(df.FilterSet):
    # DO NOT expose 'agent' in requests; use bot_id only
    bot_id = df.UUIDFilter(field_name="agent__bot_id")
    mimetype = df.CharFilter()
    index_status = df.CharFilter()
    title = df.CharFilter(lookup_expr="icontains")
    file_name = df.CharFilter(lookup_expr="icontains")

    class Meta:
        model = Knowledge
        fields = ["bot_id", "mimetype", "index_status", "title", "file_name"]
