# api/serializers.py
from __future__ import annotations
from typing import Any
from django.contrib.auth import get_user_model
from rest_framework import serializers
from agent.models import Voice, Agent
from memory.models import Session, Chat, Slides

User = get_user_model()

# ----- Voices -----
class VoiceReadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Voice
        fields = "__all__"

class VoiceWriteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Voice
        fields = ["name", "service", "voice_id", "gender", "preview"]
        extra_kwargs = {
            "service": {"required": False},
            "voice_id": {"required": False},
            "gender": {"required": False},
            "preview": {"required": False},
        }

# ----- Agents -----
class AgentReadSerializer(serializers.ModelSerializer):
    user_username = serializers.CharField(source="user.username", read_only=True)
    voice_name = serializers.CharField(source="voice.name", read_only=True)

    class Meta:
        model = Agent
        fields = [
            "id", "bot_id", "user", "user_username", "voice", "voice_name",
            "name", "persona", "max_tokens", "glb_url", "avatar",
            "is_active", "created_at", "updated_at",
        ]
        read_only_fields = fields

class AgentWriteSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(queryset=User.objects.all(), required=False)

    class Meta:
        model = Agent
        fields = ["user", "voice", "name", "persona", "max_tokens", "glb_url", "avatar", "is_active"]
        extra_kwargs = {
            "persona": {"required": False},
            "max_tokens": {"required": False},
            "glb_url": {"required": False},
            "avatar": {"required": False},
            "is_active": {"required": False},
            "voice": {"required": False},
        }

    def create(self, validated: dict[str, Any]):
        if "user" not in validated:
            validated["user"] = self.context["request"].user
        return super().create(validated)

# ----- Sessions -----
class SessionReadSerializer(serializers.ModelSerializer):
    user_username = serializers.CharField(source="user.username", read_only=True)
    agent_bot_id = serializers.UUIDField(source="agent.bot_id", read_only=True)
    agent_name = serializers.CharField(source="agent.name", read_only=True)

    class Meta:
        model = Session
        fields = [
            "id", "thread_id", "user", "user_username",
            "agent", "agent_bot_id", "agent_name",
            "title", "summary", "is_active", "created_at", "updated_at",
        ]
        read_only_fields = fields

class SessionWriteSerializer(serializers.ModelSerializer):
    # Allow referencing agent by bot_id as an alternative
    agent_bot_id = serializers.UUIDField(write_only=True, required=False)

    class Meta:
        model = Session
        fields = ["user", "agent", "agent_bot_id", "title", "summary", "is_active"]
        extra_kwargs = {
            "user": {"required": False},
            "agent": {"required": False},  # either agent or agent_bot_id
            "title": {"required": False},
            "summary": {"required": False},
            "is_active": {"required": False},
        }

    def validate(self, attrs):
        if not attrs.get("agent") and not attrs.get("agent_bot_id"):
            raise serializers.ValidationError("Provide either 'agent' or 'agent_bot_id'.")
        return attrs

    def create(self, validated):
        if "user" not in validated:
            validated["user"] = self.context["request"].user
        bot_id = validated.pop("agent_bot_id", None)
        if bot_id and not validated.get("agent"):
            from agent.models import Agent
            validated["agent"] = Agent.objects.get(bot_id=bot_id)
        return super().create(validated)

# ----- Chats -----
class ChatReadSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(source="session.thread_id", read_only=True)
    agent_bot_id = serializers.UUIDField(source="session.agent.bot_id", read_only=True)

    class Meta:
        model = Chat
        fields = [
            "id", "session", "thread_id", "query", "response",
            "emotion", "viseme", "meta", "created_at", "updated_at", "agent_bot_id",
        ]
        read_only_fields = fields

class ChatWriteSerializer(serializers.ModelSerializer):
    # Allow posting by thread_id instead of session pk
    thread_id = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = Chat
        fields = ["session", "thread_id", "query", "response", "emotion", "viseme", "meta"]
        extra_kwargs = {
            "session": {"required": False},
            "response": {"required": False},
            "emotion": {"required": False},
            "viseme": {"required": False},
            "meta": {"required": False},
        }

    def validate(self, attrs):
        if not attrs.get("session") and not attrs.get("thread_id"):
            raise serializers.ValidationError("Provide either 'session' or 'thread_id'.")
        return attrs

    def create(self, validated):
        thread_id = validated.pop("thread_id", None)
        if thread_id and not validated.get("session"):
            from memory.models import Session
            validated["session"] = Session.objects.get(thread_id=thread_id)
        return super().create(validated)

# ----- Slides -----
class SlidesReadSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(source="session.thread_id", read_only=True)

    class Meta:
        model = Slides
        fields = ["id", "session", "thread_id", "title", "summary", "editorjs", "created_at", "updated_at"]
        read_only_fields = fields

class SlidesWriteSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = Slides
        fields = ["session", "thread_id", "title", "summary", "editorjs"]
        extra_kwargs = {
            "session": {"required": False},
            "title": {"required": False},
            "summary": {"required": False},
            "editorjs": {"required": False},  # editorjs can be set later
        }

    def create(self, validated):
        thread_id = validated.pop("thread_id", None)
        from memory.models import Session
        if thread_id and not validated.get("session"):
            validated["session"] = Session.objects.get(thread_id=thread_id)
        # OneToOne: create or update existing
        obj, _ = Slides.objects.update_or_create(
            session=validated["session"], defaults={k: v for k, v in validated.items() if k != "session"}
        )
        return obj
