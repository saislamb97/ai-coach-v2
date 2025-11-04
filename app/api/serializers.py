# serializers.py
from __future__ import annotations
from typing import Any
import json

from django.contrib.auth import get_user_model
from django.db import transaction
from rest_framework import serializers

from agent.models import Voice, Agent
from memory.models import Session, Chat, Slides

User = get_user_model()


# ---------- Utilities ----------
class CoercingJSONField(serializers.JSONField):
    """
    Accepts dict/list/primitive OR a JSON-encoded string; returns proper Python obj.
    Ensures stored value is never a raw string when it should be JSON.
    """
    def to_internal_value(self, data: Any):
        # Already structured → pass through normal validation
        if isinstance(data, (dict, list)) or data is None:
            return super().to_internal_value(data)
        # If client sent a string, try to parse it
        if isinstance(data, str):
            s = data.strip()
            if s == "":
                return None
            try:
                data = json.loads(s)
            except Exception:
                raise serializers.ValidationError("Invalid JSON: expected object/array or a valid JSON string.")
            return super().to_internal_value(data)
        # Accept ints/bools/null occasionally used in metadata
        return super().to_internal_value(data)


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
    class Meta:
        model = Agent
        fields = ["voice", "name", "persona", "max_tokens", "glb_url", "avatar", "is_active"]
        extra_kwargs = {
            "persona": {"required": False},
            "max_tokens": {"required": False},
            "glb_url": {"required": False},
            "avatar": {"required": False},
            "is_active": {"required": False},
            "voice": {"required": False},
        }


# ----- Sessions -----
class SessionReadSerializer(serializers.ModelSerializer):
    user_username = serializers.CharField(source="user.username", read_only=True)
    bot_id = serializers.UUIDField(source="agent.bot_id", read_only=True)
    agent_name = serializers.CharField(source="agent.name", read_only=True)

    class Meta:
        model = Session
        fields = [
            "id", "thread_id", "user", "user_username",
            "bot_id", "agent_name",
            "title", "summary", "is_active", "created_at", "updated_at",
        ]
        read_only_fields = fields


class SessionWriteSerializer(serializers.ModelSerializer):
    bot_id = serializers.UUIDField(write_only=True, required=False)

    class Meta:
        model = Session
        fields = ["agent", "bot_id", "title", "summary", "is_active"]
        extra_kwargs = {
            "agent": {"required": False},
            "title": {"required": False},
            "summary": {"required": False},
            "is_active": {"required": False},
        }

    def validate(self, attrs):
        if not attrs.get("agent") and not attrs.get("bot_id"):
            raise serializers.ValidationError("Provide either 'agent' or 'bot_id'.")
        return attrs

    def create(self, validated):
        bot_id = validated.pop("bot_id", None)
        if bot_id and not validated.get("agent"):
            validated["agent"] = Agent.objects.get(bot_id=bot_id)
        return super().create(validated)


# ----- Chats -----
class ChatReadSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(source="session.thread_id", read_only=True)
    bot_id = serializers.UUIDField(source="session.agent.bot_id", read_only=True)

    class Meta:
        model = Chat
        fields = [
            "id", "session", "thread_id", "query", "response",
            "emotion", "viseme", "meta", "created_at", "updated_at", "bot_id",
        ]
        read_only_fields = fields


class ChatWriteSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(write_only=True, required=False)
    # Ensure JSON fields are never stored as quoted strings:
    emotion = CoercingJSONField(required=False)
    viseme  = CoercingJSONField(required=False)
    meta    = CoercingJSONField(required=False)

    class Meta:
        model = Chat
        fields = ["session", "thread_id", "query", "response", "emotion", "viseme", "meta"]
        extra_kwargs = {
            "session": {"required": False},
            "response": {"required": False},
        }

    def validate(self, attrs):
        if not attrs.get("session") and not attrs.get("thread_id"):
            raise serializers.ValidationError("Provide either 'session' or 'thread_id'.")
        return attrs

    def create(self, validated):
        thread_id = validated.pop("thread_id", None)
        if thread_id and not validated.get("session"):
            validated["session"] = Session.objects.get(thread_id=thread_id)
        return super().create(validated)


# ----- Slides -----
class SlidesReadSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(source="session.thread_id", read_only=True)

    class Meta:
        model = Slides
        fields = [
            "id", "session", "thread_id",
            "version",
            "title", "summary", "editorjs",
            "previous_title", "previous_summary", "previous_editorjs",
            "updated_by",
            "created_at", "updated_at",
        ]
        read_only_fields = fields


class SlidesWriteSerializer(serializers.ModelSerializer):
    """
    Default: rotate=True → snapshot previous_* and bump version.
    Pass rotate=false to overwrite current without snapshotting.
    """
    thread_id = serializers.CharField(write_only=True, required=False)
    rotate = serializers.BooleanField(write_only=True, required=False, default=True)
    updated_by = serializers.CharField(write_only=True, required=False, allow_blank=True, max_length=64)
    # Ensure we accept "editorjs" whether sent as object or quoted string:
    editorjs = CoercingJSONField(required=False)

    class Meta:
        model = Slides
        fields = [
            "session", "thread_id",
            "title", "summary", "editorjs",
            "rotate", "updated_by",
        ]
        extra_kwargs = {
            "session": {"required": False},
            "title": {"required": False},
            "summary": {"required": False},
        }

    def _resolve_session(self, validated):
        thread_id = validated.pop("thread_id", None)
        sess = validated.get("session")
        if thread_id and not sess:
            sess = Session.objects.get(thread_id=thread_id)
            validated["session"] = sess
        if not sess:
            raise serializers.ValidationError("Slides require 'session' or 'thread_id'.")
        return sess

    def _who(self) -> str:
        req = self.context.get("request")
        user = getattr(req, "user", None)
        if getattr(user, "is_authenticated", False):
            return f"user:{user.pk}"
        return "api"

    @transaction.atomic
    def create(self, validated):
        sess = self._resolve_session(validated)
        rotate = bool(validated.pop("rotate", True))
        updated_by = (validated.pop("updated_by", "") or self._who())[:64]

        obj, created = Slides.objects.select_for_update().get_or_create(session=sess)
        title   = validated.get("title", obj.title)
        summary = validated.get("summary", obj.summary)
        editor  = validated.get("editorjs", obj.editorjs)

        if rotate or created:
            # rotate_and_update must move current → previous, set new current, bump version
            obj.rotate_and_update(title=title or "", summary=summary or "", editorjs=editor or {}, updated_by=updated_by)
            obj.save()  # let model decide update_fields (version/previous_*/updated_at)
        else:
            if title is not None:    obj.title = title or ""
            if summary is not None:  obj.summary = summary or ""
            if editor is not None:   obj.editorjs = editor or {}
            obj.updated_by = updated_by
            obj.save(update_fields=["title", "summary", "editorjs", "updated_by", "updated_at"])

        return obj

    @transaction.atomic
    def update(self, instance: Slides, validated):
        rotate = bool(validated.pop("rotate", True))
        updated_by = (validated.pop("updated_by", "") or self._who())[:64]

        title   = validated.get("title", instance.title)
        summary = validated.get("summary", instance.summary)
        editor  = validated.get("editorjs", instance.editorjs)

        if rotate:
            instance.rotate_and_update(title=title or "", summary=summary or "", editorjs=editor or {}, updated_by=updated_by)
            instance.save()
        else:
            if "title" in validated:     instance.title = title or ""
            if "summary" in validated:   instance.summary = summary or ""
            if "editorjs" in validated:  instance.editorjs = editor or {}
            instance.updated_by = updated_by
            instance.save(update_fields=["title", "summary", "editorjs", "updated_by", "updated_at"])

        return instance
