# serializers.py
from __future__ import annotations
from typing import Any
import json

from django.contrib.auth import get_user_model
from django.db import transaction
from rest_framework import serializers
from django.core.files.base import ContentFile

from agent.models import Voice, Agent
from memory.models import Session, Chat, Slides, Knowledge

User = get_user_model()


# ---------- Utilities ----------
class CoercingJSONField(serializers.JSONField):
    """
    Accepts dict/list/primitive OR a JSON-encoded string; returns proper Python obj.
    Ensures stored value is never a raw string when it should be JSON.
    """
    def to_internal_value(self, data: Any):
        if isinstance(data, (dict, list)) or data is None:
            return super().to_internal_value(data)
        if isinstance(data, str):
            s = data.strip()
            if s == "":
                return None
            try:
                data = json.loads(s)
            except Exception:
                raise serializers.ValidationError("Invalid JSON: expected object/array or a valid JSON string.")
            return super().to_internal_value(data)
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
    # REQUESTS accept bot_id (not agent)
    bot_id = serializers.UUIDField(write_only=True, required=True)

    class Meta:
        model = Session
        fields = ["bot_id", "title", "summary", "is_active"]
        extra_kwargs = {
            "title": {"required": False},
            "summary": {"required": False},
            "is_active": {"required": False},
        }

    def create(self, validated):
        bot_id = validated.pop("bot_id")
        agent = Agent.objects.get(bot_id=bot_id)
        return Session.objects.create(agent=agent, **validated)

    def update(self, instance: Session, validated):
        # If bot_id provided on update, switch the agent
        bot_id = validated.pop("bot_id", None)
        if bot_id:
            instance.agent = Agent.objects.get(bot_id=bot_id)
        for f in ("title", "summary", "is_active"):
            if f in validated:
                setattr(instance, f, validated[f])
        instance.save()
        return instance


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
    # REQUESTS accept thread_id (not session)
    thread_id = serializers.CharField(write_only=True, required=True)
    emotion = CoercingJSONField(required=False)
    viseme  = CoercingJSONField(required=False)
    meta    = CoercingJSONField(required=False)

    class Meta:
        model = Chat
        fields = ["thread_id", "query", "response", "emotion", "viseme", "meta"]
        extra_kwargs = {
            "response": {"required": False},
        }

    def create(self, validated):
        thread_id = validated.pop("thread_id")
        sess = Session.objects.get(thread_id=thread_id)
        return Chat.objects.create(session=sess, **validated)

    def update(self, instance: Chat, validated):
        # Allow switching session via thread_id on update if provided
        thread_id = validated.pop("thread_id", None)
        if thread_id:
            instance.session = Session.objects.get(thread_id=thread_id)
        for f in ("query", "response", "emotion", "viseme", "meta"):
            if f in validated:
                setattr(instance, f, validated[f])
        instance.save()
        return instance


# ----- Slides -----
class SlidesReadSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(source="session.thread_id", read_only=True)

    class Meta:
        model = Slides
        fields = [
            "id", "session", "thread_id",
            "version",
            "title", "summary", "editorjs",
            "updated_by",
            "created_at", "updated_at",
        ]
        read_only_fields = fields


class SlidesWriteSerializer(serializers.ModelSerializer):
    """
    Default: rotate=True → snapshot previous_* and bump version.
    Pass rotate=false to overwrite current without snapshotting.
    """
    # REQUESTS accept thread_id (not session)
    thread_id = serializers.CharField(write_only=True, required=True)
    rotate = serializers.BooleanField(write_only=True, required=False, default=True)
    updated_by = serializers.CharField(write_only=True, required=False, allow_blank=True, max_length=64)
    editorjs = CoercingJSONField(required=False)

    class Meta:
        model = Slides
        fields = [
            "thread_id",
            "title", "summary", "editorjs",
            "rotate", "updated_by",
        ]
        extra_kwargs = {
            "title": {"required": False},
            "summary": {"required": False},
        }

    def _who(self) -> str:
        req = self.context.get("request")
        user = getattr(req, "user", None)
        if getattr(user, "is_authenticated", False):
            return f"user:{user.pk}"
        return "api"

    @transaction.atomic
    def create(self, validated):
        thread_id = validated.pop("thread_id")
        sess = Session.objects.select_for_update().get(thread_id=thread_id)
        rotate = bool(validated.pop("rotate", True))
        updated_by = (validated.pop("updated_by", "") or self._who())[:64]

        obj, created = Slides.objects.select_for_update().get_or_create(session=sess)
        title   = validated.get("title", obj.title)
        summary = validated.get("summary", obj.summary)
        editor  = validated.get("editorjs", obj.editorjs)

        if rotate or created:
            obj.rotate_and_update(title=title or "", summary=summary or "", editorjs=editor or {}, updated_by=updated_by)
            obj.save()
        else:
            if title is not None:    obj.title = title or ""
            if summary is not None:  obj.summary = summary or ""
            if editor is not None:   obj.editorjs = editor or {}
            obj.updated_by = updated_by
            obj.save(update_fields=["title", "summary", "editorjs", "updated_by", "updated_at"])

        return obj

    @transaction.atomic
    def update(self, instance: Slides, validated):
        # Allow switching underlying session via thread_id
        thread_id = validated.pop("thread_id", None)
        if thread_id:
            instance.session = Session.objects.get(thread_id=thread_id)

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


# ----- Knowledge -----
class KnowledgeReadSerializer(serializers.ModelSerializer):
    user_username = serializers.CharField(source="user.username", read_only=True)
    bot_id = serializers.UUIDField(source="agent.bot_id", read_only=True)
    agent_name = serializers.CharField(source="agent.name", read_only=True)
    file_url = serializers.SerializerMethodField()

    class Meta:
        model = Knowledge
        fields = [
            "id", "key",
            "user", "user_username",
            "agent", "bot_id", "agent_name",
            "title", "file_name",
            "file", "file_url",
            "size_bytes", "mimetype", "sha256", "ext",
            "pages", "rows", "cols",
            "excerpt", "meta",
            "index_status", "index_meta",
            "created_at", "updated_at",
        ]
        read_only_fields = fields

    def get_file_url(self, obj: Knowledge):
        try:
            return obj.file.url
        except Exception:
            return None


class KnowledgeWriteSerializer(serializers.ModelSerializer):
    """
    Create/update using only:
      - bot_id (agent lookup; user taken from agent.user)
      - name   (original client filename including extension)
      - file_b64 (base64 payload; data URI accepted)
    """
    bot_id = serializers.UUIDField(write_only=True, required=True)
    name = serializers.CharField(write_only=True, required=True, max_length=255)
    file_b64 = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = Knowledge
        fields = ["bot_id", "name", "file_b64"]

    # ---- helpers ----
    def _decode_b64(self, b64: str) -> bytes:
        import base64
        s = (b64 or "").strip()
        # support data: URIs (e.g., data:application/pdf;base64,AAA...)
        if "," in s and s.lower().startswith("data:"):
            s = s.split(",", 1)[1]
        try:
            return base64.b64decode(s, validate=True)
        except Exception:
            raise serializers.ValidationError("file_b64 must be valid base64 (data URI supported).")

    def validate_name(self, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise serializers.ValidationError("name is required.")
        if "." not in v or v.startswith("."):
            raise serializers.ValidationError("name must include a valid extension (e.g., report.pdf).")
        if len(v) > 255:
            raise serializers.ValidationError("name is too long.")
        return v

    def create(self, validated):
        bot_id = validated.pop("bot_id")
        name = validated.pop("name")
        file_b64 = validated.pop("file_b64")

        agent = Agent.objects.select_related("user").get(bot_id=bot_id)
        if not agent.user_id:
            raise serializers.ValidationError("Agent is not linked to a user.")

        raw = self._decode_b64(file_b64)
        if not raw:
            raise serializers.ValidationError("file_b64 decoded to empty content.")

        # Build ContentFile with the client-provided name (upload_to will record it as file_name)
        content = ContentFile(raw, name=name)

        obj = Knowledge.objects.create(
            agent=agent,
            user=agent.user,      # user collected from agent.user
            file=content,
            # title is optional; omit to keep default ""
        )
        return obj

    def update(self, instance: Knowledge, validated):
        """
        Optional: support re-upload using same input shape.
        """
        bot_id = validated.pop("bot_id", None)
        name = validated.pop("name", None)
        file_b64 = validated.pop("file_b64", None)

        if bot_id:
            agent = Agent.objects.select_related("user").get(bot_id=bot_id)
            instance.agent = agent
            instance.user = agent.user

        if file_b64 and name:
            raw = self._decode_b64(file_b64)
            content = ContentFile(raw, name=name)
            instance.file = content  # model hooks will reset index and capture file_name

        instance.save()
        return instance
