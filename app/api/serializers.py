# api/serializers.py
from __future__ import annotations
from typing import Any, Tuple
import json
import mimetypes
import re

from django.contrib.auth import get_user_model
from django.db import transaction
from django.core.files.base import ContentFile
from rest_framework import serializers

from agent.models import Voice, Agent
from memory.models import Session, Chat, Slides, Document

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

_DATA_URI_HEADER_RE = re.compile(r"^data:([^;,]+)?(?:;charset=[^;,]+)?;base64,", re.I)

def _parse_data_uri(s: str) -> Tuple[str, str]:
    """
    Returns (mime, base64_payload_str). If not a data URI, treat as raw base64 with octet-stream MIME.
    """
    s = (s or "").strip()
    if not s:
        return "application/octet-stream", ""
    if s.lower().startswith("data:") and "," in s:
        header, payload = s.split(",", 1)
        m = _DATA_URI_HEADER_RE.match(header + ",")
        mime = (m.group(1) if m and m.group(1) else "application/octet-stream")
        return mime, payload
    return "application/octet-stream", s

def _decode_b64_and_mime(s: str) -> Tuple[bytes, str]:
    """Decode and also return MIME from the data URI header."""
    import base64
    mime, payload = _parse_data_uri(s)
    if not payload:
        return b"", mime
    try:
        return base64.b64decode(payload, validate=True), mime
    except Exception:
        raise serializers.ValidationError("Invalid base64 payload (data URI supported).")

_MIME_EXT_OVERRIDES = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "application/json": ".json",
}

def _ext_from_mime(mime: str) -> str:
    if not mime:
        return ".bin"
    ext = _MIME_EXT_OVERRIDES.get(mime.lower())
    if not ext:
        ext = mimetypes.guess_extension(mime) or ".bin"
    return ext.lower()

def _safe_stem_from_title(title: str, fallback: str) -> str:
    base = (title or "").strip() or fallback
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-._")
    if not base:
        base = fallback
    return base[:80]

def _document_ext_allowed(ext: str) -> bool:
    """Check Document.file's FileExtensionValidator if present."""
    try:
        from django.core.validators import FileExtensionValidator
        f = Document._meta.get_field("file")
        for v in getattr(f, "validators", []):
            if isinstance(v, FileExtensionValidator):
                allowed = {e.lstrip(".").lower() for e in v.allowed_extensions}
                return ext.lstrip(".").lower() in allowed
    except Exception:
        # If anything goes wrong, don't block here; model validation will still run.
        return True
    return True


# ----- Voices -----
class VoiceReadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Voice
        fields = "__all__"

class VoiceWriteSerializer(serializers.ModelSerializer):
    # Upload preview via base64 (data URI supported) — infer type from header
    preview_b64 = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = Voice
        fields = ["name", "service", "voice_id", "gender", "preview_b64"]
        extra_kwargs = {
            "service": {"required": False},
            "voice_id": {"required": False},
            "gender": {"required": False},
        }

    def create(self, validated):
        raw = validated.pop("preview_b64", None)
        voice = Voice.objects.create(**validated)
        if raw:
            blob, mime = _decode_b64_and_mime(raw)
            if not blob:
                raise serializers.ValidationError("preview_b64 decoded to empty content.")
            ext = _ext_from_mime(mime)
            voice.preview = ContentFile(blob, name=f"preview{ext}")
            voice.save(update_fields=["preview"])
        return voice

    def update(self, instance: Voice, validated):
        raw = validated.pop("preview_b64", None)
        for f, v in validated.items():
            setattr(instance, f, v)
        if raw:
            blob, mime = _decode_b64_and_mime(raw)
            ext = _ext_from_mime(mime)
            instance.preview = ContentFile(blob, name=f"preview{ext}")
        instance.save()
        return instance


# ----- Agents -----
class AgentReadSerializer(serializers.ModelSerializer):
    email = serializers.CharField(source="user.email", read_only=True)
    voice = VoiceReadSerializer(read_only=True) 

    class Meta:
        model = Agent
        fields = [
            "id", "bot_id", "email", "voice", "name",
            "description", "persona", "age", "max_tokens",
            "glb", "avatar", "is_active", "created_at", "updated_at",
        ]
        read_only_fields = fields


class AgentWriteSerializer(serializers.ModelSerializer):
    # Upload avatar via base64 (data URI supported) — infer type from header
    avatar_b64 = serializers.CharField(write_only=True, required=False, allow_blank=True)

    class Meta:
        model = Agent
        fields = [
            "voice", "name", "description", "persona", "age", "max_tokens",
            "glb", "avatar_b64", "is_active"
        ]
        extra_kwargs = {
            "description": {"required": False, "allow_blank": True},
            "persona": {"required": False, "allow_blank": True},
            "age": {"required": False},
            "max_tokens": {"required": False},
            "glb": {"required": False},
            "is_active": {"required": False},
            "voice": {"required": False},
        }

    def create(self, validated):
        b64 = validated.pop("avatar_b64", None)
        agent = Agent.objects.create(**validated)
        if b64:
            blob, mime = _decode_b64_and_mime(b64)
            if blob:
                ext = _ext_from_mime(mime)
                agent.avatar = ContentFile(blob, name=f"avatar{ext}")
                agent.save(update_fields=["avatar"])
        return agent

    def update(self, instance: Agent, validated):
        b64 = validated.pop("avatar_b64", None)
        for f, v in validated.items():
            setattr(instance, f, v)
        if b64:
            blob, mime = _decode_b64_and_mime(b64)
            if blob:
                ext = _ext_from_mime(mime)
                instance.avatar = ContentFile(blob, name=f"avatar{ext}")
        instance.save()
        return instance


# ----- Sessions -----
class SessionReadSerializer(serializers.ModelSerializer):
    email = serializers.CharField(source="user.email", read_only=True)
    bot_id = serializers.UUIDField(source="agent.bot_id", read_only=True)
    agent = serializers.CharField(source="agent.name", read_only=True)

    class Meta:
        model = Session
        fields = [
            "id", "thread_id", "email", "agent", "bot_id",
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
            "id", "thread_id", "bot_id", "query", "response",
            "emotion", "meta", "created_at", "updated_at"
        ]
        read_only_fields = fields


class ChatWriteSerializer(serializers.ModelSerializer):
    # REQUESTS accept thread_id (not session)
    thread_id = serializers.CharField(write_only=True, required=True)
    emotion = CoercingJSONField(required=False)
    meta    = CoercingJSONField(required=False)

    class Meta:
        model = Chat
        fields = ["thread_id", "query", "response", "emotion", "meta"]
        extra_kwargs = {"response": {"required": False}}

    def create(self, validated):
        thread_id = validated.pop("thread_id")
        sess = Session.objects.get(thread_id=thread_id)
        return Chat.objects.create(session=sess, **validated)

    def update(self, instance: Chat, validated):
        # Allow switching session via thread_id on update if provided
        thread_id = validated.pop("thread_id", None)
        if thread_id:
            instance.session = Session.objects.get(thread_id=thread_id)
        for f in ("query", "response", "emotion", "meta"):
            if f in validated:
                setattr(instance, f, validated[f])
        instance.save()
        return instance


# ----- Slides -----
class SlidesReadSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(source="session.thread_id", read_only=True)
    bot_id = serializers.UUIDField(source="session.agent.bot_id", read_only=True)

    class Meta:
        model = Slides
        fields = [
            "id", "thread_id", "bot_id", "version",
            "title", "summary", "editorjs",
            "updated_by", "created_at", "updated_at",
        ]
        read_only_fields = fields


class SlidesWriteSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(write_only=True, required=True)
    rotate = serializers.BooleanField(write_only=True, required=False, default=True)
    updated_by = serializers.CharField(write_only=True, required=False, allow_blank=True, max_length=64)
    editorjs = CoercingJSONField(required=False)

    class Meta:
        model = Slides
        fields = [
            "thread_id", "title", "summary",
            "editorjs", "rotate", "updated_by",
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
            if "editorjs" in validated:  instance.editorjs = editor or ""
            instance.updated_by = updated_by
            instance.save(update_fields=["title", "summary", "editorjs", "updated_by", "updated_at"])

        return instance


# ----- Documents -----
class DocumentReadSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(source="session.thread_id", read_only=True)
    bot_id = serializers.UUIDField(source="session.agent.bot_id", read_only=True)

    class Meta:
        model = Document
        fields = [
            "id", "thread_id", "bot_id",
            "title", "file", "index_status",
            "meta", "created_at", "updated_at",
        ]
        read_only_fields = fields

class DocumentWriteSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(write_only=True, required=True)
    file_b64 = serializers.CharField(write_only=True, required=True)
    title = serializers.CharField(required=False, allow_blank=True, max_length=255)

    class Meta:
        model = Document
        fields = ["thread_id", "title", "file_b64"]

    def create(self, validated):
        thread_id = validated.pop("thread_id")
        file_b64 = validated.pop("file_b64")
        title = validated.pop("title", "")

        sess = Session.objects.get(thread_id=thread_id)

        blob, mime = _decode_b64_and_mime(file_b64)
        if not blob:
            raise serializers.ValidationError("file_b64 decoded to empty content.")

        ext = _ext_from_mime(mime)
        # Ensure extension is allowed for Document.file (avoid model-level 500s)
        if not _document_ext_allowed(ext):
            raise serializers.ValidationError(f"Unsupported or mismatched file type: {mime} ({ext}).")

        stem = _safe_stem_from_title(title, "document")
        # Force the extension from MIME, ignore any dots in title
        filename = f"{stem}{ext}"

        content = ContentFile(blob, name=filename)
        obj = Document.objects.create(session=sess, title=title or "", file=content)
        return obj

    def update(self, instance: Document, validated):
        # Allow switching session via thread_id
        thread_id = validated.pop("thread_id", None)
        if thread_id:
            instance.session = Session.objects.get(thread_id=thread_id)

        title = validated.pop("title", None)
        file_b64 = validated.pop("file_b64", None)

        if title is not None:
            instance.title = title or ""

        if file_b64:
            blob, mime = _decode_b64_and_mime(file_b64)
            ext = _ext_from_mime(mime)
            if not _document_ext_allowed(ext):
                raise serializers.ValidationError(f"Unsupported or mismatched file type: {mime} ({ext}).")
            stem = _safe_stem_from_title(instance.title or "document", "document")
            filename = f"{stem}{ext}"
            instance.file = ContentFile(blob, name=filename)

        instance.save()
        return instance
