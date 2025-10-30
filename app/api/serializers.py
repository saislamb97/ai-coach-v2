from rest_framework import serializers
from agent.models import Voice, Agent
from memory.models import Session, Chat, Slides


class VoiceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Voice
        fields = "__all__"


class AgentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Agent
        fields = "__all__"


class SessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Session
        fields = "__all__"


class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chat
        fields = "__all__"


class SlidesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Slides
        fields = "__all__"
