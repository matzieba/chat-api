from django.db import models
from enum import Enum


class ChoicesEnum(Enum):
    @classmethod
    def choices(cls):
        return [(item.value, item.name.replace("_", " ").capitalize()) for item in cls]

    @classmethod
    def values(cls):
        return [item.value for item in cls]


class MessageRoleEnum(ChoicesEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(models.Model):
    conversation = models.ForeignKey(
        "Conversation",
        on_delete=models.PROTECT,
        related_name="messages",
        null=True,
    )
    role = models.CharField(max_length=48, choices=MessageRoleEnum.choices(), null=True)
    timestamp = models.DateTimeField(db_index=True)

    # User communication messages
    message_text = models.TextField(null=True)

    # Functions messages
    content = models.JSONField(null=True)
    tool_calls = models.JSONField(null=True)
    tool_call_id = models.CharField(max_length=128, null=True)

    def __str__(self):
        return f"Message {self.id} - {self.conversation.user.username} ({self.role})"
