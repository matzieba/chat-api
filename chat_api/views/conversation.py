import logging

from requests import Response
from rest_framework import viewsets, serializers, status
from rest_framework.decorators import action

from chat_api.groq.groq_chat_processor import ChatProcessor
from chat_api.models.conversation import Conversation
from chat_api.models.message import Message
from rest_framework.response import Response
from django.utils.timezone import now

logger = logging.getLogger(__name__)


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = "__all__"


class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = "__all__"


class ConversationViewSet(viewsets.ModelViewSet):
    queryset = Conversation.objects.all()
    serializer_class = ConversationSerializer

    def retrieve(self, request, *args, **kwargs):
        """
        Retrieve a model instance.
        """
        instance = self.get_object()
        serializer = self.get_serializer(instance)

        if not instance.messages.exists():
            chat_processor = ChatProcessor(instance.user.id, conversation=instance)
            greeting_message = chat_processor.create_message('', 'assistant')
            greeting_text = chat_processor.process_chat([greeting_message])
            Message.objects.create(
                conversation=instance,
                message_text=greeting_text,
                timestamp=now(),
                role='assistant'
            )

        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def messages(self, request, pk=None):
        conversation_id = pk or self.kwargs.get('pk', None)
        user_message_text = request.data.get('message', None)

        if not all([conversation_id, user_message_text]):
            return Response({"error": "Invalid request"}, status=status.HTTP_400_BAD_REQUEST)

        conversation = Conversation.objects.get(pk=conversation_id)
        past_messages = Message.objects.filter(conversation=conversation).order_by('timestamp')

        chat_processor = ChatProcessor(conversation.user.id, conversation)

        chat_messages = []
        for message in past_messages:
            chat_messages.append(chat_processor.create_message(message.message_text, message.role))

        chat_messages.append(chat_processor.create_message(user_message_text, 'user'))
        chat_response = chat_processor.process_chat(chat_messages)

        user_message = Message.objects.create(
            conversation=conversation,
            message_text=user_message_text,
            timestamp=now(),
            role='user'
        )

        assistant_message = Message.objects.create(
            conversation=conversation,
            message_text=chat_response,
            timestamp=now(),
            role='assistant'
        )

        user_message_serializer = MessageSerializer(user_message)
        assistant_message_serializer = MessageSerializer(assistant_message)

        return Response({
            'user_message': user_message_serializer.data,
            'assistant_message': assistant_message_serializer.data,
        }, status=status.HTTP_200_OK)
