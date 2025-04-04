from drf_writable_nested import WritableNestedModelSerializer
from rest_framework.fields import SerializerMethodField

from chat_api.clients.firebase import FirebaseClient
from chat_api.models import User
from chat_api.models.conversation import Conversation
from chat_api.serializers.company import CompanySerializer
from chess_api.models import ChessGame
from django.db.models import ObjectDoesNotExist
import chess

class UserSerializer(WritableNestedModelSerializer):
    chat_id = SerializerMethodField()
    game_id = SerializerMethodField()
    class Meta:
        model = User
        read_only_fields = ["id", "firebase_uid"]
        fields = [
            "first_name",
            "last_name",
            "email",
            "phone",
            "id",
            "firebase_uid",
            "company",
            "job_title",
            "chat_id",
            "game_id",
        ]

    company = CompanySerializer(required=False, allow_null=True)

    def get_chat_id(self, obj):
        try:
            conversation = obj.conversations.get()
        except ObjectDoesNotExist:
            conversation = Conversation.objects.create(user=obj)
        return conversation.id

    def get_game_id(self, obj):
        try:
            game = obj.games.order_by("-created_at").first()
        except ObjectDoesNotExist:
            game = ChessGame.objects.create(human_player=obj, board_state=chess.Board().fen(), moves=[], game_status='ongoing')
        return game.game_id

    def create(self, validated_data):
        if validated_data.get("email") and not validated_data.get("firebase_uid"):
            first_name = validated_data.get("first_name")
            last_name = validated_data.get("last_name")
            email = validated_data.get("email")
            validated_data["firebase_uid"] = FirebaseClient.create_firebase_user(
                first_name, last_name, email
            )
        return super().create(validated_data)
