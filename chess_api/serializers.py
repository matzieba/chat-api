from rest_framework import  serializers

from chat_api.models import User
from chess_api.models import ChessGame


class ChessGameSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChessGame
        fields = ['game_id', 'board_state', 'moves', 'game_status', 'created_at', 'current_player']

class PlayerStatisticsSerializer(serializers.ModelSerializer):
    total_games = serializers.SerializerMethodField()
    wins = serializers.SerializerMethodField()
    win_rate = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = [
            'id',
            'first_name',
            'last_name',
            'total_games',
            'wins',
            'win_rate'
        ]

    def get_total_games(self, user):
        return ChessGame.objects.filter(human_player=user).count()

    def get_wins(self, user):
        return ChessGame.objects.filter(winner=user).count()

    def get_win_rate(self, user):
        total = self.get_total_games(user)
        if total == 0:
            return "0.0%"
        w = self.get_wins(user)
        return f"{(w / total) * 100:.1f}%"

class ChessGameCreateSerializer(serializers.ModelSerializer):
    human_player = serializers.PrimaryKeyRelatedField(queryset=User.objects.all(), required=True)
    game_id = serializers.IntegerField(read_only=True)

    class Meta:
        model = ChessGame
        fields = ['human_player', 'game_id']

    def create(self, validated_data):
        return ChessGame.objects.create(**validated_data)