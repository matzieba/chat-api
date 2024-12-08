import chess
from django.contrib.postgres.fields import ArrayField
from django.db import models
from chat_api.models.user import User
# Create your models here.
class ChessGame(models.Model):
    game_id = models.AutoField(primary_key=True)
    board_state = models.TextField(default=chess.Board().fen())
    moves = ArrayField(models.CharField(max_length=10), blank=True, null=True, default=list())
    game_status = models.CharField(max_length=10, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    current_player = models.CharField(max_length=256, default="white", blank=True, null=True)
    human_player = models.ForeignKey(User, on_delete=models.CASCADE, related_name='games')