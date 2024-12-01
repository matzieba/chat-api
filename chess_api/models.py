from django.contrib.postgres.fields import ArrayField
from django.db import models
from chat_api.models.user import User
# Create your models here.
class ChessGame(models.Model):
    game_id = models.AutoField(primary_key=True)  # unique game id
    board_state = models.TextField()  # a text field to store the FEN (Forsyth-Edwards Notation) of the current board state
    moves = ArrayField(models.CharField(max_length=10), blank=True, null=True)  # an array to store the moves as 'e2e4' etc.
    game_status = models.CharField(max_length=10, null=True)  # status of the game, can be 'ongoing', 'checkmate', 'stalemate', 'draw' etc.
    created_at = models.DateTimeField(auto_now_add=True)
    current_player = models.CharField(max_length=256, default="white", blank=True, null=True)
    human_player = models.ForeignKey(User, on_delete=models.CASCADE, related_name='games')