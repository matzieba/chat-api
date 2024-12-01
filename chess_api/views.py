import random


from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
from rest_framework import serializers
import chess
from chess_api.models import ChessGame

class ChessGameSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChessGame
        fields = ['game_id', 'board_state', 'moves', 'game_status', 'created_at', 'current_player']

class GameViewSet(viewsets.ModelViewSet):
    class Meta:
        model = ChessGame
        fields = "__all__"

    queryset = ChessGame.objects.all()
    serializer_class = ChessGameSerializer

    def post(self, request, *args, **kwargs):
        game_id = self.kwargs.get('pk', None)
        game = ChessGame.objects.get(game_id=request.data["game_id"])
        move = request.data.get("move", None)
        player = request.data.get("player", None)
        board = chess.Board(game.board_state)
        if player == "white":
            try:
                board.push_san(move)
            except:
                return Response({"error": "Invalid move"}, status=status.HTTP_400_BAD_REQUEST)
            game.board_state = board.fen()
            game.moves.append(move)
            game.current_player = 'black'
            game.save()
        elif player == "black":
            move = str(random.choice(list(board.legal_moves)))
            board.push_san(move)
            game.board_state = board.fen()
            game.moves.append(move)
            game.current_player = 'white'
            game.save()
        serializer = ChessGameSerializer(game)
        return Response(serializer.data)

