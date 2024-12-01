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
        try:
            game = ChessGame.objects.get(game_id=request.data["game_id"])
        except ChessGame.DoesNotExist:
            return Response({"error": "Game not found"}, status=status.HTTP_404_NOT_FOUND)

        move = request.data.get("move", None)
        player = request.data.get("player", None)
        board = chess.Board(game.board_state)

        if player == "white":
            move_result = self.make_player_move(board, move)
            if not move_result:
                return Response({"error": "Invalid move"}, status=status.HTTP_400_BAD_REQUEST)
            game.current_player = 'black'

        elif player == "black":
            if board.is_game_over():
                game.game_status = self.get_game_status(board)
            else:
                move = str(random.choice(list(board.legal_moves)))
                board.push_san(move)
                game.current_player = 'white'

        game.board_state = board.fen()
        game.moves.append(move)
        game.game_status = self.get_game_status(board)
        game.save()

        serializer = ChessGameSerializer(game)
        return Response(serializer.data)

    @staticmethod
    def make_player_move(board, move):
        try:
            board.push_san(move)
            return True
        except ValueError:
            return False

    @staticmethod
    def get_game_status(board):
        if board.is_checkmate():
            return "checkmate"
        if board.is_stalemate():
            return "stalemate"
        if board.is_insufficient_material():
            return "draw"
        if board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return "draw"
        return "ongoing"

