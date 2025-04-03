import random

import tensorflow as tf
from django.db import transaction
from rest_framework import viewsets, status, serializers
from rest_framework.response import Response
import chess

from chat_api.models import User
from chess_api.mcts import run_mcts_batched
from chess_api.models import ChessGame


MODEL_PATH = "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/train_supervised/my_engine_eval_model_100GB_of_parsed_games_pure_argmax.keras"
model = tf.keras.models.load_model(MODEL_PATH)


class ChessGameSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChessGame
        fields = ['game_id', 'board_state', 'moves', 'game_status', 'created_at', 'current_player']

class ChessGameCreateSerializer(serializers.ModelSerializer):
    human_player = serializers.PrimaryKeyRelatedField(queryset=User.objects.all(), required=True)
    game_id = serializers.IntegerField(read_only=True)

    class Meta:
        model = ChessGame
        fields = ['human_player', 'game_id']

    def create(self, validated_data):
        return ChessGame.objects.create(**validated_data)

class GameViewSet(viewsets.ModelViewSet):
    model = ChessGame
    queryset = ChessGame.objects.all()

    def get_serializer_class(self):
        if self.action in ['list', 'retrieve']:
            return ChessGameSerializer
        elif self.action == 'create':
            return ChessGameCreateSerializer
        return ChessGameSerializer

    @transaction.atomic
    def post(self, request, *args, **kwargs):
        try:
            game = ChessGame.objects.select_for_update().get(game_id=request.data.get("game_id"))
        except ChessGame.DoesNotExist:
            return Response({"error": "Game not found"}, status=status.HTTP_404_NOT_FOUND)

        player = request.data.get("player", None)
        move_input = request.data.get("move", None)

        if game.current_player != player:
            return Response({"error": f"It is {game.current_player}'s turn."}, status=status.HTTP_400_BAD_REQUEST)

        board = chess.Board(game.board_state)

        if player == "white":
            chess_move = self.make_player_move(board, move_input)
            if not chess_move:
                return Response({"error": "Invalid move"}, status=status.HTTP_400_BAD_REQUEST)
            move = chess_move  # Use the chess.Move object
            game.current_player = 'black'
        elif player == "black":
            if board.is_game_over():
                game.game_status = self.get_game_status(board)
            else:
                move = self.get_model_move(board)  # move is a chess.Move object
                board.push(move)
                game.current_player = 'white'

        game.board_state = board.fen()
        game.moves.append(move.uci())  # Now 'move' is always a chess.Move object
        game.game_status = self.get_game_status(board)
        game.save()
        serializer = self.get_serializer(game)
        return Response(serializer.data)

    @staticmethod
    def make_player_move(board, move_str):
        try:
            chess_move = chess.Move.from_uci(move_str)
            if not board.is_legal(chess_move):
                return None
            board.push(chess_move)
            return chess_move  # Return the chess.Move object
        except ValueError:
            return None

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

    def get_model_move(self, board: chess.Board) -> chess.Move:
        best_move = run_mcts_batched(model, board, n_simulations=24, batch_size=12)
        return best_move
        # legal_moves_list = list(board.legal_moves)
        # return random.choice(legal_moves_list)

