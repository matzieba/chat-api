import random
from django.db import transaction
from rest_framework import viewsets, status, serializers
from rest_framework.response import Response
import chess
from chat_api.models import User
from chess_api.models import ChessGame
import chess.engine

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
    class Meta:
        model = ChessGame
        fields = "__all__"

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

        promotion_piece = None
        player = request.data.get("player", None)
        if player == "white":
            move = request.data.get("move", None)
            if '=' in move:
                move, promotion_piece = move.split('=')
                promotion_piece = promotion_piece.upper()


        # Extract promotion details if they exist
         # Convert to uppercase

        if game.current_player != player:
            return Response({"error": f"It is {game.current_player}'s turn."}, status=status.HTTP_400_BAD_REQUEST)

        board = chess.Board(game.board_state)
        engine = chess.engine.SimpleEngine.popen_uci(
            "/Users/mateuszzieba/Desktop/dev/chess/stockfish/src/stockfish")


        if player == "white":
            move_result = self.make_player_move(board, move, promotion_piece)
            if not move_result:
                return Response({"error": "Invalid move"}, status=status.HTTP_400_BAD_REQUEST)
            game.current_player = 'black'

        elif player == "black":
            if board.is_game_over():
                game.game_status = self.get_game_status(board)
            else:
                move = engine.play(board, chess.engine.Limit(time=2.0))
                board.push(move.move)
                game.current_player = 'white'

        # Update game state
        game.board_state = board.fen()
        game.moves.append(move)
        game.game_status = self.get_game_status(board)
        game.save()

        serializer = self.get_serializer(game)
        return Response(serializer.data)

    @staticmethod
    def make_player_move(board, move, promotion_piece=None):
        try:
            chess_move = chess.Move.from_uci(move)
            if promotion_piece:
                piece_type = promotion_piece[1].lower()  # Handle human-readable 'WR' format
                valid_promotion_pieces = ['q', 'r', 'b', 'n']
                if piece_type in valid_promotion_pieces:
                    chess_move.promotion = chess.Piece.from_symbol(piece_type).piece_type
                else:
                    return False

            if not board.is_legal(chess_move):
                return False

            board.push(chess_move)
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