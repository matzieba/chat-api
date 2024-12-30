import pickle
import random
import numpy as np
import tensorflow as tf
from django.db import transaction
from rest_framework import viewsets, status, serializers
from rest_framework.response import Response
import chess
from chat_api.models import User
from chess_api.models import ChessGame


# Constants
NUM_SQUARES = 64
NUM_PROMOTION_OPTIONS = 5  # No promotion + 4 promotion pieces
PROMOTION_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
num_moves = NUM_SQUARES * NUM_SQUARES * NUM_PROMOTION_OPTIONS  # 64 * 64 * 5 = 20,480

# Load your trained model
MODEL_PATH = "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/models/best_trained_transformer_model.keras"
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
        move = request.data.get("move", None)

        if game.current_player != player:
            return Response({"error": f"It is {game.current_player}'s turn."}, status=status.HTTP_400_BAD_REQUEST)

        board = chess.Board(game.board_state)

        if player == "white":
            if not self.make_player_move(board, move):
                return Response({"error": "Invalid move"}, status=status.HTTP_400_BAD_REQUEST)
            game.current_player = 'black'
        elif player == "black":
            if board.is_game_over():
                game.game_status = self.get_game_status(board)
            else:
                move = self.get_model_move(board)
                board.push(move)
                game.current_player = 'white'

        # Update game state
        game.board_state = board.fen()
        game.moves.append(move)
        game.game_status = self.get_game_status(board)
        game.save()

        serializer = self.get_serializer(game)
        return Response(serializer.data)

    @staticmethod
    def make_player_move(board, move):
        try:
            chess_move = chess.Move.from_uci(move)
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

    def get_model_move(self, board):
        # Use convert_board_to_sequence to get the input_seq of shape (64,)
        state = convert_board_to_sequence(board)

        # Generate the mask_input
        mask = get_legal_moves_mask(board)

        # Expand dimensions to add batch size (1)
        state_input = np.expand_dims(state, axis=0)  # Shape: (1, 64)
        mask_input = np.expand_dims(mask, axis=0)  # Shape: (1, num_moves)

        # Prepare inputs as a dictionary
        inputs = {'input_seq': state_input, 'mask_input': mask_input}

        # Get the output probabilities
        output_probs = model.predict(inputs, verbose=0).ravel()

        # Now, select the move with the highest probability among legal moves
        legal_moves = list(board.legal_moves)
        valid_moves = []
        valid_indices = []
        for move in legal_moves:
            try:
                action_index = encode_move(move)
                valid_moves.append(move)
                valid_indices.append(action_index)
            except (ValueError, IndexError):
                continue

        # Get probabilities for valid moves
        valid_probs = output_probs[valid_indices]
        best_index = np.argmax(valid_probs)
        best_move = valid_moves[best_index]
        return best_move


# Helper Functions

def encode_move(move):
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion
    # Adjust promotion index if promotion
    if promotion:
        promotion_index = PROMOTION_PIECES.index(promotion)
    else:
        promotion_index = 0  # No promotion is at index 0
    action_index = (from_square * NUM_SQUARES * NUM_PROMOTION_OPTIONS) + \
                   (to_square * NUM_PROMOTION_OPTIONS) + \
                   promotion_index
    return action_index


def convert_board_to_sequence(board):
    piece_to_id = {
        None: 0,  # Empty square
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
    }
    # Initialize sequence with 64 tokens (one for each square)
    sequence = np.zeros(64, dtype=np.int32)
    for idx in range(64):
        piece = board.piece_at(idx)
        piece_symbol = piece.symbol() if piece else None
        token_id = piece_to_id[piece_symbol]
        sequence[idx] = token_id
    return sequence


def get_legal_moves_mask(board):
    mask = np.zeros(num_moves, dtype=np.float32)
    for move in board.legal_moves:
        try:
            action_index = encode_move(move)
            mask[action_index] = 1.0
        except (ValueError, IndexError):
            pass
    return mask