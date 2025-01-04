import numpy as np
# import tensorflow as tf
from django.db import transaction
from rest_framework import viewsets, status, serializers
from rest_framework.response import Response
import chess
import requests

import settings
from chat_api.models import User
from chess_api.models import ChessGame
# from chess_engine.engine.utils import initialize_move_encoding, board_to_planes, encode_move, get_legal_moves_mask, NUM_MOVES


# Load your trained model
MODEL_PATH = "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/best_trained_model_cnn.keras"
# model = tf.keras.models.load_model(MODEL_PATH)

# Initialize move encoding
# initialize_move_encoding()

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
                move = self.get_ai_move_from_service(board)  # move is a chess.Move object
                board.push(move)
                game.current_player = 'white'

        # Update game state
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

    # def get_model_move(self, board):
    #     # Use board_to_planes to get the input_board of shape (8, 8, 14)
    #     state = board_to_planes(board)  # Returns shape (8, 8, 14)
    #
    #     # Generate the mask_input
    #     mask = get_legal_moves_mask(board)
    #
    #     # Expand dimensions to add batch size (1)
    #     state_input = np.expand_dims(state, axis=0)  # Shape: (1, 8, 8, 14)
    #     mask_input = np.expand_dims(mask, axis=0)    # Shape: (1, NUM_MOVES)
    #
    #     # Prepare inputs as a dictionary
    #     inputs = {'input_board': state_input, 'mask_input': mask_input}
    #
    #     # Get the output probabilities
    #     model_outputs = model.predict(inputs, verbose=0)
    #     policy_output = model_outputs[0]  # Get the first output (policy)
    #     output_probs = policy_output.ravel()
    #
    #     # Now, select the move with the highest probability among legal moves
    #     legal_moves = list(board.legal_moves)
    #     valid_moves = []
    #     valid_indices = []
    #     for move in legal_moves:
    #         try:
    #             action_index = encode_move(move)
    #             valid_moves.append(move)
    #             valid_indices.append(action_index)
    #         except (ValueError, IndexError):
    #             continue
    #
    #     # Get probabilities for valid moves
    #     valid_probs = output_probs[valid_indices]
    #     best_index = np.argmax(valid_probs)
    #     best_move = valid_moves[best_index]
    #     return best_move

    def get_ai_move_from_service(self, board):
        """ Make a POST request to the tf_service to obtain the AI move and convert it to a chess.Move object. """
        tf_service_url = settings.TF_SERVICE_URL  # e.g., 'http://tf_service:8000'
        endpoint = f"{tf_service_url}/predict-move"
        payload = {"fen": board.fen()}
        try:
            response = requests.post(endpoint, json=payload, timeout=5)
        except requests.exceptions.RequestException as e:
            # Handle request exceptions
            print(f"Failed to connect to AI service: {e}")
            return None

        if response.status_code == 200:
            best_move_uci = response.json().get("best_move")
            if best_move_uci:
                try:
                    move = chess.Move.from_uci(best_move_uci)
                    return move  # Return the chess.Move object
                except ValueError:
                    print(f"Invalid move format received from AI service: {best_move_uci}")
                    return None
            else:
                print("AI service did not return a move.")
                return None
        else:
            print(f"AI service returned an error: {response.status_code} - {response.text}")
            return None