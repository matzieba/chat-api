import numpy as np
import tensorflow as tf
from django.db import transaction
from rest_framework import viewsets, status, serializers
from rest_framework.response import Response
import chess

from chat_api.models import User
from chess_api.mcts import MCTSNode, evaluate_position, expand_node, mcts_search
from chess_api.models import ChessGame
from chess_api.parallell_mcts import run_mcts_leaf_parallel

# Load your trained model
MODEL_PATH = "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/train_supervised/my_chess_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

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
                move = self.get_model_move(board)  # move is a chess.Move object
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
    #     # Create a root node for MCTS
    #     root = MCTSNode(board=board)
    #
    #     # Evaluate position for root node; expand it right away
    #     policy, value = evaluate_position(board, model)
    #     expand_node(root, policy)
    #
    #     # Run MCTS search
    #     # Adjust simulations=, c_puct= as needed
    #     result_node = mcts_search(root, model, simulations=600, c_puct=1.0)
    #
    #     # The best move is the one that leads to 'result_node' from the root
    #     best_move = None
    #     for move, child in root.children.items():
    #         if child is result_node:
    #             best_move = move
    #             break
    #
    #     return best_move

    def get_model_move(self, board):
        """
        Use a leaf-parallel MCTS approach to get the AI move.
        """
        # Create the MCTS root node
        root = MCTSNode(board=board)

        # Optionally, you can do an immediate expansion with the prior if you like:
        # from chess_api.mcts import evaluate_position, expand_node
        # policy, value = evaluate_position(board, model)
        # expand_node(root, policy)

        # Now run leaf-parallel MCTS
        best_root = run_mcts_leaf_parallel(
            root_node=root,
            model=model,
            num_simulations=600,
            batch_size=64,
            c_puct=3.5
        )

        # After MCTS, pick your best move. Typically, choose the child with highest visit_count.
        best_move, best_child = max(
            best_root.children.items(),
            key=lambda item: item[1].visit_count
        )
        return best_move

    # def get_ai_move_from_service(self, board):
    #     """ Make a POST request to the tf_service to obtain the AI move and convert it to a chess.Move object. """
    #     tf_service_url = settings.TF_SERVICE_URL  # e.g., 'http://tf_service:8000'
    #     endpoint = f"{tf_service_url}/predict-move"
    #     payload = {"fen": board.fen()}
    #     try:
    #         response = requests.post(endpoint, json=payload, timeout=5)
    #     except requests.exceptions.RequestException as e:
    #         # Handle request exceptions
    #         print(f"Failed to connect to AI service: {e}")
    #         return None
    #
    #     if response.status_code == 200:
    #         best_move_uci = response.json().get("best_move")
    #         if best_move_uci:
    #             try:
    #                 move = chess.Move.from_uci(best_move_uci)
    #                 return move  # Return the chess.Move object
    #             except ValueError:
    #                 print(f"Invalid move format received from AI service: {best_move_uci}")
    #                 return None
    #         else:
    #             print("AI service did not return a move.")
    #             return None
    #     else:
    #         print(f"AI service returned an error: {response.status_code} - {response.text}")
    #         return None