
import random

import requests
from django.db import transaction
from rest_framework import viewsets, status
from rest_framework.response import Response
import chess

import settings
from chat_api.models import User
from chess_api.models import ChessGame
from chess_api.serializers import ChessGameSerializer, PlayerStatisticsSerializer, ChessGameCreateSerializer


class GameViewSet(viewsets.ModelViewSet):
    model = ChessGame
    queryset = ChessGame.objects.all()

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return ChessGameSerializer
        elif self.action == 'list':
            return PlayerStatisticsSerializer
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
        action = request.data.get("action", None)
        if action == "timeout":
            if player == "white":
                game.game_status = "w_timeout"
            else:
                game.game_status = "b_timeout"
                game.winner = game.human_player
            game.save()
            serializer = self.get_serializer(game)
            return Response(serializer.data)
        if game.current_player != player:
            return Response({"error": f"It is {game.current_player}'s turn."}, status=status.HTTP_400_BAD_REQUEST)

        board = chess.Board(game.board_state)

        if player == "white":
            chess_move = self.make_player_move(board, move_input)
            if not chess_move:
                return Response({"error": "Invalid move"}, status=status.HTTP_400_BAD_REQUEST)
            move = chess_move
            game.current_player = 'black'
        elif player == "black":
            if board.is_game_over():
                game.game_status = self.get_game_status(board)
            else:
                move = self.get_best_move_from_engine(board)
                board.push(move)
                game.current_player = 'white'

        game.board_state = board.fen()
        game.moves.append(move.uci())
        game.game_status = self.get_game_status(board)
        if board.is_game_over():
            if board.is_checkmate():
                if board.turn:
                    pass
                else:
                    game.winner = game.human_player
            else:
                pass
        game.save()
        serializer = self.get_serializer(game)
        return Response(serializer.data)

    def list(self, request, *args, **kwargs):
        users = User.objects.all()
        serializer = self.get_serializer(users, many=True)
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

    def get_best_move_from_engine(self, board):
        url = settings.TF_SERVICE_URL + "/bestmove"
        fen_str = board.fen()
        try:
            r = requests.post(url, json={"fen": fen_str})
            r.raise_for_status()
            move_dict = r.json()
            best_move_uci = move_dict["best_move_uci"]
            best_move = chess.Move.from_uci(best_move_uci)
            return best_move

        except requests.HTTPError as e:
            raise ValueError(f"Error calling chess engine service: {e}")