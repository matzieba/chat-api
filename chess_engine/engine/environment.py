import chess
import numpy as np
import random
from utils import (
    NUM_SQUARES,
    NUM_PROMOTION_OPTIONS,
    NUM_MOVES,
    PROMOTION_PIECES,
    encode_move,
    decode_action,
    get_legal_moves_mask, board_to_planes,
)

class ChessEnvironment:
    def __init__(self, agent_color=chess.WHITE):
        self.agent_color = agent_color
        self.board = chess.Board()
        self.previous_positions = set()
        self.repetition_count = {}

    def reset(self):
        """Reset the board to the initial position."""
        self.board.reset()
        self.previous_positions = set()
        self.repetition_count = {}
        return self.board

    def move(self, move):
        """
        Makes the agent's move if it is legal. Returns (next_state, reward, done, info).
        Reward structure:
          • +1 to +9 for capturing pieces (pawn=+1, knight/bishop=+3, rook=+5, queen=+9).
          • -0.05 per move (step penalty).
          • -1 × (repetitions of position - 1), to discourage repeated boards.
          • If a checkmate/termination:
               Win: +100
               Draw: -10
               Loss: -100
          • Illegal move: -5 and terminate episode.
        """
        reward = 0.0
        done = False
        info = {}

        # If move is legal:
        if move in self.board.legal_moves:
            # Small bonus for capturing an opponent's piece
            capture_reward = self.get_capture_value(move)
            reward += capture_reward

            # Execute the move
            self.board.push(move)

            # Small step penalty to encourage faster resolution
            reward -= 0.05

            # Update repetition count / penalty
            self.update_repetition_count()
            repetition_penalty = self.get_repetition_penalty()
            reward -= repetition_penalty

            # Check for game over (checkmate, draw, etc.)
            if self.board.is_game_over():
                result = self.board.result()  # e.g. '1-0', '0-1', '1/2-1/2'
                reward += self.get_game_result_reward(result)
                done = True
                info['result'] = self.get_result_string(result)

            return self.board, reward, done, info

        else:
            # Illegal move => immediate penalty and episode ends
            return self.board, -5.0, True, {'result': 'illegal move'}

    def opponent_move(self, opponent_model):
        """
        Example method for letting an opponent (e.g., a trained model) move.
        You can also replace this with a random or heuristic opponent.
        """
        if not self.board.is_game_over():
            state_input = board_to_planes(self.board)
            mask = get_legal_moves_mask(self.board)

            # Expand dims for batch input
            state_input_expanded = np.expand_dims(state_input, axis=0)
            mask_input_expanded = np.expand_dims(mask, axis=0)

            model_inputs = {
                'input_board': state_input_expanded,
                'mask_input': mask_input_expanded
            }
            policy_output, _ = opponent_model.predict(model_inputs, verbose=0)
            action_probs = policy_output.ravel()

            # Sample an action from predicted probabilities
            action_index = np.random.choice(NUM_MOVES, p=action_probs)
            opp_move = decode_action(action_index)

            # Fallback if move is illegal
            if opp_move in self.board.legal_moves:
                self.board.push(opp_move)
            else:
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    self.board.push(random.choice(legal_moves))

    def opponent_random_move(self):
        """Example fallback: random move for the opponent."""
        if not self.board.is_game_over():
            legal_moves = list(self.board.legal_moves)
            self.board.push(random.choice(legal_moves))

    def get_game_result_reward(self, result):
        """
        Returns a large reward (positive for win, negative for loss, moderate negative for draw).
        Increase or decrease these magnitudes as desired.
        """
        if (result == '1-0' and self.agent_color == chess.WHITE) or \
           (result == '0-1' and self.agent_color == chess.BLACK):
            return 100.0   # Win
        elif result == '1/2-1/2':
            return -10.0   # Draw
        else:
            return -100.0  # Loss

    def get_result_string(self, result):
        """
        Helper for logging or debugging.
        """
        if result == '1-0':
            return 'white wins'
        elif result == '0-1':
            return 'black wins'
        else:
            return 'draw'

    def get_capture_value(self, move):
        """
        Small reward for capturing opponent pieces. Pawn=+1, Knight/Bishop=+3, Rook=+5, Queen=+9.
        You can adjust if you'd like smaller/higher capturing incentives as well.
        """
        if self.board.is_capture(move):
            captured_square = move.to_square
            captured_piece = self.board.piece_at(captured_square)
            if captured_piece and captured_piece.color != self.agent_color:
                piece_type = captured_piece.piece_type
                if piece_type == chess.PAWN:
                    return 1.0
                elif piece_type in [chess.KNIGHT, chess.BISHOP]:
                    return 3.0
                elif piece_type == chess.ROOK:
                    return 5.0
                elif piece_type == chess.QUEEN:
                    return 9.0
        return 0.0

    def update_repetition_count(self):
        """
        Track how many times the current position (FEN) has appeared.
        """
        fen = self.board.board_fen()
        self.repetition_count[fen] = self.repetition_count.get(fen, 0) + 1

    def get_repetition_penalty(self):
        """
        Returns a small penalty if positions start repeating.
        By default: -1 × (# of times that position has occurred - 1).
        This discourages the agent from repeating the same position.
        """
        fen = self.board.board_fen()
        count = self.repetition_count.get(fen, 0)
        if count > 1:
            # Example: -1 for the second time, -2 for the third time, etc.
            return float(count - 1)
        return 0.0

    def legal_moves(self):
        """Optional helper to return list of legal moves."""
        return list(self.board.legal_moves)
