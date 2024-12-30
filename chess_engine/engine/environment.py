import chess
import numpy as np
import random

from utils import (
    NUM_SQUARES,
    NUM_PROMOTION_OPTIONS,
    NUM_MOVES,
    PROMOTION_PIECES,
    convert_board_to_sequence,
    encode_move,
    decode_action,
    get_legal_moves_mask,
)

class ChessEnvironment:
    def __init__(self, agent_color=chess.WHITE):
        self.agent_color = agent_color
        self.board = chess.Board()
        self.previous_material_balance = self.get_material_balance()
        self.previous_board_total_value = self.get_board_total_value()

    def reset(self):
        self.board.reset()
        self.previous_material_balance = self.get_material_balance()
        self.previous_board_total_value = self.get_board_total_value()
        return self.board

    def move(self, move):
        reward = 0
        done = False
        info = {}

        # Check if move is legal
        if move in self.board.legal_moves:
            # Agent's move
            captured_piece_value = self.get_capture_value(move)
            self.board.push(move)

            # Additional reward for capturing opponent's pieces
            if captured_piece_value > 0:
                reward += captured_piece_value
                info['captured_piece'] = f"Captured opponent's piece worth {captured_piece_value}"

            # Penalize for losing valuable pieces
            lost_piece_value = self.get_lost_piece_value()
            if lost_piece_value > 0:
                reward -= lost_piece_value
                info['lost_piece'] = f"Lost own piece worth {lost_piece_value}"

            # Encourage development and castling
            development_reward = self.get_development_reward()
            reward += development_reward

            # Check for game over after agent's move
            if self.board.is_game_over():
                result = self.board.result()
                reward += self.get_game_result_reward(result)
                done = True
                info['result'] = self.get_result_string(result)
                return self.board, reward, done, info

            # Small time penalty per move to encourage faster wins
            reward -= 0.01  # Adjust as needed

            return self.board, reward, done, info
        else:
            # Penalize illegal moves heavily and end the episode
            reward -= 10
            done = True
            info['result'] = 'illegal move'
            return self.board, reward, done, info

    def opponent_move(self, opponent_model):
        if not self.board.is_game_over():
            # Convert board to sequence
            state_input = convert_board_to_sequence(self.board)
            # Generate legal moves mask
            mask = get_legal_moves_mask(self.board)

            # Expand dimensions for batch size
            state_input_expanded = np.expand_dims(state_input, axis=0)  # Shape: (1, 64)
            mask_input_expanded = np.expand_dims(mask, axis=0)  # Shape: (1, num_moves)

            # Prepare model inputs
            model_inputs = {'input_seq': state_input_expanded, 'mask_input': mask_input_expanded}
            # Get action probabilities
            action_probs = opponent_model.predict(model_inputs, verbose=0).ravel()
            # Choose an action based on the probabilities
            action_index = np.random.choice(NUM_MOVES, p=action_probs)
            opponent_move = decode_action(action_index)
            # Ensure the move is legal
            if opponent_move in self.board.legal_moves:
                self.board.push(opponent_move)
            else:
                # In rare cases, the chosen move might be illegal
                # Choose a random legal move as a fallback
                legal_moves = list(self.board.legal_moves)
                opponent_move = random.choice(legal_moves)
                self.board.push(opponent_move)

    def get_material_balance(self):
        # Compute material from the agent's perspective
        material = 0
        for piece in self.board.piece_map().values():
            value = self.get_piece_value(piece)
            if piece.color == self.agent_color:
                material += value
            else:
                material -= value
        return material

    def get_board_total_value(self):
        # Sum of all pieces' values on the board
        total_value = sum(self.get_piece_value(piece) for piece in self.board.piece_map().values())
        return total_value

    def get_capture_value(self, move):
        # Return the value of the captured opponent's piece
        if self.board.is_capture(move):
            captured_square = move.to_square
            captured_piece = self.board.piece_at(captured_square)
            if captured_piece and captured_piece.color != self.agent_color:
                return self.get_piece_value(captured_piece)
        return 0

    def get_lost_piece_value(self):
        # Calculate if the agent lost a piece due to the opponent's last move
        current_board_value = self.get_board_total_value()
        lost_value = self.previous_board_total_value - current_board_value
        self.previous_board_total_value = current_board_value
        if lost_value > 0:
            return lost_value
        return 0

    def get_development_reward(self):
        # Encourage development of pieces and castling
        reward = 0
        # Check if the agent has castled
        if self.has_castled():
            reward += 0.5  # Reward for castling
        # Reward for developing minor pieces
        developed_pieces = self.count_developed_pieces()
        reward += developed_pieces * 0.1  # Adjust scaling as needed
        return reward

    def has_castled(self):
        # Check if the agent has castled
        if self.agent_color == chess.WHITE:
            king_start_square = chess.E1
        else:
            king_start_square = chess.E8

        # Get the king's current square
        king_square = self.board.king(self.agent_color)

        # If the king is not on the starting square
        if king_square != king_start_square:
            # Check if the king has moved two squares horizontally (castling move)
            if abs(chess.square_file(king_square) - chess.square_file(king_start_square)) == 2:
                return True
        return False

    def count_developed_pieces(self):
        # Count the number of minor pieces (knights and bishops) that have been moved from their starting positions
        starting_positions = {
            chess.WHITE: [chess.B1, chess.G1, chess.C1, chess.F1],
            chess.BLACK: [chess.B8, chess.G8, chess.C8, chess.F8],
        }
        developed_pieces = 0
        for square in starting_positions[self.agent_color]:
            piece = self.board.piece_at(square)
            if not piece or piece.color != self.agent_color or piece.piece_type not in [chess.KNIGHT, chess.BISHOP]:
                developed_pieces += 1
        return developed_pieces

    def get_piece_value(self, piece):
        # Assign values to pieces
        values = {'p': 1.0, 'n': 3.0, 'b': 3.0, 'r': 5.0, 'q': 9.0, 'k': 0.0}
        return values[piece.symbol().lower()]

    def get_game_result_reward(self, result):
        if (result == '1-0' and self.agent_color == chess.WHITE) or \
           (result == '0-1' and self.agent_color == chess.BLACK):
            return 10  # Agent wins
        elif result == '1/2-1/2':
            return 0  # Draw
        else:
            return -10  # Agent loses

    def get_result_string(self, result):
        if result == '1-0':
            return 'white wins'
        elif result == '0-1':
            return 'black wins'
        else:
            return 'draw'

    def legal_moves(self):
        return list(self.board.legal_moves)