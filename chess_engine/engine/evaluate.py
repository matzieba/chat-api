import chess
import chess.engine
import numpy as np
import tensorflow as tf

from scipy.special import softmax



MODEL_PATH = "/chess_engine/engine/models/chess_rl_model_1000games.keras"
STOCKFISH_PATH = "/Users/mateuszzieba/Desktop/dev/chess/stockfish/src/stockfish"

def load_chess_model(model_path):
    return tf.keras.models.load_model(model_path)


def get_model_move(board, model):
    state = convert_board_to_state(board)
    state_input = np.expand_dims(state, axis=0)  # Add batch dimension
    state_input = np.expand_dims(state_input, axis=-1)  # Add channel dimension

    logits = model(state_input).numpy().ravel()
    legal_moves = list(board.legal_moves)

    # Create a mask for valid moves
    move_uci_list = [move.uci() for move in legal_moves]
    mask = np.zeros(logits.size)
    valid_move_indices = [i for i, move_uci in enumerate(move_uci_list)]
    mask[valid_move_indices] = 1

    # Compute probabilities only for valid moves
    masked_logits = logits * mask
    probabilities = softmax(masked_logits)

    # Choose the move index with the highest probability
    best_move_index = np.argmax(probabilities)
    return legal_moves[best_move_index]


def convert_board_to_state(board):
    board_state = np.zeros((8, 8), dtype=np.int32)
    piece_types = {
        chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
        chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            multiplier = 1 if piece.color == chess.WHITE else -1
            board_state[chess.square_rank(square), chess.square_file(square)] = multiplier * piece_types[
                piece.piece_type]

    return board_state


def evaluate_model_against_stockfish(num_games=5):
    model = load_chess_model(MODEL_PATH)
    stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    model_wins = 0
    stockfish_wins = 0
    draws = 0
    game_no  = 1
    for _ in range(num_games):
        print(f'playing game no {game_no}')
        game_no += 1
        move_count = 0
        board = chess.Board()
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = get_model_move(board, model)
                board.push(move)
            else:
                result = stockfish.play(board, chess.engine.Limit(time=1.0))
                board.push(result.move)
            move_count += 1

        result = board.result()
        if result == '1-0':
            model_wins += 1
        elif result == '0-1':
            stockfish_wins += 1
            print(f'stockfish wins using {move_count} moves')
        else:
            draws += 1

    stockfish.quit()

    print(f"Model Wins: {model_wins}")
    print(f"Stockfish Wins: {stockfish_wins}")
    print(f"Draws: {draws}")

evaluate_model_against_stockfish()