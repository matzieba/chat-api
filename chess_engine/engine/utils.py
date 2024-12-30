import chess
import numpy as np

# Constants
NUM_SQUARES = 64
NUM_PROMOTION_PIECES = 4  # Queen, Rook, Bishop, Knight
NUM_PROMOTION_OPTIONS = 5  # No promotion + 4 promotion pieces
PROMOTION_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
PIECE_TO_ID = {
    None: 0,  # Empty square
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
}
NUM_MOVES = NUM_SQUARES * NUM_SQUARES * NUM_PROMOTION_OPTIONS  # 64 * 64 * 5 = 20,480

def convert_board_to_sequence(board):
    """Convert the board to a sequence of token IDs."""
    sequence = np.zeros(64, dtype=np.int32)
    for idx in range(64):
        piece = board.piece_at(idx)
        piece_symbol = piece.symbol() if piece else None
        token_id = PIECE_TO_ID[piece_symbol]
        sequence[idx] = token_id
    return sequence

def encode_move(move):
    """Encode a move into an action index."""
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion
    if promotion:
        promotion_index = PROMOTION_PIECES.index(promotion)
    else:
        promotion_index = 0  # No promotion is at index 0
    action_index = (from_square * NUM_SQUARES * NUM_PROMOTION_OPTIONS) + \
                   (to_square * NUM_PROMOTION_OPTIONS) + \
                   promotion_index
    return action_index

def decode_action(action_index):
    """Decode an action index back to a move."""
    promotion_index = action_index % NUM_PROMOTION_OPTIONS
    action_index //= NUM_PROMOTION_OPTIONS
    to_square = action_index % NUM_SQUARES
    from_square = action_index // NUM_SQUARES
    promotion = PROMOTION_PIECES[promotion_index]
    move = chess.Move(from_square=from_square, to_square=to_square, promotion=promotion)
    return move

def get_legal_moves_mask(board):
    """Get a mask of legal moves for the given board position."""
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    for move in board.legal_moves:
        try:
            action_index = encode_move(move)
            mask[action_index] = 1.0
        except (ValueError, IndexError):
            pass
    return mask