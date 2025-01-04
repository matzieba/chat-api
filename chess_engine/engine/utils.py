import chess
import numpy as np

# Constants
NUM_SQUARES = 64
NUM_PROMOTION_PIECES = 4  # Queen, Rook, Bishop, Knight
NUM_PROMOTION_OPTIONS = 5  # No promotion + 4 promotion pieces
PROMOTION_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

# Piece to ID mapping for plane representation
PIECE_TO_ID = {
    None: 0,  # Empty square
    'P': 1,
    'N': 2,
    'B': 3,
    'R': 4,
    'Q': 5,
    'K': 6,
    'p': 7,
    'n': 8,
    'b': 9,
    'r': 10,
    'q': 11,
    'k': 12,
}

# Initialize move encoding
MOVE_TO_INDEX = {}
INDEX_TO_MOVE = {}
NUM_MOVES = 0  # Will be set after move encoding

def initialize_move_encoding():
    index = 0
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            # Generate all possible promotions, including None (no promotion)
            for promotion in PROMOTION_PIECES:
                move = chess.Move(from_square, to_square, promotion=promotion)
                # Check if the move is legal in at least one position
                # Since we cannot check legality without a board, we include all moves
                # and filter illegal ones later when generating the mask
                move_uci = move.uci()
                MOVE_TO_INDEX[move_uci] = index
                INDEX_TO_MOVE[index] = move
                index += 1
    return index  # Return the total number of moves

NUM_MOVES = initialize_move_encoding()  # Set NUM_MOVES after initializing move encoding

def board_to_planes(board):
    """Convert the board to an 8x8x14 plane representation."""
    planes = np.zeros((8, 8, 14), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        piece_type = piece.symbol()
        plane_index = PIECE_TO_ID[piece_type]
        row = 7 - (square // 8)  # Flip the board vertically (optional)
        col = square % 8
        planes[row][col][plane_index] = 1.0

    # Side to move plane
    planes[:, :, 12] = float(board.turn)  # 1.0 if white to move, 0.0 if black

    # Castling rights plane
    castling_rights = [
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK)),
    ]
    castling_plane_value = sum(castling_rights) / 4.0  # Normalize between 0 and 1
    planes[:, :, 13] = castling_plane_value

    return planes

def encode_move(move):
    """Encode a move into an action index."""
    move_uci = move.uci()
    if move_uci in MOVE_TO_INDEX:
        return MOVE_TO_INDEX[move_uci]
    else:
        raise ValueError(f"Move {move_uci} cannot be encoded.")

def decode_action(action_index):
    """Decode an action index back to a move."""
    if action_index in INDEX_TO_MOVE:
        return INDEX_TO_MOVE[action_index]
    else:
        raise ValueError(f"Action index {action_index} cannot be decoded.")

def get_legal_moves_mask(board):
    """Get a mask of legal moves for the current board state."""
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    for move in board.legal_moves:
        try:
            move_index = encode_move(move)
            mask[move_index] = 1.0
        except ValueError:
            continue  # Skip if move cannot be encoded
    return mask