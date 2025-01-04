# We'll define a simplistic move space:
#   index = from_sq * 64 * 5 + to_sq * 5 + prom
# where:
#   from_sq, to_sq ∈ [0..63]
#   prom ∈ {0 (none), 1 (knight), 2 (bishop), 3 (rook), 4 (queen)}
#
# That yields a maximum of 64*64*5 = 20,480. Many indices won't be legal in actual positions.
# We'll mask those out with our "mask_input."
import chess
import numpy as np

NUM_MOVES = 64 * 64 * 5

PROMOTION_MAP = {
    None: 0,           # No promotion
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4
}
REVERSE_PROM = {v: k for k, v in PROMOTION_MAP.items()}

# Our board encoding will have shape (8, 8, 14).
# We'll define:
#   Channels 0..5: White Pawn, Knight, Bishop, Rook, Queen, King
#   Channels 6..11: Black Pawn, Knight, Bishop, Rook, Queen, King
#   Channel 12: side-to-move (1 if white-to-move, else 0)
#   Channel 13: simplistic castling rights encoding (bits for each of the 4 castling rights)
###############################################################################

###############################################################################
# 2) Helper Functions
###############################################################################
def move_to_index(move: chess.Move) -> int:
    """Convert a chess.Move into an integer index based on our simplistic scheme."""
    from_sq = move.from_square
    to_sq = move.to_square
    prom = PROMOTION_MAP.get(move.promotion, 0)
    return from_sq * (64 * 5) + to_sq * 5 + prom

def index_to_move(move_idx: int, board: chess.Board) -> chess.Move:
    """
    Inverse of move_to_index.
    If it's not legal for the current board, calling code must handle that.
    """
    prom = move_idx % 5
    tmp = move_idx // 5
    to_sq = tmp % 64
    from_sq = tmp // 64

    promotion_piece = REVERSE_PROM.get(prom, None)
    return chess.Move(from_sq, to_sq, promotion=promotion_piece)

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode the board into a (8, 8, 14) numpy array.
    Channels:
      0..5   = (White Pawn, Knight, Bishop, Rook, Queen, King)
      6..11  = (Black Pawn, Knight, Bishop, Rook, Queen, King)
      12     = side to move (1 if white, else 0) (broadcast across the board)
      13     = castling rights bits (simple or-ed approach, broadcast)
    """
    # Initialize all zeros
    encoded = np.zeros((8, 8, 14), dtype=np.float32)

    piece_type_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            row = 7 - (sq // 8)
            col = sq % 8
            base_channel = 0 if piece.color == chess.WHITE else 6
            ch = base_channel + piece_type_to_channel[piece.piece_type]
            encoded[row, col, ch] = 1.0

    # Side to move (channel 12)
    side_to_move_val = 1.0 if board.turn == chess.WHITE else 0.0
    encoded[..., 12] = side_to_move_val

    # Simplistic castling rights in channel 13
    castling_val = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_val += 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_val += 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_val += 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_val += 8
    encoded[..., 13] = float(castling_val)

    return encoded

def build_move_mask(board: chess.Board) -> np.ndarray:
    """
    Return a 0/1 mask of shape (NUM_MOVES,) indicating which moves are legal
    for the current board under our indexing scheme.
    """
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    for move in board.legal_moves:
        idx = move_to_index(move)
        if idx < NUM_MOVES:
            mask[idx] = 1.0
    return mask

def inject_dirichlet_noise(root_node, alpha=0.03, eps=0.25):
    """
    For the root node, blend its children’s P-values with Dirichlet noise
    to encourage exploration.
    root_node.children[move].P is each child’s prior probability.
    alpha ~ 0.03, eps ~ 0.25 are typical for chess.
    """
    moves = list(root_node.children.keys())
    if not moves:
        return  # No children -> nothing to do

    # Extract current root priors
    old_ps = np.array([root_node.children[m].P for m in moves], dtype=np.float32)
    # Normalize if needed
    sum_ps = np.sum(old_ps)
    if sum_ps > 1e-9:
        old_ps /= sum_ps
    else:
        # If sum is ~0, just use uniform
        old_ps = np.ones_like(old_ps) / len(old_ps)

    # Sample from Dirichlet distribution
    dirichlet_noise = np.random.dirichlet([alpha] * len(moves))

    # Blend
    new_ps = (1 - eps) * old_ps + eps * dirichlet_noise

    # Assign back
    for m, p in zip(moves, new_ps):
        root_node.children[m].P = p

def apply_temperature(policy, temperature=1.0):
    """
    Apply temperature to a probability distribution:
      pᵢ^(1/temperature) / sum_j [pⱼ^(1/temperature)].
    If temperature is nearly 0, this approximates an argmax.
    """
    if temperature < 1e-9:
        # Argmax selection
        out = np.zeros_like(policy)
        out[np.argmax(policy)] = 1.0
        return out
    # Raise the distribution to the power (1/tau) for soft sampling
    adjusted = np.power(policy, 1.0 / temperature)
    adjusted_sum = np.sum(adjusted)
    if adjusted_sum < 1e-9:
        # Fallback to uniform if it's all near-zero
        return np.ones_like(policy) / len(policy)
    return adjusted / adjusted_sum