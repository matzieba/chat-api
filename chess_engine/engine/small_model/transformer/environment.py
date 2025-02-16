import chess
import numpy as np

# We reduce the move space to 64 * 64 = 4096 (ignore piece promotions).
NUM_MOVES = 64 * 64 * 2

###############################################################################
# Move Encoding/Decoding
###############################################################################
def move_to_index(move):
    from_sq = move.from_square
    to_sq = move.to_square
    base = from_sq * 64 + to_sq  # [0..4095]
    if move.promotion == chess.QUEEN:
        return 4096 + base
    else:
        return base


def index_to_move(move_idx: int, board: chess.Board) -> chess.Move:
    """
    Converts an integer in [0..8191] back into a chess.Move object.
    Skips underpromotions (only handles promotions to queen).
    """
    if move_idx < 4096:
        # Normal move
        from_sq = move_idx // 64
        to_sq   = move_idx % 64
        return chess.Move(from_sq, to_sq)
    else:
        # Queen promotion
        offset  = move_idx - 4096
        from_sq = offset // 64
        to_sq   = offset % 64
        return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

###############################################################################
# Board Encoding
###############################################################################
def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode the board into a shape (64, 14) numpy array.
    Each of the 64 squares gets 14 channels:
      Channels 0..5  = White Pawn, Knight, Bishop, Rook, Queen, King
      Channels 6..11 = Black Pawn, Knight, Bishop, Rook, Queen, King
      Channel 12     = side to move (1 if white, else 0)
      Channel 13     = castling rights bit-encoding
    """
    encoded = np.zeros((64, 14), dtype=np.float32)

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
            base_channel = 0 if piece.color == chess.WHITE else 6
            channel_offset = piece_type_to_channel[piece.piece_type]
            encoded[sq, base_channel + channel_offset] = 1.0

    # Side to move
    side_to_move_val = 1.0 if board.turn == chess.WHITE else 0.0
    encoded[:, 12] = side_to_move_val

    # Castling rights in channel 13
    castling_val = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_val += 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_val += 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_val += 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_val += 8
    encoded[:, 13] = float(castling_val)

    return encoded

###############################################################################
# Move Mask
###############################################################################
def build_move_mask(board: chess.Board) -> np.ndarray:
    """
    Return a 0/1 mask of shape (20,480) indicating which moves
    (from_sq->to_sq plus optional promotion) are legal.
    """
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = 1.0
    return mask

###############################################################################
# Root Dirichlet Noise Injection
###############################################################################
def inject_dirichlet_noise(root_node, alpha=0.03, eps=0.25):
    """
    Blend each child's 'prior' values with Dirichlet noise to encourage exploration.
    """
    import numpy as np

    moves = list(root_node.children.keys())
    if not moves:
        return

    # Gather existing priors from each child
    old_priors = np.array([root_node.children[m].prior for m in moves], dtype=np.float32)

    # Normalize if needed
    sum_ps = np.sum(old_priors)
    if sum_ps < 1e-9:
        old_priors = np.ones_like(old_priors) / len(old_priors)
    else:
        old_priors /= sum_ps

    # Generate Dirichlet noise
    noise = np.random.dirichlet([alpha] * len(moves))

    # Blend
    new_priors = (1 - eps) * old_priors + eps * noise

    # Assign blended priors back to children
    for m, p in zip(moves, new_priors):
        root_node.children[m].prior = p

###############################################################################
# Temperature Application
###############################################################################
def apply_temperature(policy, temperature=1.0):
    """
    Apply temperature to a probability distribution:
      pᵢ^(1/tau) / Σⱼ [pⱼ^(1/tau)].
    If temperature is near 0, this approximates argmax selection.
    """
    if temperature < 1e-9:
        out = np.zeros_like(policy)
        out[np.argmax(policy)] = 1.0
        return out
    adjusted = np.power(policy, 1.0 / temperature)
    adjusted_sum = np.sum(adjusted)
    if adjusted_sum < 1e-9:
        # fallback to uniform
        return np.ones_like(policy) / len(policy)
    return adjusted / adjusted_sum