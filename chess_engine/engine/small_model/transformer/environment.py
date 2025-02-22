import chess
import numpy as np
NUM_MOVES = 20480
###############################################################################
# Move Encoding/Decoding
###############################################################################
def move_to_index(move: chess.Move) -> int:
    """
    Convert chess.Move into an index that also encodes promotions.
    We use the scheme:
       index = from_square*64*5 + to_square*5 + promo_code
    where promo_code = 0 for no promotion, 1=N, 2=B, 3=R, 4=Q.
    """
    promo_code = 0
    if move.promotion:  # If it's a promotion move
        if move.promotion == chess.KNIGHT:
            promo_code = 1
        elif move.promotion == chess.BISHOP:
            promo_code = 2
        elif move.promotion == chess.ROOK:
            promo_code = 3
        elif move.promotion == chess.QUEEN:
            promo_code = 4

    return move.from_square * 64 * 5 + move.to_square * 5 + promo_code

def index_to_move(move_idx: int, board: chess.Board) -> chess.Move:
    """
    Convert our expanded index (with promotions) back to a chess.Move.
    We first parse (from_sq, to_sq, promo_code).
    Then we construct the move with or without promotion.
    """
    from_sq = (move_idx // (64*5))
    remainder = move_idx % (64*5)
    to_sq = remainder // 5
    promo_code = remainder % 5

    promotion_piece = None
    if promo_code == 1:
        promotion_piece = chess.KNIGHT
    elif promo_code == 2:
        promotion_piece = chess.BISHOP
    elif promo_code == 3:
        promotion_piece = chess.ROOK
    elif promo_code == 4:
        promotion_piece = chess.QUEEN

    # Construct the move
    return chess.Move(from_sq, to_sq, promotion=promotion_piece)

###############################################################################
# Board Encoding
###############################################################################
def encode_board(board: chess.Board) -> np.ndarray:

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
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    for move in board.legal_moves:
        idx = move_to_index(move)
        if 0 <= idx < NUM_MOVES:
            mask[idx] = 1.0
    return mask

###############################################################################
# Root Dirichlet Noise Injection
###############################################################################
def inject_dirichlet_noise(root_node, alpha=0.03, eps=0.25):

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

