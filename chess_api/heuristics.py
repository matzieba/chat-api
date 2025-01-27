import chess
import numpy as np

NUM_SQUARES = 64

# ------------------------------------------------------------------------------
# 1) MATERIAL VALUES
# ------------------------------------------------------------------------------
MATERIAL_VALUES = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK:   5.0,
    chess.QUEEN:  15.0,
    chess.KING:   0.0  # King is invaluable, but set to 0 for code consistency
}

# ------------------------------------------------------------------------------
# 2) PIECE-SQUARE TABLES (PST)
#
# Each array has 64 entries: index 0..63 corresponds to squares
# a1..h1, a2..h2, ..., a8..h8 (from White’s perspective).
# These are typical “decent” but simplified PSTs used by many chess programs.
#
# Values are in “PST points.” Some engines use centipawns; here we use a
# mid-range float, e.g. [-5..+5]. We’ll scale them later in the aggregator.
# ------------------------------------------------------------------------------
PAWN_PST = [
    # Rank 1 (a1..h1)
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    # Rank 2 (a2..h2)
    0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
    # Rank 3 (a3..h3)
    0.0,  0.0,  0.2,  0.3,  0.3,  0.2,  0.0,  0.0,
    # Rank 4 (a4..h4)
    0.0,  0.0,  0.3,  0.5,  0.5,  0.3,  0.0,  0.0,
    # Rank 5 (a5..h5)
    0.0,  0.0,  0.3,  0.5,  0.5,  0.3,  0.0,  0.0,
    # Rank 6 (a6..h6)
    0.0, -0.2, -0.2,  0.0,  0.0, -0.2, -0.2,  0.0,
    # Rank 7 (a7..h7)
    0.0,  0.0,  0.0, -0.5, -0.5,  0.0,  0.0,  0.0,
    # Rank 8 (a8..h8)
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
]

KNIGHT_PST = [
    # Rank 1
    -2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0,
    # Rank 2
    -1.5, -1.0,  0.0,  0.0,  0.0,  0.0, -1.0, -1.5,
    # Rank 3
    -1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0,
    # Rank 4
    -1.0,  0.5,  1.0,  1.5,  1.5,  1.0,  0.5, -1.0,
    # Rank 5
    -1.0,  0.5,  1.0,  1.5,  1.5,  1.0,  0.5, -1.0,
    # Rank 6
    -1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0,
    # Rank 7
    -1.5, -1.0,  0.0,  0.0,  0.0,  0.0, -1.0, -1.5,
    # Rank 8
    -2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0
]

BISHOP_PST = [
    # Rank 1
    -1.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -1.0,
    # Rank 2
    -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5,
    # Rank 3
    -0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5,
    # Rank 4
    -0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5, -0.5,
    # Rank 5
    -0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5,
    # Rank 6
    -0.5,  0.5,  1.0,  1.0,  1.0,  1.0,  0.5, -0.5,
    # Rank 7
    -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5,
    # Rank 8
    -1.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -1.0
]

ROOK_PST = [
    # Rank 1
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    # Rank 2
     0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.5,
    # Rank 3
    -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5,
    # Rank 4
    -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5,
    # Rank 5
    -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5,
    # Rank 6
    -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5,
    # Rank 7
    -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5,
    # Rank 8
     0.0,  0.0,  0.0,  0.5,  0.5,  0.0,  0.0,  0.0
]

QUEEN_PST = [
    # Rank 1
    -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0,
    # Rank 2
    -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,
    # Rank 3
    -1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0,
    # Rank 4
    -0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5,
    # Rank 5
    -0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5,
    # Rank 6
    -1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0,
    # Rank 7
    -1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0,
    # Rank 8
    -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0
]

KING_PST = [
    # Rank 1
    -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0,
    # Rank 2
    -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0,
    # Rank 3
    -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0,
    # Rank 4
    -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0,
    # Rank 5
    -2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0,
    # Rank 6
    -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0,
    # Rank 7
     2.0,  2.0,  0.0,  0.0,  0.0,  0.0,  2.0,  2.0,
    # Rank 8
     2.0,  3.0,  1.0,  0.0,  0.0,  1.0,  3.0,  2.0
]

PIECE_SQUARE_TABLES = {
    chess.PAWN:   PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK:   ROOK_PST,
    chess.QUEEN:  QUEEN_PST,
    chess.KING:   KING_PST
}

# ------------------------------------------------------------------------------
# 3) INDIVIDUAL HEURISTICS
# ------------------------------------------------------------------------------

def material_evaluation(board: chess.Board) -> float:
    """
    Basic material counting: sum(white) - sum(black).
    Normalized to ~[-1..+1] for mid-late game.
    """
    value = 0.0
    for piece_type, mat_val in MATERIAL_VALUES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        value += white_count * mat_val
        value -= black_count * mat_val

    # 39 covers typical maximum total (both sides) in the opening (1+3+3+5+9 =21,
    # but up to 39 is a common scale so that mid-late game fits in [-1..+1].
    return value / 39.0

def piece_square_evaluation(board: chess.Board) -> float:
    """
    Lookup piece-square tables for every piece on the board.
    For White on square s: +table[s].
    For Black on square s: -table[s].
    Then scale so typical values remain small.
    """
    value = 0.0
    for piece_type, table in PIECE_SQUARE_TABLES.items():
        for square in board.pieces(piece_type, chess.WHITE):
            value += table[square]
        for square in board.pieces(piece_type, chess.BLACK):
            value -= table[square]
    # PSTs are roughly in the range [-5..+5], so dividing by 100 keeps the total small.
    return value / 100.0

def pawn_structure_evaluation(board: chess.Board) -> float:
    """
    Simple pawn structure: penalize isolated pawns.
    You could also consider doubled pawns, backward pawns, etc.
    """
    isolated_pawn_penalty = -0.5
    pawn_isolation_score = 0.0
    pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)

    for sq in pawns:
        color = board.color_at(sq)
        file_ = chess.square_file(sq)
        rank_ = chess.square_rank(sq)
        neighbors = []
        if file_ > 0:
            neighbors.append(chess.square(file_ - 1, rank_))
        if file_ < 7:
            neighbors.append(chess.square(file_ + 1, rank_))

        # Check if there's a same-color pawn on either neighboring file in the same rank
        is_isolated = True
        for nb_sq in neighbors:
            if nb_sq in pawns and board.color_at(nb_sq) == color:
                is_isolated = False
                break
        # If isolated, +1 if it's White, -1 if it's Black
        if is_isolated:
            pawn_isolation_score += 1.0 if color == chess.WHITE else -1.0

    # Multiply the count by a penalty, then scale
    return (isolated_pawn_penalty * pawn_isolation_score) / 8.0

def piece_evaluation_patterns(board: chess.Board) -> float:
    """
    Example pattern: bishop pair bonus, etc.
    Expand with other patterns (rooks on open files, knight outposts, etc.)
    """
    value = 0.0
    white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    # Simple bishop pair bonus
    if white_bishops >= 2:
        value += 0.2
    if black_bishops >= 2:
        value -= 0.2
    return value

def mobility_evaluation(board: chess.Board) -> float:
    """
    Count how many moves White can make vs. how many moves Black can make.
    Higher mobility => advantage.
    """
    original_turn = board.turn
    board.turn = chess.WHITE
    white_moves = len(list(board.legal_moves))
    board.turn = chess.BLACK
    black_moves = len(list(board.legal_moves))
    board.turn = original_turn

    total_moves = white_moves + black_moves + 1e-5
    return (white_moves - black_moves) / total_moves

def center_control(board: chess.Board) -> float:
    """
    Evaluate control of center squares (d4, d5, e4, e5) by counting
    how many attackers are on each square from each side.
    """
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    control = 0
    for sq in center_squares:
        control += len(list(board.attackers(chess.WHITE, sq)))
        control -= len(list(board.attackers(chess.BLACK, sq)))
    return control / 16.0

def connectivity_evaluation(board: chess.Board) -> float:
    """
    Simple measure of how many pieces are defended by friendly pieces.
    """
    value = 0.0
    for color in [chess.WHITE, chess.BLACK]:
        piece_squares = [sq for sq, _ in board.piece_map().items() if board.color_at(sq) == color]
        defended_count = 0
        for sq in piece_squares:
            defenders = set(board.attackers(color, sq))
            defenders.discard(sq)  # don't count self as defender
            if len(defenders) > 0:
                defended_count += 1
        if color == chess.WHITE:
            value += defended_count
        else:
            value -= defended_count
    return value / 16.0

def trapped_pieces_evaluation(board: chess.Board) -> float:
    """
    If a major/minor piece (N, B, R, Q) has zero moves, penalize it for being trapped.
    """
    penalty = 0.2
    piece_types = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    value = 0.0
    for color in [chess.WHITE, chess.BLACK]:
        for pt in piece_types:
            for sq in board.pieces(pt, color):
                if not any(move for move in board.legal_moves if move.from_square == sq):
                    if color == chess.WHITE:
                        value -= penalty
                    else:
                        value += penalty
    return value

def king_safety(board: chess.Board) -> float:
    """
    Count how many attackers threaten White's king vs. Black's king.
    """
    w_king_sq = board.king(chess.WHITE)
    b_king_sq = board.king(chess.BLACK)
    # If either king is missing, must be game over or illegal
    if w_king_sq is None or b_king_sq is None:
        return 0.0

    white_king_threat = len(list(board.attackers(chess.BLACK, w_king_sq)))
    black_king_threat = len(list(board.attackers(chess.WHITE, b_king_sq)))
    # If black's king is more threatened, result is positive
    return (black_king_threat - white_king_threat) / 8.0

def space_evaluation(board: chess.Board) -> float:
    """
    Compare how many squares White attacks vs. Black attacks across all 64 squares.
    """
    white_control = 0
    black_control = 0
    for sq in range(NUM_SQUARES):
        if board.is_attacked_by(chess.WHITE, sq):
            white_control += 1
        if board.is_attacked_by(chess.BLACK, sq):
            black_control += 1
    return (white_control - black_control) / 64.0

def tempo_evaluation(board: chess.Board) -> float:
    """
    +0.05 if White to move, -0.05 if Black to move.
    """
    return 0.05 if board.turn == chess.WHITE else -0.05

# ------------------------------------------------------------------------------
# 4) MORE NUANCED REPETITION PENALTY
#
#   If the position repeated >=3 times, and also check >=4 or 5 times:
#   For each repetition i in 3..5, apply an extra -0.2 * (i - 2).
#   So 3x repetition => -0.2, 4x => -0.4, 5x => -0.6, etc.
# ------------------------------------------------------------------------------
def repetition_penalty_evaluation(board: chess.Board) -> float:
    penalty_total = 0.0
    for i in range(3, 6):  # check 3x,4x,5x
        if board.is_repetition(i):
            penalty_total += -0.2 * (i - 2)
    return penalty_total

# ------------------------------------------------------------------------------
# 5) MASTER AGGREGATOR: Combine heuristics with chosen weights.
#    Adjust the multipliers so the final range is around [-2..+2] or so.
# ------------------------------------------------------------------------------
def evaluate_with_heuristics(board: chess.Board) -> float:
    mat   = 0.4 * material_evaluation(board)
    psts  = 0.2 * piece_square_evaluation(board)
    pawn  = 0.0 * pawn_structure_evaluation(board)
    piece = 0.05 * piece_evaluation_patterns(board)
    mob   = 0.10 * mobility_evaluation(board)
    ctr   = 0.05 * center_control(board)
    conn  = 0.05 * connectivity_evaluation(board)
    trap  = 0.00 * trapped_pieces_evaluation(board)
    king  = 0.1 * king_safety(board)
    spce  = 0.1 * space_evaluation(board)
    temp  = 0.00 * tempo_evaluation(board)

    # Heavier multiplier for the more nuanced repetition penalty:
    repetition = 0.5 * repetition_penalty_evaluation(board)

    total = (
        mat + psts + pawn + piece +
        mob + ctr + conn + trap +
        king + spce + temp + repetition
    )
    return total