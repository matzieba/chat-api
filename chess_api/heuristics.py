import chess
import numpy as np

from chess_api.mcts import run_mcts_batched
from chess_engine.engine.train_supervised.parse_pgn_alpha0 import move_to_index

###############################################################################
#  Caching dictionaries to avoid re-computing repeated positions
###############################################################################
_evaluation_cache = {}
_pinned_cache = {}

###############################################################################
#  Piece Values (Simplified) for Michniewski's approach
#  plus piece-square tables from White's perspective
###############################################################################
PIECE_BASE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000
}

# Commonly used piece-square tables (64 entries) for White.
# Index 0 corresponds to A8, and 63 to H1 (row-major from top to bottom).
# For a black piece, we'll mirror the square rank to read the correct row, then negate.

PAWN_TABLE = [
     0,   0,   0,   0,   0,   0,   0,   0,
     5,   5,   5,  -5,  -5,   5,   5,   5,
     5,   5,  10,   0,   0,  10,   5,   5,
     0,   0,   0,  20,  20,   0,   0,   0,
     5,   5,   5,  10,  10,   5,   5,   5,
    10,  10,  10,  20,  20,  10,  10,  10,
    50,  50,  50,  50,  50,  50,  50,  50,
     0,   0,   0,   0,   0,   0,   0,   0
]

KNIGHT_TABLE = [
   -50, -40, -30, -30, -30, -30, -40, -50,
   -40, -20,   0,   0,   0,   0, -20, -40,
   -30,   0,  10,  15,  15,  10,   0, -30,
   -30,   5,  15,  20,  20,  15,   5, -30,
   -30,   0,  15,  20,  20,  15,   0, -30,
   -30,   5,  10,  15,  15,  10,   5, -30,
   -40, -20,   0,   5,   5,   0, -20, -40,
   -50, -40, -30, -30, -30, -30, -40, -50
]

BISHOP_TABLE = [
   -20, -10, -10, -10, -10, -10, -10, -20,
   -10,   5,   0,   0,   0,   0,   5, -10,
   -10,  10,  10,  10,  10,  10,  10, -10,
   -10,   0,  10,  10,  10,  10,   0, -10,
   -10,   5,   5,  10,  10,   5,   5, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -20, -10, -10, -10, -10, -10, -10, -20
]

ROOK_TABLE = [
    0,   0,   0,   0,   0,   0,   0,   0,
    5,  10,  10,  10,  10,  10,  10,   5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
    0,   0,   0,   5,   5,   0,   0,   0
]

QUEEN_TABLE = [
   -20, -10, -10,  -5,  -5, -10, -10, -20,
   -10,   0,   5,   0,   0,   0,   0, -10,
   -10,   5,   5,   5,   5,   5,   0, -10,
    -5,   0,   5,   5,   5,   5,   0,  -5,
    -5,   0,   5,   5,   5,   5,   0,  -5,
   -10,   5,   5,   5,   5,   5,   0, -10,
   -10,   0,   5,   0,   0,   0,   0, -10,
   -20, -10, -10,  -5,  -5, -10, -10, -20
]

KING_TABLE = [
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
   -20, -30, -30, -40, -40, -30, -30, -20,
   -10, -20, -20, -20, -20, -20, -20, -10,
    20,  20,   0,   0,   0,   0,  20,  20,
    20,  30,  10,   0,   0,  10,  30,  20
]

PIECE_SQ_TABLES = {
    chess.PAWN:   PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK:   ROOK_TABLE,
    chess.QUEEN:  QUEEN_TABLE,
    chess.KING:   KING_TABLE
}

###############################################################################
#  Helper: Flip the rank for black pieces
###############################################################################
def mirror_square(sq: chess.Square) -> chess.Square:
    """
    If the square is on rank r, mirror it to rank 7-r for use with White's tables.
    """
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    return chess.square(file, 7 - rank)

###############################################################################
#  Simplified Evaluation Function (SEF) per Tomasz Michniewski, White perspective
###############################################################################
def simplified_eval_michniewski_white(board: chess.Board) -> float:
    """
    Sums base piece values + piece-square table offsets from White's perspective.
    Positive => better for White, negative => better for Black.
    """
    score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        piece_type = piece.piece_type
        base_val = PIECE_BASE_VALUES.get(piece_type, 0)
        table = PIECE_SQ_TABLES.get(piece_type, None)
        if table is None:
            continue

        # White perspective => if piece is black, multiply by -1
        sign = 1.0 if piece.color == chess.WHITE else -1.0

        # If black piece => we mirror the square so we index White’s table from black’s viewpoint
        sq_index = sq
        if piece.color == chess.BLACK:
            sq_index = mirror_square(sq)

        # Convert square (0..63) from top-left = A8 to bottom-right = H1
        # python-chess square 0 => a1, but our table index 0 => a8
        # We can transform: index_in_table = 8*(7 - rank) + file
        # but python-chess “square” is file + 8*rank with rank=0=1st rank
        # So let's do it by direct approach:
        rank = 7 - chess.square_rank(sq_index)
        file = chess.square_file(sq_index)
        index_in_table = rank * 8 + file

        piece_sq_offset = table[index_in_table]
        piece_score = sign * (base_val + piece_sq_offset)
        score += piece_score

    return score

###############################################################################
#  The 10 Heuristics from White perspective
###############################################################################
PIECE_VALUES_OUR = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

def material_score_white(board: chess.Board) -> float:
    mat = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is not None:
            val = PIECE_VALUES_OUR.get(p.piece_type, 0)
            mat += val if p.color == chess.WHITE else -val
    return float(mat)

def squares_around(square: chess.Square):
    if square is None:
        return []
    f = chess.square_file(square)
    r = chess.square_rank(square)
    res = []
    for df in [-1,0,1]:
        for dr in [-1,0,1]:
            if df==0 and dr==0:
                continue
            nf = f+df
            nr = r+dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                res.append(chess.square(nf, nr))
    return res

def king_safety_score_white(board: chess.Board) -> float:
    score = 0.0
    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)
    if w_king:
        wk_neighbors = squares_around(w_king)
        attacked_w = sum(1 for sq in wk_neighbors if board.is_attacked_by(chess.BLACK, sq))
        score -= 0.1 * attacked_w
    if b_king:
        bk_neighbors = squares_around(b_king)
        attacked_b = sum(1 for sq in bk_neighbors if board.is_attacked_by(chess.WHITE, sq))
        score += 0.1 * attacked_b
    return score

def center_control_score_white(board: chess.Board) -> float:
    wc = 0
    bc = 0
    for sq in CENTER_SQUARES:
        if board.is_attacked_by(chess.WHITE, sq):
            wc += 1
        if board.is_attacked_by(chess.BLACK, sq):
            bc += 1
    return 0.05 * (wc - bc)

def mobility_score_white(board: chess.Board) -> float:
    # We'll measure how many moves White vs. Black could have if it were their turn
    white_moves = len(_possible_moves_for_color(board, chess.WHITE))
    black_moves = len(_possible_moves_for_color(board, chess.BLACK))
    return 0.01 * (white_moves - black_moves)

def _possible_moves_for_color(board: chess.Board, color: bool):
    temp = board.copy()
    temp.turn = color
    return list(temp.legal_moves)

def passed_pawns_score_white(board: chess.Board) -> float:
    w_passed = _count_passed_pawns(board, chess.WHITE)
    b_passed = _count_passed_pawns(board, chess.BLACK)
    return 0.02 * (w_passed - b_passed)

def _count_passed_pawns(board: chess.Board, color: bool) -> int:
    pawns = board.pieces(chess.PAWN, color)
    return sum(1 for sq in pawns if _is_passed_pawn(board, sq, color))

def _is_passed_pawn(board: chess.Board, sq: chess.Square, color: bool) -> bool:
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    direction = 1 if color == chess.WHITE else -1
    opp = not color
    test_rank = rank + direction
    while 0 <= test_rank < 8:
        for df in [-1,0,1]:
            nf = file+df
            if 0 <= nf < 8:
                check_sq = chess.square(nf, test_rank)
                piece = board.piece_at(check_sq)
                if piece and piece.color == opp and piece.piece_type == chess.PAWN:
                    return False
        test_rank += direction
    return True

def rook_open_file_score_white(board: chess.Board) -> float:
    w_rooks = board.pieces(chess.ROOK, chess.WHITE)
    b_rooks = board.pieces(chess.ROOK, chess.BLACK)
    wo = sum(1 for r in w_rooks if not _file_has_pawn(board, chess.square_file(r)))
    bo = sum(1 for r in b_rooks if not _file_has_pawn(board, chess.square_file(r)))
    return 0.01 * (wo - bo)

def _file_has_pawn(board: chess.Board, file: int) -> bool:
    for rank in range(8):
        piece = board.piece_at(chess.square(file, rank))
        if piece and piece.piece_type == chess.PAWN:
            return True
    return False

def development_score_white(board: chess.Board) -> float:
    w_minors = board.pieces(chess.KNIGHT, chess.WHITE) | board.pieces(chess.BISHOP, chess.WHITE)
    b_minors = board.pieces(chess.KNIGHT, chess.BLACK) | board.pieces(chess.BISHOP, chess.BLACK)
    wd = sum(1 for sq in w_minors if chess.square_rank(sq) != 0)
    bd = sum(1 for sq in b_minors if chess.square_rank(sq) != 7)
    return 0.01 * (wd - bd)

def check_score_white(board: chess.Board) -> float:
    if board.is_game_over():
        return 0.0
    score = 0.0
    if board.is_check():
        # if black to move => black is in check => good for white
        # if white to move => white is in check => bad for white
        if board.turn == chess.BLACK:
            score += 0.03
        else:
            score -= 0.03
    return score

def checkmate_score_white(board: chess.Board) -> float:
    if board.is_game_over():
        res = board.result()
        if res == "1-0":
            return 100.0
        elif res == "0-1":
            return -100.0
        else:
            return 0.0
    return 0.0

def pinned_move_penalty(board: chess.Board, move: chess.Move, side: bool) -> float:
    fen_before = board.fen()
    key = (fen_before, move.uci(), side)
    if key in _pinned_cache:
        return _pinned_cache[key]

    temp = board.copy()
    temp.turn = side
    if temp.is_pinned(side, move.from_square):
        val = -0.02
    else:
        val = 0.0
    _pinned_cache[key] = val
    return val

###############################################################################
#  Overall "other heuristics" aggregator (White perspective)
###############################################################################
def other_heuristics_white(board: chess.Board) -> float:
    val = 0.0
    val += material_score_white(board)
    val += king_safety_score_white(board)
    val += center_control_score_white(board)
    val += mobility_score_white(board)
    val += passed_pawns_score_white(board)
    val += rook_open_file_score_white(board)
    val += development_score_white(board)
    val += check_score_white(board)
    val += checkmate_score_white(board)
    return val

###############################################################################
#  Evaluate position from side's perspective:
#  final = 0.5 * SimplifiedEvalMichniewski + 0.5 * other_heuristics
###############################################################################
def evaluate_position_for_side(board: chess.Board, side: bool) -> float:
    """
    side=True => White to move, side=False => Black to move.
    We'll compute everything from White perspective, then flip if side=Black.
    """
    fen_key = (board.fen(), side)
    if fen_key in _evaluation_cache:
        return _evaluation_cache[fen_key]

    # 1) Michniewski's SEF (White perspective)
    white_eval_mich = simplified_eval_michniewski_white(board)
    # 2) Our other heuristics (White perspective)
    white_eval_ours = other_heuristics_white(board)

    # Weighted average
    white_eval = 0.5 * white_eval_mich + 0.5 * white_eval_ours

    # If it’s black to move, flip sign
    sign = 1.0 if side == chess.WHITE else -1.0
    final_eval = sign * white_eval

    _evaluation_cache[fen_key] = final_eval
    return final_eval

###############################################################################
#  Move-level heuristic: how does a move affect the position eval for 'side'?
###############################################################################
def compute_heuristic_factor(board: chess.Board, move: chess.Move, side: bool) -> float:
    """
    The bigger the factor, the better for 'side' according to our combined heuristics.
    We'll look at (eval_after - eval_before) + pinned penalty, then exponentiate.
    """
    old_eval = evaluate_position_for_side(board, side)
    pin_pen = pinned_move_penalty(board, move, side)

    temp = board.copy()
    temp.turn = side
    temp.push(move)

    # After 'side' moves, it's the opponent's turn => not side
    new_eval = evaluate_position_for_side(temp, not side)

    delta = (new_eval - old_eval) + pin_pen
    scale = 0.1
    factor = float(np.exp(scale * delta))
    return factor

###############################################################################
#  HeuristicsEngine - orchestrates MCTS + combined heuristics for both sides
###############################################################################
class HeuristicsEngine:
    """
    Production-style class that runs MCTS and then re-ranks the moves using:
      final_score = (MCTS_probability) * (heuristic_factor).
    You can pick argmax or sample from this distribution.
    Caches are used for repeated position evaluations.
    """

    def __init__(self,
                 model,
                 n_mcts_sims=1000,
                 mcts_batch_size=512,
                 temperature=0.5,
                 deterministic=True):
        """
        :param model: your Keras (AlphaZero-like) policy/value model
        :param n_mcts_sims: number of MCTS rollouts per move
        :param mcts_batch_size: batch size for MCTS expansions
        :param temperature: MCTS temperature. 0 => argmax, 1 => sample
        :param deterministic: if True, pick argmax after weighting, else sample
        """
        self.model = model
        self.n_mcts_sims = n_mcts_sims
        self.mcts_batch_size = mcts_batch_size
        self.temperature = temperature
        self.deterministic = deterministic

    def get_move(self, board: chess.Board) -> chess.Move:
        side_to_move = board.turn

        best_move, distribution = run_mcts_batched(
            model=self.model,
            root_board=board,
            n_simulations=self.n_mcts_sims,
            batch_size=self.mcts_batch_size,
            temperature=self.temperature,
            add_dirichlet_noise=False
        )

        # If MCTS gave no move, fallback
        if best_move is None or distribution is None:
            return best_move

        # Build a list of (move, mcts_prob)
        legal_moves = list(board.legal_moves)
        moves_probs = []
        for mv in legal_moves:
            idx = move_to_index(mv)
            mcts_p = distribution[idx]
            moves_probs.append((mv, mcts_p))

        # Apply heuristics
        weighted_scores = []
        for (mv, prob) in moves_probs:
            h_factor = compute_heuristic_factor(board, mv, side_to_move)
            weighted_scores.append(prob * h_factor)

        total = sum(weighted_scores)
        if total < 1e-12:
            # fallback if everything is near zero
            return best_move

        # Re-normalize
        final_dist = [s / total for s in weighted_scores]

        if self.deterministic:
            # Pick argmax
            chosen_index = int(np.argmax(final_dist))
            final_move = legal_moves[chosen_index]
        else:
            # Sample from final_dist
            chosen_index = np.random.choice(len(legal_moves), p=final_dist)
            final_move = legal_moves[chosen_index]

        return final_move