import chess
import numpy as np

from environment import encode_single_board, move_to_index
from mcts import MCTSNode, terminal_value, backprop
from pst import piece_square_table_score_white

##############################################################################
# 1) Central dictionary of numeric constants for easy tuning
##############################################################################
PARAMS = {
    # Overall scaling for how strongly one move's eval difference affects MCTS weighting
    "SCALE_FACTOR": 0.8,

    # Penalty if a piece is pinned
    "PINNED_PENALTY": -0.03,

    # Base heuristics
    "KING_SAFETY": 0.15,        # penalty/bonus factor for squares attacked around kings
    "CENTER_CONTROL": 0.05,    # factor for controlling D4,E4,D5,E5
    "MOBILITY": 0.01,          # factor for difference in move counts
    "PASSED_PAWNS": 0.02,      # factor for difference in passed pawns
    "ROOK_OPEN_FILE": 0.01,    # bonus for rooks on open files
    "DEVELOPMENT": 0.05,       # reward for minor pieces leaving back rank
    "BISHOP_PAIR": 0.15,       # advantage or penalty if only one side has 2 bishops
    "PAWN_STRUCTURE": 0.02,    # difference in doubled/isolated pawn counts
    "CHECK_SCORE": 0.03,       # bonus if opponent in check, penalty if we are
    # advanced heuristics
    "KNIGHT_OUTPOST": 0.1,     # bonus for White knights in advanced squares not attacked by black pawns
    "ROOK_7TH": 0.1,           # bonus for rooks on 7th (white) / penalty if black rook on 2nd
    "KING_ENDGAME": 0.03,      # factor for king centralization in endgame
    "ROOKS_CONNECTED": 0.05,   # bonus/penalty for rooks connected
    "SPACE_ADVANTAGE": 0.01,   # factor for advanced pawns for White vs. Black
    # super advanced
    "PAWN_PROMO": 0.02,        # factor for pawn promotion potential
    "TRADE_PREF": 0.08,        # if White leads in material => encourage trades, if behind => discourage
    "CASTLE_BONUS": 0.07,      # small bonus if White's king is obviously castled
    "KING_SHIELD": 0.03,       # bonus for pawns in front of White's king on rank=1
    "PST_FACTOR": 0.10,
}

##############################################################################
# 2) Basic piece values
##############################################################################
PIECE_VALUES_OUR = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

_evaluation_cache = {}
_pinned_cache = {}

def draw_avoidance_score_white(board: chess.Board) -> float:
    """
    Encourages White to avoid drifting into 50-move-rule draws.
    The idea: if no pawns moved or pieces were captured in the last few moves,
    we give White a small incentive to 'mix things up'.
    If White is better, we want to push. If White is worse, we might want to
    keep it quiet. Modify to personal preference.
    """

    # Count halfmove clock (for the 50-move rule).
    # By default, board.halfmove_clock is how many halfmoves since last capture/pawn advance.
    halfmoves_since_progress = board.halfmove_clock

    # If halfmoves_since_progress is large, we are approaching 50 moves => a potential draw.
    # Let's add a small bonus (if White is not losing).
    # You can scale this up or down; it’s just an example.
    if halfmoves_since_progress >= 20:
        # Use White’s material lead as a sign we want to push.
        # If White is behind, maybe we are actually content with a draw, so do less.
        mat_score = material_score_white(board)
        if mat_score > 0.0:
            # If White is better, encourage it NOT to accept a draw by repetition or 50 moves
            return 0.05 * (halfmoves_since_progress - 19)  # grows as we get closer to 50

    return 0.0

##############################################################################
# 3) Base Heuristics
##############################################################################
def material_score_white(board: chess.Board) -> float:
    mat = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES_OUR.get(piece.piece_type, 0)
            mat += val if piece.color == chess.WHITE else -val
    return float(mat)

def squares_around(square: chess.Square):
    if square is None:
        return []
    f = chess.square_file(square)
    r = chess.square_rank(square)
    res = []
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            nf = f + df
            nr = r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                res.append(chess.square(nf, nr))
    return res

def king_safety_score_white(board: chess.Board) -> float:
    """
    If squares around White's king are attacked by Black => negative.
    If squares around Black's king are attacked by White => positive.
    """
    score = 0.0
    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)

    if w_king is not None:
        wk_neighbors = squares_around(w_king)
        attacked_w = sum(1 for sq in wk_neighbors if board.is_attacked_by(chess.BLACK, sq))
        score -= PARAMS["KING_SAFETY"] * attacked_w

    if b_king is not None:
        bk_neighbors = squares_around(b_king)
        attacked_b = sum(1 for sq in bk_neighbors if board.is_attacked_by(chess.WHITE, sq))
        score += PARAMS["KING_SAFETY"] * attacked_b

    return score

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

def center_control_score_white(board: chess.Board) -> float:
    wc = 0
    bc = 0
    for sq in CENTER_SQUARES:
        if board.is_attacked_by(chess.WHITE, sq):
            wc += 1
        if board.is_attacked_by(chess.BLACK, sq):
            bc += 1
    return PARAMS["CENTER_CONTROL"] * (wc - bc)

def _possible_moves_for_color(board: chess.Board, color: bool):
    temp = board.copy()
    temp.turn = color
    return list(temp.legal_moves)

def mobility_score_white(board: chess.Board) -> float:
    white_moves = len(_possible_moves_for_color(board, chess.WHITE))
    black_moves = len(_possible_moves_for_color(board, chess.BLACK))
    return PARAMS["MOBILITY"] * (white_moves - black_moves)

def _is_passed_pawn(board: chess.Board, sq: chess.Square, color: bool) -> bool:
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    direction = 1 if color == chess.WHITE else -1
    opp_color = not color
    check_rank = rank + direction
    while 0 <= check_rank < 8:
        for df in [-1, 0, 1]:
            nf = file + df
            if 0 <= nf < 8:
                test_sq = chess.square(nf, check_rank)
                piece = board.piece_at(test_sq)
                if piece and piece.color == opp_color and piece.piece_type == chess.PAWN:
                    return False
        check_rank += direction
    return True

def _count_passed_pawns(board: chess.Board, color: bool) -> int:
    pawns = board.pieces(chess.PAWN, color)
    return sum(1 for sq in pawns if _is_passed_pawn(board, sq, color))

def passed_pawns_score_white(board: chess.Board) -> float:
    w_passed = _count_passed_pawns(board, chess.WHITE)
    b_passed = _count_passed_pawns(board, chess.BLACK)
    return PARAMS["PASSED_PAWNS"] * (w_passed - b_passed)

def _file_has_pawn(board: chess.Board, file_idx: int) -> bool:
    for rank in range(8):
        piece = board.piece_at(chess.square(file_idx, rank))
        if piece and piece.piece_type == chess.PAWN:
            return True
    return False

def rook_open_file_score_white(board: chess.Board) -> float:
    w_rooks = board.pieces(chess.ROOK, chess.WHITE)
    b_rooks = board.pieces(chess.ROOK, chess.BLACK)
    wo = sum(1 for r in w_rooks if not _file_has_pawn(board, chess.square_file(r)))
    bo = sum(1 for r in b_rooks if not _file_has_pawn(board, chess.square_file(r)))
    return PARAMS["ROOK_OPEN_FILE"] * (wo - bo)

def development_score_white(board: chess.Board) -> float:
    w_minors = board.pieces(chess.KNIGHT, chess.WHITE) | board.pieces(chess.BISHOP, chess.WHITE)
    b_minors = board.pieces(chess.KNIGHT, chess.BLACK) | board.pieces(chess.BISHOP, chess.BLACK)
    wd = sum(1 for sq in w_minors if chess.square_rank(sq) != 0)
    bd = sum(1 for sq in b_minors if chess.square_rank(sq) != 7)
    return PARAMS["DEVELOPMENT"] * (wd - bd)

def bishop_pair_score_white(board: chess.Board) -> float:
    white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    white_has_pair = (white_bishops >= 2)
    black_has_pair = (black_bishops >= 2)
    score = 0.0
    if white_has_pair and not black_has_pair:
        score += PARAMS["BISHOP_PAIR"]
    if black_has_pair and not white_has_pair:
        score -= PARAMS["BISHOP_PAIR"]
    return score

from collections import Counter
def _pawn_structure_penalties(board: chess.Board, color: bool) -> int:
    pawns = list(board.pieces(chess.PAWN, color))
    files_occupied = [chess.square_file(p) for p in pawns]
    penalty = 0

    # Count doubled
    counts = Counter(files_occupied)
    for fcount in counts.values():
        if fcount > 1:
            penalty += (fcount - 1)

    # Count isolated
    for p_sq in pawns:
        p_file = chess.square_file(p_sq)
        left_file = p_file - 1
        right_file = p_file + 1
        has_left = any(chess.square_file(s) == left_file for s in pawns) if left_file >= 0 else False
        has_right = any(chess.square_file(s) == right_file for s in pawns) if right_file < 8 else False
        if (not has_left) and (not has_right):
            penalty += 1

    return penalty

def pawn_structure_score_white(board: chess.Board) -> float:
    white_penalties = _pawn_structure_penalties(board, chess.WHITE)
    black_penalties = _pawn_structure_penalties(board, chess.BLACK)
    return PARAMS["PAWN_STRUCTURE"] * (black_penalties - white_penalties)

def check_score_white(board: chess.Board) -> float:
    if board.is_game_over():
        return 0.0
    if board.is_check():
        # if black is in check => bonus for white, if white is in check => negative
        if board.turn == chess.BLACK:
            return PARAMS["CHECK_SCORE"]
        else:
            return -PARAMS["CHECK_SCORE"]
    return 0.0

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

##############################################################################
# 4) Advanced Heuristics
##############################################################################
def knight_outpost_score_white(board: chess.Board) -> float:
    score = 0.0
    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
        r = chess.square_rank(sq)
        if 3 <= r <= 6:
            if not _black_can_pawn_attack_square(board, sq):
                score += PARAMS["KNIGHT_OUTPOST"]
    return score

def _black_can_pawn_attack_square(board: chess.Board, sq: chess.Square) -> bool:
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    check_r = rank + 1
    if not (0 <= check_r < 8):
        return False
    for f in [file - 1, file + 1]:
        if 0 <= f < 8:
            candidate_sq = chess.square(f, check_r)
            piece = board.piece_at(candidate_sq)
            if piece and piece.color == chess.BLACK and piece.piece_type == chess.PAWN:
                return True
    return False

def rook_on_7th_rank_score_white(board: chess.Board) -> float:
    score = 0.0
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        if chess.square_rank(sq) == 6:
            score += PARAMS["ROOK_7TH"]
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        if chess.square_rank(sq) == 1:
            score -= PARAMS["ROOK_7TH"]
    return score

def is_endgame(board: chess.Board) -> bool:
    val = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type != chess.PAWN:
            val += PIECE_VALUES_OUR.get(piece.piece_type, 0)
    return (val <= 13)

def king_endgame_score_white(board: chess.Board) -> float:
    if not is_endgame(board):
        return 0.0
    center_sqs = [chess.D4, chess.E4, chess.D5, chess.E5]
    def best_distance_score(k_sq: chess.Square):
        d_min = min(chess.square_distance(k_sq, c) for c in center_sqs)
        return max(0, 4 - d_min) * PARAMS["KING_ENDGAME"]

    score = 0.0
    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)
    if w_king:
        score += best_distance_score(w_king)
    if b_king:
        score -= best_distance_score(b_king)
    return score

def connected_rooks_score_white(board: chess.Board) -> float:
    w_r = list(board.pieces(chess.ROOK, chess.WHITE))
    b_r = list(board.pieces(chess.ROOK, chess.BLACK))
    w_connected = _count_connected_rook_pairs(board, w_r)
    b_connected = _count_connected_rook_pairs(board, b_r)
    return PARAMS["ROOKS_CONNECTED"] * (w_connected - b_connected)

def _count_connected_rook_pairs(board: chess.Board, rooks_squares):
    from itertools import combinations
    total = 0
    for r1, r2 in combinations(rooks_squares, 2):
        if _are_rooks_connected(board, r1, r2):
            total += 1
    return total

def _are_rooks_connected(board: chess.Board, sq1: chess.Square, sq2: chess.Square) -> bool:
    f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
    f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
    if r1 == r2:
        step = 1 if f2 > f1 else -1
        for f in range(f1 + step, f2, step):
            if board.piece_at(chess.square(f, r1)):
                return False
        return True
    elif f1 == f2:
        step = 1 if r2 > r1 else -1
        for r in range(r1 + step, r2, step):
            if board.piece_at(chess.square(f1, r)):
                return False
        return True
    return False

def space_advantage_score_white(board: chess.Board) -> float:
    white_space = 0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        r = chess.square_rank(sq)
        if r > 3:
            white_space += (r - 3)
    black_space = 0
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        r = chess.square_rank(sq)
        if r < 4:
            black_space += (3 - r)
    return PARAMS["SPACE_ADVANTAGE"] * (white_space - black_space)

##############################################################################
# 5) Super-Advanced Heuristics
##############################################################################
def pawn_promotion_potential_white(board: chess.Board) -> float:
    score = 0.0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        r = chess.square_rank(sq)
        score += PARAMS["PAWN_PROMO"] * (r - 1)
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        r = chess.square_rank(sq)
        if r < 6:
            score -= PARAMS["PAWN_PROMO"] * (6 - r)
    return score

def trade_preference_white(board: chess.Board) -> float:
    mat = material_score_white(board)
    return PARAMS["TRADE_PREF"] * mat

def is_white_king_castled(board: chess.Board) -> bool:
    w_king_sq = board.king(chess.WHITE)
    if w_king_sq == chess.G1:
        if (chess.F1 in board.pieces(chess.ROOK, chess.WHITE)) or (chess.H1 in board.pieces(chess.ROOK, chess.WHITE)):
            return True
    if w_king_sq == chess.C1:
        if (chess.D1 in board.pieces(chess.ROOK, chess.WHITE)) or (chess.A1 in board.pieces(chess.ROOK, chess.WHITE)):
            return True
    return False

def castle_bonus_white(board: chess.Board) -> float:
    if is_white_king_castled(board):
        return PARAMS["CASTLE_BONUS"]
    return 0.0

def king_shield_score_white(board: chess.Board) -> float:
    if is_endgame(board):
        return 0.0
    w_king_sq = board.king(chess.WHITE)
    if w_king_sq is None:
        return 0.0
    kr = chess.square_rank(w_king_sq)
    kf = chess.square_file(w_king_sq)
    if kr == 0:
        shield = 0
        for df in [-1,0,1]:
            nf = kf + df
            if 0 <= nf < 8:
                sq2 = chess.square(nf,1)
                p = board.piece_at(sq2)
                if p and p.color==chess.WHITE and p.piece_type==chess.PAWN:
                    shield += 1
        return PARAMS["KING_SHIELD"] * shield
    return 0.0

def super_advanced_heuristics_white(board: chess.Board) -> float:
    val = 0.0
    val += pawn_promotion_potential_white(board)
    val += trade_preference_white(board)
    val += castle_bonus_white(board)
    val += king_shield_score_white(board)
    return val

##############################################################################
# 6) Combine heuristics
##############################################################################
def base_heuristics_white(board: chess.Board) -> float:
    val = 0.0
    val += material_score_white(board)
    val += king_safety_score_white(board)
    val += center_control_score_white(board)
    val += mobility_score_white(board)
    val += passed_pawns_score_white(board)
    val += rook_open_file_score_white(board)
    val += development_score_white(board)
    val += bishop_pair_score_white(board)
    val += pawn_structure_score_white(board)
    val += check_score_white(board)
    val += 5*checkmate_score_white(board)
    return val

def advanced_heuristics_white(board: chess.Board) -> float:
    val = 0.0
    val += knight_outpost_score_white(board)
    val += rook_on_7th_rank_score_white(board)
    val += king_endgame_score_white(board)
    val += connected_rooks_score_white(board)
    val += space_advantage_score_white(board)
    return val

def final_heuristics_white(board: chess.Board) -> float:
    val = base_heuristics_white(board)
    val += advanced_heuristics_white(board)
    val += super_advanced_heuristics_white(board)
    pst_score = piece_square_table_score_white(board)
    val += draw_avoidance_score_white(board)
    val += PARAMS["PST_FACTOR"] * pst_score
    return val

##############################################################################
# 7) Evaluate from side's perspective
##############################################################################
def evaluate_position_for_side(board: chess.Board, side: bool) -> float:
    fen_key = (board.fen(), side)
    if fen_key in _evaluation_cache:
        return _evaluation_cache[fen_key]

    white_eval = final_heuristics_white(board)
    sign = 1.0 if side == chess.WHITE else -1.0
    final_eval = sign * white_eval

    _evaluation_cache[fen_key] = final_eval
    return final_eval

##############################################################################
# 8) Move-level heuristic factor
##############################################################################
def pinned_move_penalty(board: chess.Board, move: chess.Move, side: bool) -> float:
    fen_before = board.fen()
    key = (fen_before, move.uci(), side)
    if key in _pinned_cache:
        return _pinned_cache[key]

    temp = board.copy()
    temp.turn = side
    if temp.is_pinned(side, move.from_square):
        val = PARAMS["PINNED_PENALTY"]
    else:
        val = 0.0

    _pinned_cache[key] = val
    return val

def compute_heuristic_factor(board: chess.Board, move: chess.Move, side: bool) -> float:
    old_eval = evaluate_position_for_side(board, side)
    pin_pen = pinned_move_penalty(board, move, side)

    temp = board.copy()
    temp.push(move)
    new_eval = evaluate_position_for_side(temp, side)

    delta = (new_eval - old_eval) + pin_pen
    scale = PARAMS["SCALE_FACTOR"]
    clamp_value = 50.0

    raw_exponent = scale * delta
    raw_exponent_clamped = max(-clamp_value, min(clamp_value, raw_exponent))

    factor = float(np.exp(raw_exponent_clamped))
    return factor

##############################################################################
# 9) MCTS wrapper returning (move_list, visits_list)
##############################################################################
def run_mcts_batched_return_stats(
    model,
    root_board: chess.Board,
    n_simulations=128,
    batch_size=32,
    mate_depth=0,
    temperature=1.0,
    add_dirichlet_noise=False,
    dirichlet_alpha=0.03,
    dirichlet_epsilon=0.25,
):
    root_node = MCTSNode(board=root_board.copy(), parent=None, prior=1.0)
    expansions_done = 0
    root_expanded = False

    while expansions_done < n_simulations:
        leaf_nodes = []
        while len(leaf_nodes) < batch_size and expansions_done < n_simulations:
            node = root_node
            while node.children and not node.is_terminal:
                node = node.best_child_for_selection

            if node.is_terminal:
                backprop(node, terminal_value(node.board))
                expansions_done += 1
                continue

            leaf_nodes.append(node)
            expansions_done += 1

        if not leaf_nodes:
            continue

        leaf_enc_list = []
        for ln in leaf_nodes:
            leaf_enc_list.append(encode_single_board(ln.board))
        leaf_enc_array = np.array(leaf_enc_list, dtype=np.float32)

        policy_batch, value_batch = model.predict(leaf_enc_array, verbose=0)

        for i, node in enumerate(leaf_nodes):
            policy_vec = policy_batch[i]
            leaf_value = float(value_batch[i])

            legal_moves = list(node.board.legal_moves)
            if not legal_moves:
                backprop(node, terminal_value(node.board))
                continue

            priors = {}
            total_p = 1e-8
            for mv in legal_moves:
                idx = move_to_index(mv)
                p = policy_vec[idx]
                priors[mv] = p
                total_p += p
            for mv in priors:
                priors[mv] /= total_p

            if add_dirichlet_noise and node is root_node and not root_expanded:
                alpha_vec = [dirichlet_alpha] * len(legal_moves)
                noise = np.random.dirichlet(alpha_vec)
                for idx_mv, mv in enumerate(legal_moves):
                    priors[mv] = (
                        (1 - dirichlet_epsilon)*priors[mv]
                        + dirichlet_epsilon*noise[idx_mv]
                    )
                root_expanded = True

            for mv in legal_moves:
                nb = node.board.copy()
                nb.push(mv)
                node.children[mv] = MCTSNode(board=nb, parent=node, prior=priors[mv])

            backprop(node, leaf_value)

    if not root_node.children:
        return ([], [])
    move_list = []
    visits_list = []
    for mv, child in root_node.children.items():
        move_list.append(mv)
        visits_list.append(child.visit_count)
    return (move_list, visits_list)

##############################################################################
# 10) HeuristicsEngine
##############################################################################
class HeuristicsEngine:
    """
    1) Runs MCTS at the root node.
    2) Converts child visit_count -> MCTS probability.
    3) For each root move, compute factor = exp(SCALE_FACTOR * eval_diff + pinned_penalty).
    4) final_score = MCTS_probability * factor; pick argmax.
    """

    def __init__(self, model, temperature=1.0):
        self.model = model
        self.temperature = temperature

    def get_best_move(
        self,
        board: chess.Board,
        n_simulations=256,
        batch_size=128,
        mate_depth=0,
        add_dirichlet_noise=True,
        dirichlet_alpha=0.03,
        dirichlet_epsilon=0.25,
    ) -> chess.Move:

        move_list, visits_list = run_mcts_batched_return_stats(
            model=self.model,
            root_board=board,
            n_simulations=n_simulations,
            batch_size=batch_size,
            mate_depth=mate_depth,
            temperature=self.temperature,
            add_dirichlet_noise=add_dirichlet_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

        if not move_list:  # No legal moves => checkmate/stalemate
            return None

        total_visits = float(np.sum(visits_list))
        if total_visits <= 1e-8:
            return None

        mcts_probs = [v / total_visits for v in visits_list]
        side = board.turn

        final_scores = []
        for prob, mv in zip(mcts_probs, move_list):
            h_factor = compute_heuristic_factor(board, mv, side)
            final_scores.append(prob * h_factor)

        best_idx = np.argmax(final_scores)
        return move_list[best_idx]