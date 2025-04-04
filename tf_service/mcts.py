
import chess
import chess.syzygy
import numpy as np

from environment import encode_single_board, move_to_index
local_syzygy_dir = "3-4-5"

tb_handle = chess.syzygy.open_tablebase(local_syzygy_dir)

def is_tb_position(board: chess.Board, max_pieces=5) -> bool:

    piece_count = len(board.piece_map())  # or sum(1 for sq in chess.SQUARES if board.piece_at(sq))
    return piece_count <= max_pieces

def tablebase_best_move(board: chess.Board) -> chess.Move:
    try:
        if board.is_game_over():
            return None
        best_move = None
        best_wdl = -99999

        for move in board.legal_moves:
            board.push(move)
            wdl = tb_handle.probe_wdl(board)
            board.pop()
            if wdl > best_wdl:
                best_wdl = wdl
                best_move = move

        return best_move

    except chess.syzygy.MissingTableError:
        return None
    except KeyError:
        return None

def forced_mate_moves_in_1(board: chess.Board, side_to_move_result: str):
    """
    Find all moves that deliver immediate mate (mate in 1).
    side_to_move_result is "1-0" if White is to move, else "0-1".
    """
    mate_moves = []
    for mv in board.legal_moves:
        tmp = board.copy()
        tmp.push(mv)
        if tmp.is_game_over() and tmp.result() == side_to_move_result:
            mate_moves.append(mv)
    return mate_moves


def forced_mate_moves_in_2(board: chess.Board, side_to_move_result: str):
    good_moves = []
    if board.is_game_over():
        return good_moves  # If it's over, no moves

    for mv in board.legal_moves:
        next_pos = board.copy()
        next_pos.push(mv)
        if next_pos.is_game_over():
            # Actually that's mate in 1 already
            if next_pos.result() == side_to_move_result:
                good_moves.append(mv)
            continue

        all_replies_lead_to_mate_in_1 = True
        for opp_move in next_pos.legal_moves:
            next_pos2 = next_pos.copy()
            next_pos2.push(opp_move)
            # Now it's again 'side_to_move' turn
            mate_in_1_moves = forced_mate_moves_in_1(next_pos2, side_to_move_result)
            if not mate_in_1_moves:
                all_replies_lead_to_mate_in_1 = False
                break

        if all_replies_lead_to_mate_in_1:
            good_moves.append(mv)
    return good_moves


def forced_mate_moves_in_3(board: chess.Board, side_to_move_result: str):
    """
    Find moves M such that after M (opponent moves R),
    we still have a forced mate in 2 for the same side.
    """
    good_moves = []
    if board.is_game_over():
        return good_moves

    for mv in board.legal_moves:
        next_pos = board.copy()
        next_pos.push(mv)
        if next_pos.is_game_over():
            # Actually that's mate in 1
            if next_pos.result() == side_to_move_result:
                good_moves.append(mv)
            continue

        all_replies_lead_to_mate_in_2 = True
        for opp_move in next_pos.legal_moves:
            next_pos2 = next_pos.copy()
            next_pos2.push(opp_move)

            mate_in_2_moves = forced_mate_moves_in_2(next_pos2, side_to_move_result)
            if not mate_in_2_moves:
                all_replies_lead_to_mate_in_2 = False
                break

        if all_replies_lead_to_mate_in_2:
            good_moves.append(mv)

    return good_moves

def forced_mate_moves_in_4(board: chess.Board, side_to_move_result: str):
    """
    Find moves M such that, after M (opponent moves R),
    we still have a forced mate in 3 for the same side.
    """
    good_moves = []
    if board.is_game_over():
        return good_moves

    for mv in board.legal_moves:
        next_pos = board.copy()
        next_pos.push(mv)
        if next_pos.is_game_over():
            # Already mate in 1
            if next_pos.result() == side_to_move_result:
                good_moves.append(mv)
            continue

        all_replies_lead_to_mate_in_3 = True
        for opp_move in next_pos.legal_moves:
            next_pos2 = next_pos.copy()
            next_pos2.push(opp_move)
            mate_in_3_moves = forced_mate_moves_in_3(next_pos2, side_to_move_result)
            if not mate_in_3_moves:
                all_replies_lead_to_mate_in_3 = False
                break

        if all_replies_lead_to_mate_in_3:
            good_moves.append(mv)

    return good_moves
def forced_mate_moves_in_5(board: chess.Board, side_to_move_result: str):
    """
    Find moves M such that, after M (opponent moves R),
    we still have a forced mate in 4 for the same side.
    """
    good_moves = []
    if board.is_game_over():
        return good_moves

    for mv in board.legal_moves:
        next_pos = board.copy()
        next_pos.push(mv)
        if next_pos.is_game_over():
            # Already mate in 1
            if next_pos.result() == side_to_move_result:
                good_moves.append(mv)
            continue

        all_replies_lead_to_mate_in_4 = True
        for opp_move in next_pos.legal_moves:
            next_pos2 = next_pos.copy()
            next_pos2.push(opp_move)
            mate_in_4_moves = forced_mate_moves_in_4(next_pos2, side_to_move_result)
            if not mate_in_4_moves:
                all_replies_lead_to_mate_in_4 = False
                break

        if all_replies_lead_to_mate_in_4:
            good_moves.append(mv)

    return good_moves
def forced_mate_moves_in_6(board: chess.Board, side_to_move_result: str):
    """
    Find moves M such that, after M (opponent moves R),
    we still have a forced mate in 5 for the same side.
    """
    good_moves = []
    if board.is_game_over():
        return good_moves

    for mv in board.legal_moves:
        next_pos = board.copy()
        next_pos.push(mv)
        if next_pos.is_game_over():
            # Already mate in 1
            if next_pos.result() == side_to_move_result:
                good_moves.append(mv)
            continue

        all_replies_lead_to_mate_in_5 = True
        for opp_move in next_pos.legal_moves:
            next_pos2 = next_pos.copy()
            next_pos2.push(opp_move)
            mate_in_5_moves = forced_mate_moves_in_5(next_pos2, side_to_move_result)
            if not mate_in_5_moves:
                all_replies_lead_to_mate_in_5 = False
                break

        if all_replies_lead_to_mate_in_5:
            good_moves.append(mv)

    return good_moves

def find_forced_mate_moves_up_to_n(board: chess.Board, max_n=3):
    """
    Returns (distance, moves_list), where 'distance' is 1..max_n if found,
    or None if no forced mate up to that depth. moves_list are the moves
    that achieve that forced mate.
    """
    side_to_move_result = "1-0" if board.turn == chess.WHITE else "0-1"

    # Mate in 1
    if max_n >= 1:
        m1 = forced_mate_moves_in_1(board, side_to_move_result)
        if m1:
            return (1, m1)

    # Mate in 2
    if max_n >= 2:
        m2 = forced_mate_moves_in_2(board, side_to_move_result)
        if m2:
            return (2, m2)

    # Mate in 3
    if max_n >= 3:
        m3 = forced_mate_moves_in_3(board, side_to_move_result)
        if m3:
            return (3, m3)

    # Mate in 4
    if max_n >= 4:
        m4 = forced_mate_moves_in_4(board, side_to_move_result)
        if m4:
            return (4, m4)

    # Mate in 5
    if max_n >= 5:
        m5 = forced_mate_moves_in_5(board, side_to_move_result)
        if m5:
            return (5, m5)

    # Mate in 6
    if max_n >= 6:
        m6 = forced_mate_moves_in_6(board, side_to_move_result)
        if m6:
            return (6, m6)

    return (None, [])
########################
# MCTS Node
###############################################################################
class MCTSNode:
    __slots__ = [
        'board',      # chess.Board
        'parent',
        'children',   # dict: move -> MCTSNode
        'prior',      # float
        'visit_count',
        'value_sum',
        'is_terminal',
    ]

    def __init__(self, board: chess.Board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_terminal = board.is_game_over()

    @property
    def q_value(self):
        # Mean value
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def u_value(self):
        # Upper Confidence Bound
        c_puct = 2.5
        if self.parent is None:
            return 0.0
        return c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)

    @property
    def best_child_for_selection(self):
        # Choose child maximizing (Q + U)
        best_score = float("-inf")
        best_child = None
        for child in self.children.values():
            score = child.q_value + child.u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


###############################################################################
# Terminal value from White's perspective
###############################################################################
def terminal_value(board: chess.Board) -> float:
    """+1 if White won, -1 if Black won, else 0."""
    if board.is_game_over():
        res = board.result()  # "1-0", "0-1", "1/2-1/2", or "*"
        if res == "1-0":
            return 1.0
        elif res == "0-1":
            return -1.0
        else:
            return 0.0
    return 0.0


###############################################################################
# Backprop
###############################################################################
def backprop(node: MCTSNode, leaf_value: float):
    """
    Propagate a value up the tree. The value is from White's perspective:
    +1 = White is winning, -1 = Black is winning.
    """
    cur = node
    while cur is not None:
        cur.visit_count += 1
        cur.value_sum += leaf_value
        cur = cur.parent


###############################################################################
# Main MCTS function
###############################################################################
def run_mcts_batched(
    model,
    root_board: chess.Board,
    n_simulations=100,
    batch_size=32,
    mate_depth=0,
    temperature=0.2,
    add_dirichlet_noise=False,
    dirichlet_alpha=0.03,
    dirichlet_epsilon=0.25
):
    """
    Runs MCTS for 'n_simulations' playouts in mini-batches of size 'batch_size'.
    Optionally checks forced mate up to 'mate_depth' half-moves.

    After expansions, pick a move from the root by sampling visits^(1/temperature).
      - If temperature=0, it picks the child with the most visits (argmax).

    Example usage:
      best_move = run_mcts_batched(model, board,
                                   n_simulations=24,
                                   batch_size=12,
                                   mate_depth=3)
    """
    root_node = MCTSNode(board=root_board.copy(), parent=None, prior=1.0)
    expansions_done = 0
    root_expanded = False

    if is_tb_position(root_board, max_pieces=5):
        tb_move = tablebase_best_move(root_board)
        if tb_move is not None:
            return tb_move

    while expansions_done < n_simulations:
        # Collect up to 'batch_size' leaves
        leaf_nodes = []
        while len(leaf_nodes) < batch_size and expansions_done < n_simulations:
            node = root_node

            # A) Selection
            while node.children and not node.is_terminal:
                node = node.best_child_for_selection

            # If terminal => backprop
            if node.is_terminal:
                backprop(node, terminal_value(node.board))
                expansions_done += 1
                continue

            # Leaf found
            leaf_nodes.append(node)
            expansions_done += 1

        if not leaf_nodes:
            # Possibly all terminal
            continue

        # B) Evaluate leaf nodes in a single batch:
        leaf_enc_list = []
        for ln in leaf_nodes:
            leaf_enc_list.append(encode_single_board(ln.board))
        leaf_enc_array = np.array(leaf_enc_list, dtype=np.float32)  # (B, 64, 17)

        policy_batch, value_batch = model.predict(leaf_enc_array, verbose=0)
        # policy_batch => (B, 20480)
        # value_batch  => (B, 1)

        # C) Expansion + forced-mate check + backprop
        for i, node in enumerate(leaf_nodes):
            policy_vec = policy_batch[i]
            leaf_value = float(value_batch[i])

            legal_moves = list(node.board.legal_moves)
            if not legal_moves:
                backprop(node, terminal_value(node.board))
                continue

            # Build priors
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
                for j, mv in enumerate(legal_moves):
                    priors[mv] = (
                        (1 - dirichlet_epsilon) * priors[mv]
                        + dirichlet_epsilon * noise[j]
                    )
                root_expanded = True

            # Create children
            for mv in legal_moves:
                next_board = node.board.copy()
                next_board.push(mv)
                node.children[mv] = MCTSNode(
                    board=next_board,
                    parent=node,
                    prior=priors[mv]
                )

            # Backprop final value
            backprop(node, leaf_value)

    # Root selection
    if not root_node.children:
        return None  # No legal move

    # temperature-based pick
    move_list = []
    visits_list = []
    for mv, child in root_node.children.items():
        move_list.append(mv)
        visits_list.append(child.visit_count)

    sum_visits = float(np.sum(visits_list))
    if sum_visits < 1e-8:
        return None

    if temperature < 1e-8:
        # Argmax
        best_idx = np.argmax(visits_list)
    else:
        dist = np.array(visits_list, dtype=np.float32) / sum_visits
        dist_pow = dist ** (1.0 / temperature)
        dist_pow_sum = dist_pow.sum()
        if dist_pow_sum < 1e-8:
            best_idx = np.argmax(visits_list)
        else:
            dist_pow /= dist_pow_sum
            best_idx = np.random.choice(len(move_list), p=dist_pow)

    return move_list[best_idx]

