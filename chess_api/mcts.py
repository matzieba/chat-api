import chess
import numpy as np
from chess_engine.engine.train_supervised.parse_pgn_alpha0 import move_to_index, encode_single_board


class MCTSNode:
    """
    A single node in the MCTS tree for AlphaZero‐style search.
    """
    __slots__ = [
        'board',  # chess.Board for this node's position
        'parent',  # parent MCTSNode
        'children',  # dict[move -> MCTSNode]
        'prior',  # prior probability (from the NN policy)
        'visit_count',
        'value_sum',
        'is_terminal'
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
        # Mean value so far
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def u_value(self):
        # Upper Confidence Bound
        c_puct = 2.5  # typical in AlphaZero is 1.5
        if self.parent is None:
            return 0  # root has no parent
        return c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)

    @property
    def best_child_for_selection(self):
        # Pick the child that maximizes Q + U
        best_score = -1e9
        best_child = None
        for child in self.children.values():
            score = child.q_value + child.u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


def encode_node_4frames(node, max_frames=4):
    """
    Climbs up the node's ancestors to gather up to 'max_frames' boards,
    from oldest to newest. Each is encoded to shape (64,17), then concatenated
    into (64,17*max_frames) = (64,68). Zero-pad if fewer than max_frames.
    """
    boards = []
    cur = node
    while cur is not None and len(boards) < max_frames:
        boards.append(cur.board)
        cur = cur.parent
    boards.reverse()  # oldest -> newest

    frames = []
    missing = max_frames - len(boards)
    for _ in range(missing):
        frames.append(np.zeros((64, 17), dtype=np.float32))
    for b in boards:
        frames.append(encode_single_board(b))
    return np.concatenate(frames, axis=1)  # shape (64, 17*max_frames)


###############################################################################
# Terminal value: for a game_over position
###############################################################################
def terminal_value(board: chess.Board) -> float:
    """
    Returns a value in [-1..+1] if board is terminal.
    +1 if White has won, -1 if Black has won, 0 for a draw.
    This is from White's perspective.
    """
    if not board.is_game_over():
        return 0.0  # not terminal
    result = board.result()  # e.g. "1-0", "0-1", "1/2-1/2"
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    else:
        return 0.0


###############################################################################
# Backprop
###############################################################################
def backprop(node: MCTSNode, leaf_value: float):
    """
    Propagate the evaluation up the tree. If you want side-to-move perspective,
    you can flip sign each step (commented out below). Example here is from
    White's perspective only.
    """
    cur = node
    # sign = 1.0  # if flipping, you'd do sign flips each step
    while cur is not None:
        cur.visit_count += 1
        cur.value_sum += leaf_value  # * sign
        # sign = -sign
        cur = cur.parent


###############################################################################
# Batched MCTS with Dirichlet noise & Temperature
###############################################################################
def run_mcts_batched(
        model,
        root_board: chess.Board,
        n_simulations=100,
        batch_size=32,
        temperature=1.0,
        add_dirichlet_noise=False,
        dirichlet_alpha=0.03,
        dirichlet_epsilon=0.25
):
    """
    Perform MCTS simulations, optionally applying Dirichlet noise to the
    root node's prior, then pick a move from the root according to the
    temperature. Returns (chosen_move, (policy distribution over 20480 moves)).
    """
    root_node = MCTSNode(board=root_board.copy(), parent=None, prior=1.0)
    expansions_done = 0
    root_expanded = False  # Track if we've expanded the root once

    while expansions_done < n_simulations:
        leaf_nodes = []
        # Collect up to 'batch_size' leaves needing expansion
        while len(leaf_nodes) < batch_size and expansions_done < n_simulations:
            node = root_node

            # (A) Selection: descend tree
            while node.children and not node.is_terminal:
                node = node.best_child_for_selection

            # If terminal => backprop and skip
            if node.is_terminal:
                final_value = terminal_value(node.board)
                backprop(node, final_value)
                expansions_done += 1
                continue

            # We have a leaf to expand
            leaf_nodes.append(node)
            expansions_done += 1

        if len(leaf_nodes) == 0:
            # no expansions needed (maybe all terminal)
            continue

        # (B) Evaluate in a single batch
        leaf_enc_list = []
        for ln in leaf_nodes:
            enc_4f = encode_node_4frames(ln, max_frames=4)
            leaf_enc_list.append(enc_4f)

        leaf_enc_array = np.array(leaf_enc_list, dtype=np.float32)  # (B,64,68)
        policy_batch, value_batch = model.predict(leaf_enc_array, verbose=0)
        # policy_batch.shape => (B, 20480)
        # value_batch.shape  => (B, 1)

        # (C) Expansion + Backprop
        for i, node in enumerate(leaf_nodes):
            policy_vec = policy_batch[i]
            leaf_value = float(value_batch[i])

            legal_moves = list(node.board.legal_moves)
            if not legal_moves:
                # no moves => terminal
                tv = terminal_value(node.board)
                backprop(node, tv)
                continue

            # Convert raw policy to priors for the legal moves
            priors = {}
            total_p = 1e-8
            for mv in legal_moves:
                idx = move_to_index(mv)
                p = policy_vec[idx]
                priors[mv] = p
                total_p += p

            # Normalize
            for mv in priors:
                priors[mv] /= total_p

            # (Optional) Apply Dirichlet noise if this is the first time we expand the root
            if add_dirichlet_noise and node is root_node and not root_expanded:
                alpha_vec = [dirichlet_alpha] * len(legal_moves)
                noise = np.random.dirichlet(alpha_vec)
                for j, mv in enumerate(legal_moves):
                    blended = (1 - dirichlet_epsilon) * priors[mv] + dirichlet_epsilon * noise[j]
                    priors[mv] = blended
                root_expanded = True

            # Create child nodes
            for mv in legal_moves:
                next_board = node.board.copy()
                next_board.push(mv)
                node.children[mv] = MCTSNode(
                    board=next_board,
                    parent=node,
                    prior=priors[mv]
                )

            # Backprop the leaf value
            backprop(node, leaf_value)

    # If no children at root => no moves
    if not root_node.children:
        return None, None

    # Build final distribution from visit counts
    distribution = np.zeros(20480, dtype=np.float32)
    move_visit_counts = {}
    for mv, child in root_node.children.items():
        idx = move_to_index(mv)
        distribution[idx] = child.visit_count
        move_visit_counts[mv] = child.visit_count

    sum_visits = np.sum(distribution)
    if sum_visits > 1e-8:
        distribution /= sum_visits

    # (D) Use temperature to pick the final move from the root
    if temperature < 1e-8:
        # ~Argmax selection
        best_move = max(move_visit_counts, key=move_visit_counts.get)
    else:
        # Sample from distribution^(1/temperature)
        dist_pow = distribution ** (1.0 / temperature)
        dist_pow_sum = np.sum(dist_pow)
        if dist_pow_sum < 1e-8:
            # fallback to argmax
            best_move = max(move_visit_counts, key=move_visit_counts.get)
        else:
            dist_pow /= dist_pow_sum
            # Sample an index
            move_idx = np.random.choice(len(dist_pow), p=dist_pow)
            # Map index back to the corresponding move
            # We'll invert the dictionary or just do a second pass
            # But more straightforward to loop again:
            best_move = None
            for mv, child in root_node.children.items():
                if move_to_index(mv) == move_idx:
                    best_move = mv
                    break

    return best_move, distribution