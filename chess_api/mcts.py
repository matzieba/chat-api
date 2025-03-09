import chess
import numpy as np
from chess_engine.engine.train_supervised.parse_pgn_alpha0 import (
    move_to_index,
    encode_single_board
)


class MCTSNode:
    """
    A single node in the MCTS tree for an AlphaZero‐style search.
    """
    __slots__ = [
        'board',      # chess.Board for this node's position
        'parent',     # parent MCTSNode
        'children',   # dict[move -> MCTSNode]
        'prior',      # prior probability (from the NN policy)
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
        c_puct = 2.5  # typical AlphaZero might use 1.5..2.5
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


def terminal_value(board: chess.Board) -> float:
    """
    Returns a value in [-1..+1] if board is terminal, from White's perspective:
      +1 if White has won,
      -1 if Black has won,
       0 if draw (or not terminal).
    """
    if not board.is_game_over():
        return 0.0
    result = board.result()  # e.g. "1-0", "0-1", "1/2-1/2"
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    else:
        return 0.0


def backprop(node: MCTSNode, leaf_value: float):
    """
    Propagate the evaluation up the tree.
    If you wanted side-to-move perspective, you could flip signs; here it is from White's perspective.
    """
    cur = node
    while cur is not None:
        cur.visit_count += 1
        cur.value_sum += leaf_value
        cur = cur.parent


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
    Perform MCTS simulations, optionally applying Dirichlet noise to the root node's prior.
    Afterwards, choose a move from the root according to the temperature and return only the best_move.
    """
    root_node = MCTSNode(board=root_board.copy(), parent=None, prior=1.0)
    expansions_done = 0
    root_expanded = False  # track if we've expanded the root once

    while expansions_done < n_simulations:
        # Collect leaves needing expansion in a mini‐batch
        leaf_nodes = []
        while len(leaf_nodes) < batch_size and expansions_done < n_simulations:
            node = root_node

            # (A) Selection: descend the tree
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

        if not leaf_nodes:
            # Possibly no expansions needed if everything is terminal
            continue

        # (B) Evaluate the leaf nodes in a single batch
        leaf_enc_list = []
        for ln in leaf_nodes:
            # Single‐board encoding
            enc_single = encode_single_board(ln.board)
            leaf_enc_list.append(enc_single)

        leaf_enc_array = np.array(leaf_enc_list, dtype=np.float32)  # shape => (B,64,17)
        policy_batch, value_batch = model.predict(leaf_enc_array, verbose=0)
        # policy_batch.shape => (B, NUM_MOVES=20480)
        # value_batch.shape  => (B, 1)

        # (C) Expansion + Backprop
        for i, node in enumerate(leaf_nodes):
            policy_vec = policy_batch[i]
            leaf_value = float(value_batch[i])

            legal_moves = list(node.board.legal_moves)
            if not legal_moves:
                # No moves => terminal
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

            # Apply Dirichlet noise once if at the root
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

    # If no children at root => no legal moves => return None
    if not root_node.children:
        return None

    # Build a visit-count list so we can pick a final move via temperature
    move_list = []
    visits_list = []
    for mv, child in root_node.children.items():
        move_list.append(mv)
        visits_list.append(child.visit_count)

    sum_visits = float(np.sum(visits_list))
    if sum_visits < 1e-8:
        # No valid expansions or all zero
        return None

    # Temperature sampling
    if temperature < 1e-8:
        # Argmax selection
        best_idx = np.argmax(visits_list)
        best_move = move_list[best_idx]
    else:
        dist = np.array(visits_list, dtype=np.float32) / sum_visits
        dist_pow = dist ** (1.0 / temperature)
        dist_pow_sum = dist_pow.sum()
        if dist_pow_sum < 1e-8:
            # fallback to argmax
            best_idx = np.argmax(visits_list)
        else:
            dist_pow /= dist_pow_sum
            best_idx = np.random.choice(len(move_list), p=dist_pow)
        best_move = move_list[best_idx]

    return best_move