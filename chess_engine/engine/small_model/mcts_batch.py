import chess
import numpy as np

# Adjust these imports as needed to match your file structure.
# They should point to your existing environment utilities (encode_board, build_move_mask, move_to_index, etc.).
from chess_engine.engine.small_model.environment import (
    encode_board,
    build_move_mask,
    move_to_index,
    NUM_MOVES
)


###############################################################################
# MCTS Node Definition
###############################################################################
class MCTSNode:
    def __init__(self, board: chess.Board):
        self.board = board.copy()
        self.children = {}  # move -> MCTSNode
        self.N = 0  # visit count
        self.W = 0.0  # total value
        self.Q = 0.0  # mean value = W/N
        self.P = 0.0  # prior from neural net
        self.eval_value = None  # store the NN value (for batched approach)


###############################################################################
# Final Game Result from White's Perspective
###############################################################################
def final_result(board: chess.Board) -> float:
    """
    Returns +1 for White win, -1 for Black win, and 0 for draw, based on board.result().
    """
    result_str = board.result()
    if result_str == "1-0":
        return 1.0
    elif result_str == "0-1":
        return -1.0
    else:
        return 0.0


###############################################################################
# Select Leaf Using UCB
###############################################################################
def select_leaf(node: MCTSNode, c_puct=1.0):
    """
    Traverse the tree from `node`, picking children with max UCB (Q + c_puct * P * sqrt(N_parent)/(1+N_child)).
    Returns (leaf_node, path) for backprop.
    """
    path = []
    current = node
    while current.children:
        best_ucb = -999999.0
        best_move = None
        best_child = None
        for move, child in current.children.items():
            ucb = child.Q + c_puct * child.P * np.sqrt(current.N + 1e-8) / (1.0 + child.N)
            if ucb > best_ucb:
                best_ucb = ucb
                best_move = move
                best_child = child
        path.append((current, best_move))
        current = best_child
    return current, path


###############################################################################
# Backpropagate the Value
###############################################################################
def backpropagate(path, leaf_value):
    """
    Update the nodes along the path with the evaluation (leaf_value).
    If you want fully-correct perspective switching, you can alternate sign at each step:
       leaf_value = -leaf_value
    as you go up the tree. But for simplicity, we skip it here.
    """
    for parent, _move in reversed(path):
        parent.N += 1
        parent.W += leaf_value
        parent.Q = parent.W / parent.N
        # (Optional advanced approach: leaf_value = -leaf_value)


###############################################################################
# Expand Node Using Precomputed Policy
###############################################################################
def expand_node_batch(node: MCTSNode, policy_vect: np.ndarray):
    """
    Populate `node.children` with prior probabilities from a precomputed policy vector.
    We do not do another forward pass here—this is for batch inference usage.
    """
    legal_moves = list(node.board.legal_moves)
    node.children = {}
    for move in legal_moves:
        idx = move_to_index(move)
        child = MCTSNode(node.board)
        child.board.push(move)
        child.P = policy_vect[idx]  # prior from network
        node.children[move] = child

    node.N = 0
    node.W = 0.0
    node.Q = 0.0


###############################################################################
# Build a Policy Vector from Root Visits (Same as Before)
###############################################################################
def build_policy_vector(root: MCTSNode) -> np.ndarray:
    """
    Build a (NUM_MOVES,) array from the children visit counts of `root`.
    If total visits < 1e-9, fallback to uniform among children.
    """
    policy_vec = np.zeros(NUM_MOVES, dtype=np.float32)
    total_visits = sum(child.N for child in root.children.values())
    if total_visits < 1e-9:
        # fallback to uniform among children if any
        if len(root.children) == 0:
            return policy_vec
        uniform_prob = 1.0 / len(root.children)
        for mv in root.children.keys():
            idx = move_to_index(mv)
            policy_vec[idx] = uniform_prob
        return policy_vec
    for mv, child in root.children.items():
        idx = move_to_index(mv)
        policy_vec[idx] = child.N / total_visits
    # Normalize
    sum_prob = policy_vec.sum()
    if sum_prob > 1e-9:
        policy_vec /= sum_prob
    return policy_vec


###############################################################################
# Run MCTS in Batches
###############################################################################
def run_mcts_batch(root: MCTSNode, model, num_simulations: int, batch_size=8, c_puct=1.0):
    """
    Performs `num_simulations` MCTS traversals starting from `root`, but in batches.

    1. Collect up to `batch_size` leaves by calling select_leaf multiple times
       without updating the tree in between.
    2. Split leaves into terminal vs. non-terminal.
    3. For non-terminal leaves, do a single model.predict() across the entire batch.
       - Expand each leaf using the predicted policy.
       - Store the predicted value for backprop.
    4. Backpropagate each leaf result.

    Returns a policy vector for the root (e.g., can be used for move selection).
    """
    i = 0
    while i < num_simulations:
        # 1) Collect a batch of leaf nodes
        leaves = []
        paths = []
        for _ in range(batch_size):
            if i >= num_simulations:
                break
            leaf, path = select_leaf(root, c_puct)
            leaves.append(leaf)
            paths.append(path)
            i += 1

        # 2) Separate terminal vs. non-terminal
        terminal_leaves = []
        nonterminal_leaves = []
        for lf in leaves:
            if lf.board.is_game_over():
                terminal_leaves.append(lf)
            else:
                nonterminal_leaves.append(lf)

        # 3) If we have non-terminal leaves, do a single batch inference
        if nonterminal_leaves:
            boards_batch = []
            masks_batch = []
            for lf in nonterminal_leaves:
                boards_batch.append(encode_board(lf.board))
                masks_batch.append(build_move_mask(lf.board))
            boards_batch = np.array(boards_batch, dtype=np.float32)
            masks_batch = np.array(masks_batch, dtype=np.float32)

            policy_preds, value_preds = model.predict([boards_batch, masks_batch],
                                                      batch_size=len(nonterminal_leaves),
                                                      verbose=0)

            # Expand each leaf with the predicted policy
            # Store the value for later backprop
            for lf, pol, val in zip(nonterminal_leaves, policy_preds, value_preds):
                expand_node_batch(lf, pol)
                lf.eval_value = val[0]  # value_preds is shape (batch, 1)

        # 4) Backprop each leaf
        for (lf, path) in zip(leaves, paths):
            if lf in nonterminal_leaves:
                leaf_value = lf.eval_value  # from the batched NN inference
            else:
                # Terminal leaf => compute final result
                leaf_value = final_result(lf.board)
            backpropagate(path, leaf_value)

    # Optionally build a policy vector from the root (for move selection)
    return build_policy_vector(root)