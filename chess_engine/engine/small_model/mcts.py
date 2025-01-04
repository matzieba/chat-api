import chess
import numpy as np

from chess_engine.engine.small_model.environment import encode_board, build_move_mask, move_to_index, NUM_MOVES


class MCTSNode:
    def __init__(self, board: chess.Board):
        self.board = board.copy()
        self.children = {}  # move -> MCTSNode
        self.N = 0          # visit count
        self.W = 0.0        # total value
        self.Q = 0.0        # mean value = W/N
        self.P = 0.0        # prior from neural net

def run_mcts(root: MCTSNode, model, num_simulations, c_puct=1.0):
    for _ in range(num_simulations):
        node, path = select_leaf(root, c_puct)
        if not node.board.is_game_over():
            expand_node(node, model)
            value = simulate_playout(node, model)
        else:
            # If the game is over at this node, the value is straightforward:
            # 1.0 if white wins, -1.0 if black wins, 0 if draw
            result = node.board.result()
            if result == '1-0':
                value = 1.0
            elif result == '0-1':
                value = -1.0
            else:
                value = 0.0
        backpropagate(path, value)
    return build_policy_vector(root)

def select_leaf(node: MCTSNode, c_puct=1.0):
    """
    Traverse the tree from the root, picking children with max UCB = Q + c_puct * P * sqrt(N_parent)/(1+N_child).
    Returns (leaf_node, path) where path is the route taken for backprop.
    """
    path = []
    current = node
    while current.children:
        best_ucb = -999999.0
        best_move = None
        best_child = None
        for move, child in current.children.items():
            ucb = child.Q + c_puct * child.P * np.sqrt(current.N + 1e-8) / (1 + child.N)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
                best_move = move
        path.append((current, best_move))
        current = best_child
    return current, path

def expand_node(node: MCTSNode, model):
    """Run the policy/value network to populate node.children with prior probabilities."""
    encoded_board = encode_board(node.board)[np.newaxis, ...]   # (1, 8, 8, 14)
    mask = build_move_mask(node.board)[np.newaxis, ...]         # (1, NUM_MOVES)
    policy, value = model.predict([encoded_board, mask], verbose=0)
    policy = policy[0]  # shape (NUM_MOVES,)
    # clamp or ensure no negative probabilities:
    policy = np.clip(policy, 1e-7, 1.0)
    value = value[0][0] # scalar in [-1,1]

    legal_moves = list(node.board.legal_moves)
    for move in legal_moves:
        idx = move_to_index(move)
        child_board = node.board.copy()
        child_board.push(move)
        child_node = MCTSNode(child_board)
        child_node.P = policy[idx]  # prior from net
        node.children[move] = child_node

    node.N = 0
    node.W = 0.0
    node.Q = 0.0

def simulate_playout(node: MCTSNode, model):
    """
    We do a single forward pass to get the value estimate for this node's board.
    """
    encoded_board = encode_board(node.board)[np.newaxis, ...]
    mask = build_move_mask(node.board)[np.newaxis, ...]
    _, value = model.predict([encoded_board, mask], verbose=0)
    return value[0][0]

def backpropagate(path, leaf_value):
    """
    Update the nodes along the path with the evaluation (leaf_value).
    Each step, we flip the sign of leaf_value to account for the opponent's perspective.
    """
    for parent, _move in reversed(path):
        parent.N += 1
        parent.W += leaf_value
        parent.Q = parent.W / parent.N

        # Flip the perspective for the next parent up the tree
        leaf_value = -leaf_value


def build_policy_vector(root: MCTSNode) -> np.ndarray:
    """
    Build a (NUM_MOVES,) array from the children visit counts of 'root.'
    If the total visit count is zero (or very small), fallback to a uniform distribution
    among the children.
    """
    policy_vec = np.zeros(NUM_MOVES, dtype=np.float32)
    total_visits = sum(child.N for child in root.children.values())

    # If no children or total visits are ~0, fallback to uniform or just return zeros if no moves
    if total_visits < 1e-9:
        if len(root.children) == 0:
            # No children => terminal node => return zeros
            return policy_vec
        # Otherwise, use a uniform distribution among the children
        uniform_prob = 1.0 / len(root.children)
        for mv in root.children.keys():
            idx = move_to_index(mv)
            policy_vec[idx] = uniform_prob
        return policy_vec

    # Normal case: distribute according to visit counts
    for mv, child in root.children.items():
        idx = move_to_index(mv)
        policy_vec[idx] = child.N / total_visits

    # Numerical safety: ensure it sums to ~1.0
    sum_prob = policy_vec.sum()
    if sum_prob < 1e-9:
        # If it still doesn't sum to anything, fallback to uniform
        if len(root.children) == 0:
            return policy_vec
        uniform_prob = 1.0 / len(root.children)
        for mv in root.children.keys():
            idx = move_to_index(mv)
            policy_vec[idx] = uniform_prob
    else:
        policy_vec /= sum_prob

    return policy_vec
