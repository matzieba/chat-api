import numpy as np
import chess

from chess_api.heuristics import evaluate_with_heuristics
from chess_engine.engine.small_model.transformer.environment import encode_board, build_move_mask, move_to_index


class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.prior = prior   # P(s,a) from policy
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def u_value(self, c_puct=1.0):
        return c_puct * self.prior * (
            (self.parent.visit_count**0.5) / (1 + self.visit_count)
        )

def mcts_search(root, model, simulations=100, c_puct=1.0):
    for _ in range(simulations):
        node = root

        # 1. Selection: navigate down the tree
        while node.children:
            node = select_child(node, c_puct)

        # 2. Expansion: expand the leaf
        if not node.board.is_game_over():
            policy, value = evaluate_position(node.board, model)
            expand_node(node, policy)

        else:
            # If game is over, evaluate final outcome
            if node.board.is_checkmate():
                # If the side to move is in checkmate, it’s losing for that side
                value = -1.0
            else:
                # Stalemate or draw
                value = 0.0

        # 3. Backpropagation
        backpropagate(node, value)

    # After simulations, pick the move with highest visit_count
    best_child = max(root.children.values(), key=lambda c: c.visit_count)
    return best_child

def select_child(node, c_puct):
    """Select the child whose (Q + U) is maximum."""
    best_score = -float('inf')
    best_child = None
    for move, child in node.children.items():
        score = child.q_value + child.u_value(c_puct)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child

def evaluate_position(board, model):
    """
    Return (policy, value) from your neural net model for the given board,
    then combine with heuristics.
    """
    # 1) Encode board
    state = encode_board(board)
    mask = build_move_mask(board)

    # 2) NN prediction
    state_input = np.expand_dims(state, axis=0)
    mask_input = np.expand_dims(mask, axis=0)

    policy_logits, nn_value = model.predict(
        {"input_board": state_input, "mask_input": mask_input},
        verbose=0
    )
    policy_logits = policy_logits[0]  # shape: (NUM_MOVES,)
    nn_value = float(nn_value[0])     # shape: ()

    # 3) Build a dictionary {move: prior} only for legal moves
    legal_moves = list(board.legal_moves)
    move_policy = {}
    for move in legal_moves:
        try:
            action_index = move_to_index(move)
            move_policy[move] = policy_logits[action_index]
        except (ValueError, IndexError):
            move_policy[move] = 0.0

    # Normalize policy to get probabilities
    total_policy = sum(move_policy[m] for m in move_policy)
    if total_policy > 0:
        for m in move_policy:
            move_policy[m] /= total_policy
    else:
        # fallback -> uniform
        uniform_prob = 1.0 / max(1, len(legal_moves))
        for m in move_policy:
            move_policy[m] = uniform_prob

    # 4) Combine the network's value with heuristic
    #    You may choose different blending factors or clamping.
    heuristic_value = evaluate_with_heuristics(board)

    # Optionally clamp the heuristic to the [-1,+1] range if it can get large:
    # or you can map it in some other way
    h_clamped = max(-1.0, min(1.0, heuristic_value))

    # Weighted sum:
    alpha = 0.7 # 70% neural net, 30% heuristics
    combined_value = alpha * nn_value + (1 - alpha) * h_clamped

    return move_policy, combined_value

def expand_node(node, move_policy):
    """
    Create a child node for each legal move, setting the prior from the policy distribution.
    """
    for move, prior in move_policy.items():
        board_copy = node.board.copy()
        board_copy.push(move)
        child_node = MCTSNode(board=board_copy, parent=node, prior=prior)
        node.children[move] = child_node

def backpropagate(node, value):
    current = node
    # “value” is from the perspective of the side to move on 'node'
    while current is not None:
        current.visit_count += 1
        current.value_sum += value
        # Flip perspective (assuming the model always outputs from side-to-move’s perspective)
        value = -value
        current = current.parent


def build_policy_vector(root_node):
    """
    Builds a policy vector of shape (NUM_MOVES,) from the MCTS root node
    by normalizing each child's visit count.

    :param root_node: The MCTSNode representing the root of the current search tree.
    :return: A NumPy array of shape (NUM_MOVES,) representing the policy distribution
             (probability of each move) based on visit counts.
    """
    import numpy as np
    from chess_engine.engine.small_model.transformer.environment import move_to_index, NUM_MOVES

    # Sum the visit counts over all children
    total_visits = sum(child.visit_count for child in root_node.children.values())

    # Initialize a zero policy vector
    policy_vector = np.zeros(NUM_MOVES, dtype=np.float32)

    # If there are no children (terminal position), return the zero vector
    if total_visits == 0:
        return policy_vector

    # Set policy_vector[move_index] based on normalized visit counts
    for move, child in root_node.children.items():
        move_idx = move_to_index(move)
        policy_vector[move_idx] = child.visit_count / total_visits

    return policy_vector