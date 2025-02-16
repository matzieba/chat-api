import math
import numpy as np
import chess
from collections import deque

# Adjust if needed
from chess_engine.engine.small_model.transformer.environment import (
    encode_board, move_to_index
)

NUM_MOVES = 8192  # Typically 4096 for 64 squares x 64 squares


class MCTSNode:
    """
    A single node in the MCTS tree.
      - board: current chess.Board
      - parent: parent node
      - prior: policy probability for this node
      - children: dict of {move: MCTSNode}
      - visit_count: how many times this node has been visited
      - value_sum: sum of backpropagated values
    """
    def __init__(self, board, parent=None, prior=1.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def Q(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    @property
    def P(self):
        return self.prior

    def expand(self, policy_logits):
        """
        Create children for all legal moves, with prior = policy_logits[move_index].
        Skips promotions if that matches your training approach.
        """
        if self.board.is_game_over():
            return

        for move in self.board.legal_moves:
            idx = move_to_index(move)
            if not (0 <= idx < NUM_MOVES):
                continue

            prob = policy_logits[idx]
            next_board = self.board.copy()
            next_board.push(move)
            self.children[move] = MCTSNode(board=next_board, parent=self, prior=prob)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None


def select_child(node, c_puct):
    """
    Select child with maximum UCB = Q + c_puct * P * sqrt(node.visit_count)/(1+child.visit_count).
    """
    best_score = -float('inf')
    best_move, best_child = None, None

    for move, child in node.children.items():
        ucb = (child.Q +
               c_puct * child.P * math.sqrt(node.visit_count + 1) / (child.visit_count + 1))
        if ucb > best_score:
            best_score = ucb
            best_move = move
            best_child = child
    return best_move, best_child


def backpropagate(node, value):
    """
    Traverse back up the tree, flipping 'value' each step, because each parent
    is the opposing side's perspective.
    """
    current = node
    current_value = value
    while current is not None:
        current.visit_count += 1
        current.value_sum += current_value
        current_value = -current_value
        current = current.parent


def simulate_one(node, pending_expansions, c_puct):
    """
    Perform one simulation step:
      1) Selection: descend tree until leaf or terminal node
      2) If terminal, backprop result
      3) Else add leaf to 'pending_expansions'
    Returns: the leaf node (or None if terminal).
    """
    current = node
    while not current.is_leaf() and not current.board.is_game_over():
        _, current = select_child(current, c_puct)

    if current.board.is_game_over():
        # Terminal position => assign value
        if current.board.is_checkmate():
            # The side to move was just mated => from their perspective, value = -1
            value = -1.0
        else:
            # Stalemate or draw => value = 0
            value = 0.0
        backpropagate(current, value)
        return None
    else:
        pending_expansions.append(current)
        return current


def expand_and_backprop(batch_nodes, policy_logits_list, values_list):
    """
    For each leaf node in 'batch_nodes', expand its children with policy_logits
    and backprop the returned 'value'.
    """
    for node, policy_logits, value in zip(batch_nodes, policy_logits_list, values_list):
        node.expand(policy_logits)
        backpropagate(node, value)


def run_mcts(root, model, simulations=1600, c_puct=1.0, batch_size=16):
    """
    Run MCTS from 'root' for 'simulations' iterations, doing batched expansions
    of up to 'batch_size' leaves per neural net call.
    """
    if root.board.is_game_over():
        return

    # Expand root once
    input_data = np.expand_dims(encode_board(root.board), axis=0)
    policy, val = model.predict(input_data, verbose=0)
    root.expand(policy[0])
    backpropagate(root, val[0][0])

    sim_count = 0
    while sim_count < simulations:
        pending_expansions = []

        # Up to batch_size selection steps
        for _ in range(batch_size):
            if sim_count >= simulations:
                break
            leaf = simulate_one(root, pending_expansions, c_puct)
            sim_count += 1

        if not pending_expansions:
            continue

        # Prepare a single batch inference
        boards_encoded = [
            encode_board(leaf_node.board) for leaf_node in pending_expansions
        ]
        inp = np.array(boards_encoded, dtype=np.float32)
        policy_logits_batch, value_batch = model.predict(inp, verbose=0)

        # Expand children and backprop for each leaf
        expand_and_backprop(pending_expansions, policy_logits_batch, value_batch)


def mcts_search(board, model, simulations=1600, c_puct=1.0, batch_size=16):
    """
    Create a root node for 'board', run MCTS, pick the best move by visit_count.
    Returns the best move.
    """
    root = MCTSNode(board=board, parent=None, prior=1.0)
    if not board.is_game_over():
        run_mcts(root, model, simulations=simulations, c_puct=c_puct, batch_size=batch_size)

    best_move, best_child = None, None
    best_visits = -1
    for move, child in root.children.items():
        if child.visit_count > best_visits:
            best_visits = child.visit_count
            best_move = move
            best_child = child

    return best_move


def get_mcts_move(board, model, simulations=1024, batch_size=256):
    """
    High-level convenience function to run MCTS on 'board' for 'simulations'
    and return the best move. Fallback to any legal move if no child found.
    """
    if board.is_game_over():
        return None

    move = mcts_search(
        board=board,
        model=model,
        simulations=simulations,
        c_puct=1.0,
        batch_size=batch_size
    )

    if move is None:
        # fallback
        legal_moves = list(board.legal_moves)
        return legal_moves[0] if legal_moves else None
    return move


###############################################################################
# ADDITIONAL HELPER METHODS REQUESTED
###############################################################################

def get_mcts_root(board, model, simulations=500, batch_size=16, c_puct=1.0):
    """
    Create and return the fully searched MCTS root node for 'board'.
    Useful if you want to examine or manipulate the root after MCTS.
    """
    root = MCTSNode(board=board, parent=None, prior=1.0)
    if not board.is_game_over():
        run_mcts(root, model, simulations=simulations, c_puct=c_puct, batch_size=batch_size)
    return root


def extract_policy_vector(root, num_moves):
    policy_vec = np.zeros(num_moves, dtype=np.float32)
    total_visits = 0.0
    for move, child in root.children.items():
        idx = move_to_index(move)
        if 0 <= idx < num_moves:
            # Make sure child's visit_count is finite
            if not np.isfinite(child.visit_count):
                # You can log or raise an error if it's NaN
                continue
            policy_vec[idx] = child.visit_count
            total_visits += child.visit_count

    # Normalizing if total_visits > 0
    if total_visits > 0.0:
        policy_vec /= total_visits
    else:
        # Fallback: either leave it all zeros (which is invalid)
        # or distribute uniform among the known children:
        legal_moves = list(root.children.keys())
        for move in legal_moves:
            idx = move_to_index(move)
            if 0 <= idx < num_moves:
                policy_vec[idx] = 1.0 / len(legal_moves)

    return policy_vec


def select_move_from_root(root, temperature=1.0):
    """
    Select a move from the root node's children according to visit counts,
    applying 'temperature'. If temperature=0, picks argmax. Otherwise,
    picks stochastically in proportion to (visit_count ^ (1/temperature)).

    Returns a (move, chosen_child) pair, or (None, None) if board is game-over.
    """
    if root.board.is_game_over() or not root.children:
        return None, None

    # Build array of children with their visits
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float32)

    if temperature <= 1e-6:
        # Argmax (no randomness)
        best_idx = np.argmax(visits)
        best_move = moves[best_idx]
        return best_move, root.children[best_move]
    else:
        # Weighted random
        # (visit_count ^ (1 / temperature))
        exponentiated = np.power(visits, 1.0 / temperature)
        total = np.sum(exponentiated)
        if total < 1e-12:
            # fallback if all zero
            best_idx = np.random.choice(len(moves))
            best_move = moves[best_idx]
            return best_move, root.children[best_move]
        probs = exponentiated / total
        choice_idx = np.random.choice(len(moves), p=probs)
        chosen_move = moves[choice_idx]
        return chosen_move, root.children[chosen_move]