import chess
import numpy as np
import tensorflow as tf

from chess_api.heuristics import evaluate_with_heuristics
from chess_api.mcts import MCTSNode
from chess_engine.engine.small_model.transformer.environment import (
    encode_board,

    move_to_index
)

def evaluate_batch_in_worker(boards, model):
    """
    Single-process version that evaluates a batch of boards in one forward pass.
    We do NOT load the model inside this function anymore; the model is passed in.
    """
    policy_dicts = []
    values = []

    # Encode all boards in a batch
    encoded_batch = []

    for board in boards:
        enc = encode_board(board)
        encoded_batch.append(enc)
    encoded_batch = np.array(encoded_batch, dtype=np.float32)
    # Single forward pass for the entire batch
    policy_logits_batch, net_value_batch = model.predict(
        {"board_input": encoded_batch},
        verbose=0
    )

    # Convert each board's outputs
    for i, board in enumerate(boards):
        policy_logits = policy_logits_batch[i]  # shape (NUM_MOVES,)
        net_value = float(net_value_batch[i])   # shape ()
        # Incorporate basic heuristics into the value
        heur_value = evaluate_with_heuristics(board)
        alpha = 0.3
        # final_value = net_value + alpha * heur_value
        final_value = net_value
        # Build a dictionary {move: prior}
        legal_moves = list(board.legal_moves)
        move_policy = {}
        for move in legal_moves:
            idx = move_to_index(move)
            move_policy[move] = policy_logits[idx]

        # Normalize or fallback to uniform
        total_p = sum(move_policy.values())
        if total_p > 1e-12:
            for m in move_policy:
                move_policy[m] /= total_p
        else:
            for m in move_policy:
                move_policy[m] = 1.0 / len(move_policy)

        policy_dicts.append(move_policy)
        values.append(final_value)

    return policy_dicts, values


def run_mcts_leaf_parallel(
    root_node,
    model,
    num_simulations=100,
    batch_size=16,
    c_puct=3.0
):
    """
    Single-process "leaf-parallel" MCTS, but we batch expansions into chunks
    and run them through 'evaluate_batch_in_worker' in a single forward pass.
    """
    sims_done = 0
    while sims_done < num_simulations:
        expansions = []
        for _ in range(batch_size):
            if sims_done >= num_simulations:
                break
            node = root_node
            path = [node]
            # Selection
            while node.children:
                node = select_child(node, c_puct)
                path.append(node)
            # Leaf node
            if not node.board.is_game_over():
                expansions.append((node, node.board.copy(), path))
            else:
                # Terminal node, propagate result
                outcome_value = evaluate_terminal(node.board)
                backpropagate_multi(path, outcome_value)
            sims_done += 1

        if not expansions:
            break

        # Evaluate all expansions in a single batch
        boards_to_eval = [b for (_, b, _) in expansions]
        policy_dicts, values = evaluate_batch_in_worker(boards_to_eval, model)

        # Expand each leaf and backpropagate
        for i, (leaf_node, leaf_board, path) in enumerate(expansions):
            move_policy = policy_dicts[i]
            child_value = values[i]
            expand_node(leaf_node, move_policy)
            backpropagate_multi(path, child_value)

    return root_node

###############################################################################
#                               HELPER METHODS
###############################################################################
def select_child(node, c_puct):
    """Select the child with largest Q + U."""
    best_score = -float('inf')
    best_child = None
    for _, child in node.children.items():
        q = child.q_value
        u = c_puct * child.prior * ((node.visit_count ** 0.5) / (1 + child.visit_count))
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child
    return best_child

def evaluate_terminal(board):
    """Returns the final value from perspective of side-to-move at terminal."""
    result = board.result()
    if result == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    elif result == "0-1":
        return 1.0 if board.turn == chess.BLACK else -1.0
    else:
        return 0.0

def expand_node(node, move_policy):
    for move, prior in move_policy.items():
        board_copy = node.board.copy()
        board_copy.push(move)
        if move not in node.children:
            child_node = MCTSNode(board=board_copy, parent=node, prior=prior)
            node.children[move] = child_node

def backpropagate_multi(path, value):
    """
    Backpropagate 'value' along the path. Flip sign each level.
    """
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value