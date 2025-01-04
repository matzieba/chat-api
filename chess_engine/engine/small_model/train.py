import os
import random
import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import numpy as np
from tensorboard.compat import tf
from tensorflow.keras.optimizers import Adam

from chess_engine.engine.small_model.environment import (
    encode_board,
    build_move_mask,
    NUM_MOVES,
    index_to_move, apply_temperature, inject_dirichlet_noise,
)
from chess_engine.engine.small_model.mcts import (
    MCTSNode,
    expand_node,
    run_mcts,
    build_policy_vector,
)
from chess_engine.engine.small_model.mcts_batch import run_mcts_batch
from chess_engine.engine.small_model.model import create_small_chess_model


MAX_BUFFER_SIZE = 500000


def load_chess_model(filepath, learning_rate=1e-3):
    """
    Load a previously saved Keras model and recompile it with the same structure
    (policy_output, value_output).
    """
    model = tf.keras.models.load_model(filepath, compile=False)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            "policy_output": "categorical_crossentropy",
            "value_output": "mean_squared_error",
        },
        loss_weights={"policy_output": 1.0, "value_output": 1.0},
        metrics={
            "policy_output": [
                tf.keras.metrics.CategoricalAccuracy(name="policy_accuracy"),
                tf.keras.metrics.TopKCategoricalAccuracy(
                    k=5, name="policy_top5_accuracy"
                ),
            ]
        },
    )
    return model


def train_model(model, training_data, batch_size=64, epochs=1):
    """
    Train on the full policy distribution (policy_target as shape (NUM_MOVES,) instead of argmax).
    """
    encoded_boards, masks, policy_targets, value_targets = zip(*training_data)

    encoded_boards = np.array(encoded_boards, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    policy_targets = np.array(policy_targets, dtype=np.float32)  # (batch, NUM_MOVES)
    value_targets = np.array(value_targets, dtype=np.float32)    # (batch,)

    # Expand value_targets to match the output shape
    value_targets = value_targets.reshape(-1, 1)

    history = model.fit(
        x=[encoded_boards, masks],
        y={"policy_output": policy_targets, "value_output": value_targets},
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
    )
    return history


def self_play_game(
    model_path,
    num_simulations,
    max_moves=800,
    rng_seed=None,
    random_opening_moves=10,
    temperature=1.0,
    alpha=0.03,
    eps=0.25
):
    """
    Plays a single self-play game using MCTS, loading the model within this process.
    Returns a list of (board_encoded, mask, policy_vec, game_result).
    """
    if rng_seed is not None:
        random.seed(rng_seed)
        np.random.seed(rng_seed)

    # Each process re-loads the model from disk here
    model = load_chess_model(model_path)

    board = chess.Board()
    for _ in range(random_opening_moves):
        if board.is_game_over():
            break
        # Pick a random legal move
        move = random.choice(list(board.legal_moves))
        board.push(move)

    game_history = []

    while not board.is_game_over() and board.fullmove_number <= max_moves:
        # Create a root node
        root_node = MCTSNode(board)
        expand_node(root_node, model)
        # run_mcts(root_node, model, num_simulations=num_simulations)
        run_mcts_batch(root_node, model, num_simulations=10, batch_size=64)
        encoded = encode_board(board)
        mask = build_move_mask(board)
        inject_dirichlet_noise(root_node, alpha=alpha, eps=eps)
        policy_target = build_policy_vector(root_node)

        policy_temp = apply_temperature(policy_target, temperature=temperature)

        # Sample a move from the policy distribution for exploration
        move_idx = np.random.choice(np.arange(NUM_MOVES), p=policy_target)
        move = index_to_move(move_idx, board)

        if move not in board.legal_moves:
            # Fallback if the sampled move is illegal
            move = random.choice(list(board.legal_moves))

        board.push(move)
        game_history.append((encoded, mask, policy_target))

    # Final outcome from White's perspective (1=White win, -1=Black win, 0=Draw)
    result_str = board.result()  # e.g., "1-0", "0-1", "1/2-1/2"
    if result_str == "1-0":
        outcome = 1.0
    elif result_str == "0-1":
        outcome = -1.0
    else:
        outcome = 0.0
    print(f"Process {multiprocessing.current_process().name} game finished with result: {result_str}")

    # Assign the same outcome for all states (simple approach)
    training_data = [
        (enc, m, pi, outcome) for (enc, m, pi) in game_history
    ]
    return training_data


def main_self_play_loop(
    num_iterations,
    games_per_iteration,
    num_simulations,
    model_path="best_trained_model_small_cnn_l2.keras",
    learning_rate=1e-3,
):
    """
    Main loop that:
    1) Loads/creates a model
    2) Generates self-play data in parallel (using a ProcessPoolExecutor)
    3) Trains the model
    4) Saves the model back to the same path after training
    """
    # We need to ensure the 'spawn' method is used on some platforms (esp. if on Windows),
    # so that TF can be safely loaded in child processes.
    # You can force it with:
    # multiprocessing.set_start_method('spawn', force=True)

    # Create or load the model in the main process
    if os.path.isfile(model_path):
        print(f"Loading model from: {model_path}")
        model = load_chess_model(model_path, learning_rate=learning_rate)
    else:
        print("Creating a new model (no checkpoint found).")
        model = create_small_chess_model()
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                "policy_output": "categorical_crossentropy",
                "value_output": "mean_squared_error",
            },
            loss_weights={"policy_output": 1.0, "value_output": 1.0},
            metrics={
                "policy_output": [
                    tf.keras.metrics.CategoricalAccuracy(name="policy_accuracy"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="policy_top5_accuracy"),
                ]
            },
        )
        print("Model summary:")
        model.summary()

    # A replay buffer to store (board_encoding, mask, policy, value) from many games
    REPLAY_BUFFER = deque(maxlen=MAX_BUFFER_SIZE)

    num_workers = os.cpu_count() or 4
    print(f"Using up to {num_workers} processes for self-play.")

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # 1. Generate new self-play data (in parallel, via ProcessPoolExecutor)
        iteration_data = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Optionally fix random seeds for reproducibility
            # rng_seeds = [None]*games_per_iteration  # or use a list of actual seeds
            futures = [
                executor.submit(
                    self_play_game,
                    model_path,
                    num_simulations,
                    # max_moves=500,
                    # rng_seed=rng_seeds[i],
                )
                for i in range(games_per_iteration)
            ]

            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    game_data = future.result()
                    iteration_data.extend(game_data)
                    print(f"Completed game {idx}/{games_per_iteration}")
                except Exception as e:
                    print(f"Game {idx} generated an exception: {e}")

        # 2. Add the new data to the replay buffer
        for item in iteration_data:
            REPLAY_BUFFER.append(item)

        # 3. Sample from the replay buffer
        batch_size_to_train = 9500000  # The maximum number of samples we train on each iteration
        if len(REPLAY_BUFFER) < batch_size_to_train:
            sample_data = list(REPLAY_BUFFER)
            print(f"Not enough data in buffer. Using all {len(sample_data)} samples for training.")
        else:
            sample_data = random.sample(REPLAY_BUFFER, batch_size_to_train)
            print(f"Sampling {batch_size_to_train} samples from replay buffer.")

        # 4. Train the model on the sampled data
        print(f"Training on {len(sample_data)} samples...")
        train_model(model, sample_data, batch_size=64, epochs=1)

        # 5. Save the updated model back to the same path it was loaded from
        model.save(model_path)
        print(f"Model updated and saved: {model_path}")

    print("Training loop complete!")


if __name__ == "__main__":
    # Optionally set multiprocessing to spawn, if needed.
    # multiprocessing.set_start_method("spawn", force=True)

    main_self_play_loop(
        num_iterations=4,
        games_per_iteration=128,
        num_simulations=10,
        model_path='best_trained_model_small_cnn_l2.keras',
        learning_rate=1e-3,
    )