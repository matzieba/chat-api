import os
import random
import pickle
from collections import deque

import chess
import chess.engine
import numpy as np
from tensorboard.compat import tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ---------------
# Local imports from your code
# ---------------
from openings import opening_fens
from chess_api.heuristics import evaluate_with_heuristics
from environment import (
    encode_board,
    build_move_mask,
    inject_dirichlet_noise,
    apply_temperature,
    index_to_move,
    NUM_MOVES,
    move_to_index
)
from chess_api.mcts import MCTSNode, build_policy_vector
from chess_api.parallell_mcts import run_mcts_leaf_parallel

MAX_BUFFER_SIZE = 400000
REPLAY_BUFFER_FILE = "replay_buffer.pkl"
REPLAY_BUFFER_FILE_STOCKFISH = "replay_buffer_stock_fish_mixed.pkl"

###############################################################################
# Medium Model Definition (unchanged)
###############################################################################
def create_medium_chess_model(
    board_shape=(64, 14),
    num_moves=NUM_MOVES,
    num_filters=64,
    num_res_blocks=3,
    learning_rate=2e-3,
    l2_reg=1e-5
):
    input_board = layers.Input(shape=board_shape, name="input_board")
    mask_input = layers.Input(shape=(num_moves,), name="mask_input")

    # Reshape to (8,8,14)
    x = layers.Reshape((8, 8, board_shape[-1]))(input_board)
    x = layers.Conv2D(
        filters=num_filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=l2(l2_reg)
    )(x)

    for _ in range(num_res_blocks):
        residual = x
        x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False, kernel_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False, kernel_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Activation("relu")(x)

    # Policy head
    policy_conv = layers.Conv2D(
        filters=2,
        kernel_size=1,
        activation="relu",
        kernel_regularizer=l2(l2_reg)
    )(x)
    policy_flat = layers.Flatten()(policy_conv)
    logits = layers.Dense(num_moves, kernel_regularizer=l2(l2_reg), name="logits")(policy_flat)

    # Mask out illegal moves with a large negative
    negative_inf = -1e9
    masked_logits = logits + (1.0 - mask_input) * negative_inf
    policy_output = layers.Softmax(name="policy_output")(masked_logits)

    # Value head
    value_conv = layers.Conv2D(
        filters=1,
        kernel_size=1,
        activation="relu",
        kernel_regularizer=l2(l2_reg)
    )(x)
    value_flat = layers.Flatten()(value_conv)
    value_dense = layers.Dense(128, activation="relu", kernel_regularizer=l2(l2_reg))(value_flat)
    value_output = layers.Dense(1, activation="tanh", name="value_output")(value_dense)

    model = Model(inputs=[input_board, mask_input], outputs=[policy_output, value_output])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            "policy_output": "categorical_crossentropy",
            "value_output": "mean_squared_error"
        },
        loss_weights={
            "policy_output": 1.0,
            "value_output": 1.0
        },
        metrics={
            "policy_output": [
                tf.keras.metrics.CategoricalAccuracy(name="policy_accuracy"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="policy_top5_accuracy")
            ]
        }
    )
    return model

###############################################################################
# Training on (board, mask, policy_target, value_target)
###############################################################################
def train_model(model, training_data, batch_size=64, epochs=3):
    encoded_boards, masks, policy_targets, value_targets = zip(*training_data)

    encoded_boards = np.array(encoded_boards, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    policy_targets = np.array(policy_targets, dtype=np.float32)
    value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

    history = model.fit(
        x=[encoded_boards, masks],
        y={
            "policy_output": policy_targets,
            "value_output": value_targets
        },
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    return history

###############################################################################
# Self-Play Game
###############################################################################
def self_play_game(
    model,
    temperature,
    alpha,
    eps,
    num_simulations,
    opening_fens=opening_fens,
    opening_probability=0.9
):
    """
    Generate one game via self-play. Returns a list of (encoded_board, mask, policy_vec, final_value)
    where final_value is from the perspective of the side that was to move in that position.
    """
    # Randomly pick an opening or start from scratch
    use_opening = (opening_fens is not None) and (random.random() < opening_probability)
    if use_opening:
        fen = random.choice(opening_fens)
        board = chess.Board(fen)
    else:
        board = chess.Board()

    game_history = []

    while not board.is_game_over() and len(board.move_stack) < 600:
        side_to_move = board.turn  # ADDED: record who is about to move
        if num_simulations == 1:
            # Quick heuristic-based move
            enc = encode_board(board)
            mask = build_move_mask(board)
            legal_moves = list(board.legal_moves)

            best_move = None
            best_value = -9999.0
            sign = 1 if board.turn == chess.WHITE else -1

            for mv in legal_moves:
                board.push(mv)
                val = sign * evaluate_with_heuristics(board)
                board.pop()
                if val > best_value:
                    best_value = val
                    best_move = mv

            policy_target = np.zeros(NUM_MOVES, dtype=np.float32)
            if best_move:
                idx = move_to_index(best_move)
                policy_target[idx] = 1.0
                board.push(best_move)
            else:
                best_move = random.choice(legal_moves)
                board.push(best_move)
                idx = move_to_index(best_move)
                policy_target[idx] = 1.0

            # Store position
            game_history.append((enc, mask, policy_target, side_to_move))  # ADDED side_to_move
        else:
            # MCTS
            root_node = MCTSNode(board)
            best_root = run_mcts_leaf_parallel(
                root_node=root_node,
                model=model,
                num_simulations=num_simulations,
                batch_size=32,
                c_puct=2.5
            )

            enc = encode_board(board)
            mask = build_move_mask(board)
            if len(board.move_stack) < 10:
                inject_dirichlet_noise(best_root, alpha=alpha, eps=eps)
            policy_target = build_policy_vector(best_root)

            # Temperature
            if len(board.move_stack) < 5:
                current_temperature = temperature
            else:
                current_temperature = 0.0
            policy_target = apply_temperature(policy_target, temperature=current_temperature)

            # Sample a move
            move_idx = np.random.choice(np.arange(NUM_MOVES), p=policy_target)
            move = index_to_move(move_idx, board)
            if move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))

            board.push(move)
            game_history.append((enc, mask, policy_target, side_to_move))  # ADDED side_to_move

    # Game outcome from White's perspective
    result_str = board.result()
    if result_str == "1-0":
        white_outcome = 1.0
    elif result_str == "0-1":
        white_outcome = -1.0
    else:
        white_outcome = 0.0

    print(f"Self-play game finished with result: {result_str}")

    # ADDED: Build training_data with perspective flipping
    training_data = []
    for (enc, m, pi, side) in game_history:
        # If side == chess.WHITE, value is white_outcome
        # If side == chess.BLACK, value is -white_outcome
        # This ensures each position is labeled from the side-to-move perspective
        value_label = white_outcome if side == chess.WHITE else -white_outcome
        training_data.append((enc, m, pi, value_label))

    return training_data

###############################################################################
# Model vs. Stockfish Game
###############################################################################
def play_game_vs_stockfish(
    model,
    stockfish_path,
    stockfish_params,
    temperature=1.2,
    alpha=0.1,
    eps=0.2,
    num_simulations=10,
    opening_fens=opening_fens,
    opening_probability=0.1
):
    """
    One game between the model (using MCTS) and Stockfish.
    We randomly pick who plays White. We only collect positions from the model's moves,
    labeling them from the model's perspective (i.e. +1 if the model eventually won,
    -1 if it lost) rather than side-to-move perspective.
    """
    # Open engine
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    # Configure Stockfish parameters
    for param, value in stockfish_params.items():
        engine.configure({param: value})

    # Optionally pick an opening or start from scratch
    use_opening = (opening_fens is not None) and (random.random() < opening_probability)
    if use_opening:
        fen = random.choice(opening_fens)
        board = chess.Board(fen)
    else:
        board = chess.Board()

    # Randomly decide who will play White
    model_white = bool(random.getrandbits(1))
    print(f"Model plays {'White' if model_white else 'Black'} against Stockfish.")

    game_history = []

    while not board.is_game_over() and len(board.move_stack) < 600:
        if (board.turn == chess.WHITE and model_white) or (board.turn == chess.BLACK and not model_white):
            # Model's turn
            if board.is_game_over():
                break

            if num_simulations == 1:
                # Quick heuristic-based move
                enc = encode_board(board)
                mask = build_move_mask(board)
                legal_moves = list(board.legal_moves)

                best_move = None
                best_value = -9999.0
                sign = 1 if board.turn == chess.WHITE else -1

                for mv in legal_moves:
                    board.push(mv)
                    val = sign * evaluate_with_heuristics(board)
                    board.pop()
                    if val > best_value:
                        best_value = val
                        best_move = mv

                policy_target = np.zeros(NUM_MOVES, dtype=np.float32)
                if best_move:
                    idx = move_to_index(best_move)
                    policy_target[idx] = 1.0
                    board.push(best_move)
                else:
                    best_move = random.choice(legal_moves)
                    board.push(best_move)
                    idx = move_to_index(best_move)
                    policy_target[idx] = 1.0

                game_history.append((enc, mask, policy_target))
            else:
                # MCTS
                root_node = MCTSNode(board)
                best_root = run_mcts_leaf_parallel(
                    root_node=root_node,
                    model=model,
                    num_simulations=num_simulations,
                    batch_size=32,
                    c_puct=3.5
                )

                enc = encode_board(board)
                mask = build_move_mask(board)
                if len(board.move_stack) < 10:
                    inject_dirichlet_noise(best_root, alpha=alpha, eps=eps)
                policy_target = build_policy_vector(best_root)

                # Temperature
                if len(board.move_stack) < 20:
                    current_temperature = temperature
                else:
                    current_temperature = 0.0
                policy_target = apply_temperature(policy_target, temperature=current_temperature)

                # Sample a move
                move_idx = np.random.choice(np.arange(NUM_MOVES), p=policy_target)
                move = index_to_move(move_idx, board)
                if move not in board.legal_moves:
                    move = random.choice(list(board.legal_moves))

                board.push(move)
                game_history.append((enc, mask, policy_target))
        else:
            # Stockfish's turn
            if board.is_game_over():
                break
            stockfish_move = engine.play(board, chess.engine.Limit(depth=8)).move
            board.push(stockfish_move)

    engine.quit()

    # Game outcome from the model's perspective
    result_str = board.result()
    if result_str == "1-0":
        outcome = 1.0 if model_white else -1.0
    elif result_str == "0-1":
        outcome = -1.0 if model_white else 1.0
    else:
        outcome = 0.0

    print(f"Stockfish game finished with result: {result_str}")

    # Attach final result to model's positions
    training_data = [(enc, m, pi, outcome) for (enc, m, pi) in game_history]
    return training_data

###############################################################################
# Model vs. Second Saved Model
###############################################################################
def play_game_vs_second_model(
    main_model,
    second_model,
    temperature=1.2,
    alpha=0.1,
    eps=0.2,
    num_simulations=10,
    opening_fens=opening_fens,
    opening_probability=0.2
):
    """
    One game between main_model and second_model, both using MCTS.
    We collect positions/training data only from main_model's moves,
    labeling them from main_model's perspective.
    """
    # Optionally pick an opening or start from scratch
    use_opening = (opening_fens is not None) and (random.random() < opening_probability)
    if use_opening:
        fen = random.choice(opening_fens)
        board = chess.Board(fen)
    else:
        board = chess.Board()

    # Randomly decide who plays White
    main_model_white = bool(random.getrandbits(1))
    print(
        "Main model plays {} vs second model.".format(
            "White" if main_model_white else "Black"
        )
    )

    game_history = []

    while not board.is_game_over() and len(board.move_stack) < 600:
        if (board.turn == chess.WHITE and main_model_white) or (board.turn == chess.BLACK and not main_model_white):
            # Main model's turn
            if board.is_game_over():
                break

            root_node = MCTSNode(board)
            best_root = run_mcts_leaf_parallel(
                root_node=root_node,
                model=main_model,
                num_simulations=num_simulations,
                batch_size=32,
                c_puct=3.5
            )

            enc = encode_board(board)
            mask = build_move_mask(board)
            if len(board.move_stack) < 10:
                inject_dirichlet_noise(best_root, alpha=alpha, eps=eps)
            policy_target = build_policy_vector(best_root)

            if len(board.move_stack) < 20:
                current_temperature = temperature
            else:
                current_temperature = 0.0
            policy_target = apply_temperature(policy_target, temperature=current_temperature)

            move_idx = np.random.choice(np.arange(NUM_MOVES), p=policy_target)
            move = index_to_move(move_idx, board)
            if move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))

            board.push(move)
            game_history.append((enc, mask, policy_target))
        else:
            # Second model's turn
            if board.is_game_over():
                break

            root_node = MCTSNode(board)
            best_root = run_mcts_leaf_parallel(
                root_node=root_node,
                model=second_model,
                num_simulations=1,
                batch_size=32,
                c_puct=3.5
            )

            # For the second model, we don't collect training data for our buffer
            policy_target = build_policy_vector(best_root)
            if len(board.move_stack) < 20:
                current_temperature = temperature
            else:
                current_temperature = 0.0
            policy_target = apply_temperature(policy_target, temperature=current_temperature)

            move_idx = np.random.choice(np.arange(NUM_MOVES), p=policy_target)
            move = index_to_move(move_idx, board)
            if move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))

            board.push(move)

    # Game outcome from main_model's perspective
    result_str = board.result()
    if result_str == "1-0":
        outcome = 1.0 if main_model_white else -1.0
    elif result_str == "0-1":
        outcome = -1.0 if main_model_white else 1.0
    else:
        outcome = 0.0

    print(f"Model vs. second model game finished with result: {result_str}")

    # Attach final result to main model's positions
    training_data = [(enc, m, pi, outcome) for (enc, m, pi) in game_history]
    return training_data

###############################################################################
# Main Self-Play + Extra Opponents Training Loop
###############################################################################
def main_self_play_loop(
    num_iterations,
    games_per_iteration,
    num_simulations,
    model_path,
    learning_rate,
    stockfish_ratio=0.2,
    stockfish_path=None,
    stockfish_params=None,
    second_model_path=None,
    second_model_ratio=0.2,
):
    """
    In each iteration:
      - A portion of games is self-play,
      - A portion vs. Stockfish (stockfish_ratio),
      - A portion vs. second saved model (second_model_ratio).
    The total of (stockfish_ratio + second_model_ratio) can be less than 1.0,
    meaning the remainder of games is self-play.
    """

    # Default Stockfish params if not provided
    if stockfish_params is None:
        stockfish_params = {"Skill Level": 5}

    # Step A: Load or create main model
    if model_path and os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        main_model = tf.keras.models.load_model(model_path, compile=False)
        main_model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                "policy_output": "categorical_crossentropy",
                "value_output": "mean_squared_error"
            },
            loss_weights={
                "policy_output": 1.0,
                "value_output": 1.0
            },
            metrics={
                "policy_output": [
                    tf.keras.metrics.CategoricalAccuracy(name="policy_accuracy"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="policy_top5_accuracy")
                ]
            }
        )
    else:
        print("Creating a new main model (no existing checkpoint found).")
        main_model = create_medium_chess_model(
            board_shape=(64, 14),
            num_moves=NUM_MOVES,
            num_filters=128,
            num_res_blocks=8,
            learning_rate=learning_rate,
            l2_reg=1e-5
        )
        main_model.save(model_path)
        main_model.summary()

    # Optionally load second model (if provided)
    second_model = None
    if second_model_path and os.path.isfile(second_model_path):
        print(f"Loading second model from {second_model_path}")
        second_model = tf.keras.models.load_model(second_model_path, compile=False)
        # We don't necessarily need to compile second model if only using inference,
        # but you could do so:
        second_model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={"policy_output": "categorical_crossentropy", "value_output": "mean_squared_error"}
        )

    # Step B: Load or create replay buffer
    if os.path.isfile(REPLAY_BUFFER_FILE_STOCKFISH):
        print(f"Loading replay buffer from {REPLAY_BUFFER_FILE_STOCKFISH}")
        with open(REPLAY_BUFFER_FILE_STOCKFISH, "rb") as f:
            replay_buffer = pickle.load(f)
    else:
        print("No existing replay buffer found. Creating a new one.")
        replay_buffer = deque(maxlen=MAX_BUFFER_SIZE)

    print(f"Replay buffer size is currently {len(replay_buffer)}.")

    # Step C: Self-play + extra opponents for multiple iterations
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
        iteration_data = []

        for g_idx in range(games_per_iteration):
            # Randomly select the opponent type based on ratio
            r = random.random()
            if (stockfish_path is not None) and (r < stockfish_ratio):
                # Play vs. Stockfish
                data = play_game_vs_stockfish(
                    model=main_model,
                    stockfish_path=stockfish_path,
                    stockfish_params=stockfish_params,
                    temperature=1,
                    alpha=0.1,
                    eps=0.2,
                    num_simulations=num_simulations,
                    opening_fens=opening_fens,
                    opening_probability=0.0
                )
                game_type = "Stockfish"
            elif (second_model is not None) and (r < stockfish_ratio + second_model_ratio):
                # Play vs second saved model
                data = play_game_vs_second_model(
                    main_model=main_model,
                    second_model=second_model,
                    temperature=1,
                    alpha=0.1,
                    eps=0.2,
                    num_simulations=num_simulations,
                    opening_fens=opening_fens,
                    opening_probability=0.0
                )
                game_type = "Second model"
            else:
                # Normal self-play (AlphaZero style for both sides)
                data = self_play_game(
                    model=main_model,
                    temperature=1,
                    alpha=0.1,
                    eps=0.2,
                    num_simulations=num_simulations,
                    opening_fens=opening_fens,
                    opening_probability=0.6
                )
                game_type = "Self-play"

            iteration_data.extend(data)
            print(f"Completed {game_type} game {g_idx + 1}/{games_per_iteration}")

        # Add new data to buffer
        for item in iteration_data:
            replay_buffer.append(item)

        print(f"Replay buffer size after iteration: {len(replay_buffer)}")

        # Step D: Sample from the buffer
        if len(replay_buffer) < 100000:
            sample_data = list(replay_buffer)
            print(f"Using all {len(sample_data)} samples (buffer still small).")
        else:
            sample_data = random.sample(replay_buffer, 100000)
            print(f"Sampling 100000 positions from replay buffer of size {len(replay_buffer)}.")

        # Step E: Train the main model
        print(f"Training on {len(sample_data)} samples...")
        train_model(main_model, sample_data, batch_size=32, epochs=4)

        # Step F: Save updated model
        main_model.save(model_path)
        print(f"Main model saved to {model_path}")

    # Step G: Save the replay buffer
    with open(REPLAY_BUFFER_FILE_STOCKFISH, "wb") as f:
        pickle.dump(replay_buffer, f)
    print(f"Replay buffer saved to {REPLAY_BUFFER_FILE_STOCKFISH}")
    print("All iterations complete!")


if __name__ == "__main__":
    main_self_play_loop(
        num_iterations=2,
        games_per_iteration=300,
        num_simulations=10,
        model_path="/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/small_model/transformer/best_trained_model_training_against_stockfish.keras",
        learning_rate=5e-4,
        stockfish_ratio=0.9,
        stockfish_path="/Users/mateuszzieba/Desktop/dev/chess/stockfish/src/stockfish",
        stockfish_params={"Skill Level": 5},
        second_model_path=(
            "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/small_model/transformer/best_trained_model_training_against_stockfish_copy.keras"
        ),
        second_model_ratio=0.0,
    )