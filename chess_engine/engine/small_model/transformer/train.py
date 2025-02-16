import os
import random
import pickle
from collections import deque

import chess
import chess.engine
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# -------------------------------------------------------------------------
# Local imports (placeholders: replace with your actual code)
# -------------------------------------------------------------------------
from openings import opening_fens
from chess_api.heuristics import evaluate_with_heuristics
from environment import (
    encode_board,
    inject_dirichlet_noise,
    apply_temperature,
    index_to_move,
    move_to_index
)
# MCTS
from chess_api.mcts import (
    get_mcts_root,
    extract_policy_vector,
)

# -------------------------------------------------------------------------
# 1) Global constants & file paths
# -------------------------------------------------------------------------
MAX_BUFFER_SIZE = 400_000
REPLAY_BUFFER_FILE_STOCKFISH = "replay_buffer_stock_fish_mixed.pkl"

# IMPORTANT: This should match the policy head dimension in your supervised model (often 4096 for chess).
NUM_MOVES = 8192


def train_model(model, training_data, batch_size=64, epochs=3):
    encoded_boards, policy_vecs, value_targets = zip(*training_data)

    encoded_boards = np.array(encoded_boards, dtype=np.float32)
    policy_vecs = np.array(policy_vecs, dtype=np.float32)
    value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

    history = model.fit(
        x=encoded_boards,
        y={
            "policy_head": policy_vecs,     # distribution label
            "value_head": value_targets
        },
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    return history


# -------------------------------------------------------------------------
# 4) Self-play: produce a full policy_vector from MCTS, store it
# -------------------------------------------------------------------------
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
    Generate one self-play game. Returns a list of:
       [(encoded_board, policy_vector, final_value), ...]
    """
    use_opening = (opening_fens is not None) and (random.random() < opening_probability)
    board = chess.Board(random.choice(opening_fens)) if use_opening else chess.Board()

    game_history = []  # will store (encoded_board, policy_vec, side_to_move)
    while not board.is_game_over() and len(board.move_stack) < 600:
        side_to_move = board.turn

        if num_simulations == 1:
            # Quick heuristic approach → create a 1-hot policy vector for the chosen move
            enc = encode_board(board)
            legal_moves = list(board.legal_moves)
            best_move = None
            best_value = -9999.0
            sign = 1 if side_to_move == chess.WHITE else -1

            for mv in legal_moves:
                board.push(mv)
                val = sign * evaluate_with_heuristics(board)
                board.pop()
                if val > best_value:
                    best_value = val
                    best_move = mv

            if best_move is None:
                best_move = random.choice(legal_moves)

            move_label = move_to_index(best_move)
            if not (0 <= move_label < NUM_MOVES):
                # fallback
                possible_indices = [
                    move_to_index(mv)
                    for mv in legal_moves
                    if 0 <= move_to_index(mv) < NUM_MOVES
                ]
                move_label = random.choice(possible_indices) if possible_indices else 0

            policy_vec = np.zeros(NUM_MOVES, dtype=np.float32)
            policy_vec[move_label] = 1.0

            game_history.append((enc, policy_vec, side_to_move))
            board.push(best_move)

        else:
            # MCTS approach
            enc = encode_board(board)
            root = get_mcts_root(
                board=board,
                model=model,
                simulations=num_simulations,
                batch_size=512,
                c_puct=1
            )
            # if len(board.move_stack) < 10:
            #     inject_dirichlet_noise(root, alpha=alpha, eps=eps)

            policy_vector = extract_policy_vector(root, num_moves=NUM_MOVES)

            # Apply temperature in early moves
            # if len(board.move_stack) < 5:
            #     current_temperature = temperature
            # else:
            #     current_temperature = 0.0
            # policy_vector = apply_temperature(policy_vector, current_temperature)

            # Store the full policy distribution
            game_history.append((enc, policy_vector, side_to_move))

            # Sample a move from that distribution
            move_label = np.random.choice(np.arange(NUM_MOVES), p=policy_vector)
            move = index_to_move(move_label, board)
            if (move not in board.legal_moves) or (move is None):
                # fallback
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break  # no moves
                move = random.choice(legal_moves)
                move_label = move_to_index(move)

            board.push(move)

    # Determine game result
    result_str = board.result()
    if result_str == "1-0":
        white_outcome = 1.0
    elif result_str == "0-1":
        white_outcome = -1.0
    else:
        white_outcome = 0.0

    print(f"Self-play game finished with result: {result_str}")

    # Build training data, flipping the final value for the side
    training_data = []
    for (enc, policy_vec, side) in game_history:
        value_label = white_outcome if side == chess.WHITE else -white_outcome
        training_data.append((enc, policy_vec, value_label))

    return training_data


# -------------------------------------------------------------------------
# 5) vs. Stockfish - produce a full policy distribution for the model's moves
# -------------------------------------------------------------------------
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
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    for param, value in stockfish_params.items():
        engine.configure({param: value})

    use_opening = (opening_fens is not None) and (random.random() < opening_probability)
    board = chess.Board(random.choice(opening_fens)) if use_opening else chess.Board()

    # Randomly decide which side is the model
    model_white = bool(random.getrandbits(1))
    print(f"Model plays {'White' if model_white else 'Black'} vs. Stockfish.")

    game_history = []  # store (enc, policy_vec) only for the model's moves
    side_list = []

    while not board.is_game_over() and len(board.move_stack) < 600:
        if (board.turn == chess.WHITE and model_white) or (board.turn == chess.BLACK and not model_white):
            # Model's turn
            enc = encode_board(board)
            side_list.append(board.turn)

            if num_simulations == 1:
                # Quick heuristic → 1-hot distribution
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
                if best_move is None:
                    best_move = random.choice(legal_moves)

                move_label = move_to_index(best_move)
                if not (0 <= move_label < NUM_MOVES):
                    # fallback
                    possible_indices = [
                        move_to_index(mv)
                        for mv in legal_moves
                        if 0 <= move_to_index(mv) < NUM_MOVES
                    ]
                    move_label = random.choice(possible_indices) if possible_indices else 0

                policy_vec = np.zeros((NUM_MOVES,), dtype=np.float32)
                policy_vec[move_label] = 1.0

                game_history.append((enc, policy_vec))
                board.push(best_move)

            else:
                root = get_mcts_root(
                    board=board,
                    model=model,
                    simulations=num_simulations,
                    batch_size=512,
                    c_puct=1)
                # Dirichlet noise in early moves
                # if len(board.move_stack) < 10:
                #     inject_dirichlet_noise(root, alpha=alpha, eps=eps)
                #
                policy_vector = extract_policy_vector(root, num_moves=NUM_MOVES)
                # if len(board.move_stack) < 20:
                #     current_temperature = temperature
                # else:
                #     current_temperature = 0.0
                # policy_vector = apply_temperature(policy_vector, current_temperature)

                # Store the policy distribution
                game_history.append((enc, policy_vector))

                move_label = np.random.choice(np.arange(NUM_MOVES), p=policy_vector)
                move = index_to_move(move_label, board)
                if (move not in board.legal_moves) or (move is None):
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        board.is_game_over(claim_draw=True)
                        print('No legal moves for player')
                        break
                    move = random.choice(legal_moves)
                board.push(move)
        else:
            # Stockfish's turn
            stockfish_move = engine.play(board, chess.engine.Limit(depth=1)).move
            board.push(stockfish_move)

    engine.quit()

    result_str = board.result()
    if result_str == "1-0":
        outcome = 1.0 if model_white else -1.0
    elif result_str == "0-1":
        outcome = -1.0 if model_white else 1.0
    else:
        outcome = 0.0

    print(f"Stockfish game finished with result: {result_str} "
          f"(model {'won' if outcome == 1 else 'did not win'})")

    # Build training data for the model's moves only
    training_data = []
    for i, (enc, policy_vec) in enumerate(game_history):
        side = side_list[i]
        value_label = outcome if side == chess.WHITE else -outcome
        training_data.append((enc, policy_vec, value_label))

    return training_data


# -------------------------------------------------------------------------
# 6) Model vs. second model - store full policy distribution for main_model
# -------------------------------------------------------------------------
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
    use_opening = (opening_fens is not None) and (random.random() < opening_probability)
    board = chess.Board(random.choice(opening_fens)) if use_opening else chess.Board()

    main_model_white = bool(random.getrandbits(1))
    print(f"Main model plays {'White' if main_model_white else 'Black'} vs. second model.")

    game_history = []  # store (enc, policy_vec) only for main_model's moves
    side_list = []

    while not board.is_game_over() and len(board.move_stack) < 600:
        if (board.turn == chess.WHITE and main_model_white) or (board.turn == chess.BLACK and not main_model_white):
            # Main model's turn
            enc = encode_board(board)
            side_list.append(board.turn)

            root = get_mcts_root(
                board=board,
                model=main_model,
                simulations=num_simulations,
                batch_size=512,
                c_puct=1
            )
            if len(board.move_stack) < 10:
                inject_dirichlet_noise(root, alpha=alpha, eps=eps)

            policy_vector = extract_policy_vector(root, num_moves=NUM_MOVES)
            if len(board.move_stack) < 20:
                current_temperature = temperature
            else:
                current_temperature = 0.0
            policy_vector = apply_temperature(policy_vector, current_temperature)

            game_history.append((enc, policy_vector))

            move_label = np.random.choice(np.arange(NUM_MOVES), p=policy_vector)
            move = index_to_move(move_label, board)
            if (move not in board.legal_moves) or (move is None):
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                else:
                    break
            board.push(move)
        else:
            # Second model's turn (we don't store second model's policy distribution,
            # but we do need a move to continue the game).
            root = get_mcts_root(
                board=board,
                model=second_model,
                simulations=1,
                batch_size=1024,
                c_puct=1
            )
            policy_vector = extract_policy_vector(root, num_moves=NUM_MOVES)
            if len(board.move_stack) < 20:
                current_temperature = temperature
            else:
                current_temperature = 0.0
            policy_vector = apply_temperature(policy_vector, current_temperature)

            move_label = np.random.choice(np.arange(NUM_MOVES), p=policy_vector)
            move = index_to_move(move_label, board)
            if (move not in board.legal_moves) or (move is None):
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                else:
                    break
            board.push(move)

    result_str = board.result()
    if result_str == "1-0":
        outcome = 1.0 if main_model_white else -1.0
    elif result_str == "0-1":
        outcome = -1.0 if main_model_white else 1.0
    else:
        outcome = 0.0

    print(f"Model vs. second model game finished with result: {result_str}")

    # Build training data for the main model's moves only
    training_data = []
    for i, (enc, policy_vec) in enumerate(game_history):
        side = side_list[i]
        value_label = outcome if side == chess.WHITE else -outcome
        training_data.append((enc, policy_vec, value_label))

    return training_data


# -------------------------------------------------------------------------
# 7) Main self-play loop
# -------------------------------------------------------------------------
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
    second_model_ratio=0.2
):
    if stockfish_params is None:
        stockfish_params = {"Skill Level": 5}

    # Load or create main model
    if model_path and os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        main_model = tf.keras.models.load_model(model_path, compile=False)
        # Re-compile with "categorical_crossentropy" for policy
        main_model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                "policy_head": "categorical_crossentropy",
                "value_head": "mse"
            },
            loss_weights={"policy_head": 1.0, "value_head": 1.0},
            metrics={"policy_head": ["categorical_accuracy"], "value_head": ["mse"]}
        )
        main_model.summary()
    else:
        print("No existing model found")
        return


    # Optionally load second model
    second_model = None
    if second_model_path and os.path.isfile(second_model_path):
        print(f"Loading second model from {second_model_path}")
        second_model = tf.keras.models.load_model(second_model_path, compile=False)
        second_model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                "policy_head": "categorical_crossentropy",
                "value_head": "mse"
            },
            loss_weights={"policy_head": 1.0, "value_head": 1.0},
            metrics={"policy_head": ["categorical_accuracy"], "value_head": ["mse"]}
        )

    # Load or create replay buffer
    if os.path.isfile(REPLAY_BUFFER_FILE_STOCKFISH):
        print(f"Loading replay buffer from {REPLAY_BUFFER_FILE_STOCKFISH}")
        with open(REPLAY_BUFFER_FILE_STOCKFISH, "rb") as f:
            replay_buffer = pickle.load(f)
    else:
        print("No existing replay buffer found. Creating a new one.")
        replay_buffer = deque(maxlen=MAX_BUFFER_SIZE)

    print(f"Replay buffer size is currently {len(replay_buffer)}.")

    # Loop for multiple iterations
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
        iteration_data = []

        for g_idx in range(games_per_iteration):
            r = random.random()

            if (stockfish_path is not None) and (r < stockfish_ratio):
                # vs Stockfish
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
                # vs second model
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
                # Self-play
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

        # Add the new data to the replay buffer
        for item in iteration_data:
            replay_buffer.append(item)
        print(f"Replay buffer size after iteration: {len(replay_buffer)}")

        # Sample from replay buffer
        if len(replay_buffer) < 150_000:
            sample_data = list(replay_buffer)
            print(f"Using all {len(sample_data)} samples (buffer still small).")
        else:
            sample_data = random.sample(replay_buffer, 150_000)
            print(f"Sampling 150,000 positions from replay buffer of size {len(replay_buffer)}.")

        # Train
        print(f"Training on {len(sample_data)} samples...")
        train_model(main_model, sample_data, batch_size=64, epochs=4)

        # Save
        if model_path:
            main_model.save(model_path)
            print(f"Main model saved to {model_path}")

        with open(REPLAY_BUFFER_FILE_STOCKFISH, "wb") as f:
            pickle.dump(replay_buffer, f)
        print(f"Replay buffer saved to {REPLAY_BUFFER_FILE_STOCKFISH}")

    print("All iterations complete!")


if __name__ == "__main__":
    main_self_play_loop(
        num_iterations=3,
        games_per_iteration=100,
        num_simulations=5,
        model_path="/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/train_supervised/alphazero_full_policy_model_kopia.keras",
        learning_rate=1e-4,
        stockfish_ratio=0.3,
        stockfish_path="/Users/mateuszzieba/Desktop/dev/chess/stockfish/src/stockfish",
        stockfish_params={"Skill Level": 0},
        second_model_path=None,
        second_model_ratio=0.0
    )