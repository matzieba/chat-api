import os
import random
import pickle
from collections import deque
import concurrent.futures
import time

import chess
import chess.engine
import numpy as np
import tensorflow as tf

from chess_api.mcts import run_mcts_batched, encode_node_4frames, MCTSNode
from chess_engine.engine.train_supervised.parse_pgn_alpha0 import (
    encode_single_board,
    move_to_index
)

###############################################################################
# 0) Constants & Utilities
###############################################################################
NUM_MOVES = 64 * 64 * 5  # from_sq * 64 * 5 + to_sq * 5 + promo_code
MAX_BUFFER_SIZE = 90000
REPLAY_BUFFER_FILE = "replay_buffer.pkl"

###############################################################################
# 1) Popular Openings (in SAN)
#    We'll pick a random one for the model to *force* in the early moves
###############################################################################
POPULAR_OPENINGS = [
    "e4 e5 Nf3 Nc6 Bb5",
    "e4 e5 Nf3 d6 d4 exd4 Nxd4",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6",
    "d4 d5 c4 e6 Nc3 Nf6",
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6",
    "e4 e6 d4 d5 Nc3 Bb4",
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4",
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6",
    "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6",
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6",
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6",
    "e4 g6 d4 Bg7 Nc3 d6 f4 Nf6",
    "c4 e6 Nc3 d5 d4 Nf6",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 d6",
    "d4 d5 Bf4 Nf6 e3 e6 Nf3 c5 c3"
]


def select_forced_opening(prob=0.6):
    """
    With some probability, choose a random opening line (SAN strings).
    Returns a list of SAN moves or None if we skip forcing any opening.
    """
    if random.random() < prob:
        opening_line = random.choice(POPULAR_OPENINGS)
        return opening_line.split()  # list of SAN moves
    return None


###############################################################################
# Global model pointer used by each worker
###############################################################################
MODEL = None  # Will be set in init_worker


def init_worker(model_path):
    """
    Called once in each worker process to load the model into a global variable.
    Avoids reloading the model for each game; ensures distinct RNG seeds.
    """
    worker_seed = os.getpid() ^ int(time.time())
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    tf.random.set_seed(worker_seed)

    global MODEL
    if MODEL is None:
        print(f"Worker {os.getpid()} is loading model from: {model_path}")
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
        # IMPORTANT: Use sparse CE, since we store single integer labels:
        loaded_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=["sparse_categorical_crossentropy", "mean_squared_error"]
        )
        MODEL = loaded_model
    else:
        print(f"Worker {os.getpid()} already has a loaded model.")


###############################################################################
# 1) Self-play & Stockfish Game Logic with forced opening moves
###############################################################################
def self_play_game_batched(
    model,
    n_mcts_sims=400,
    mcts_batch_size=200,
    opening_prob=0.5,
    add_dirichlet_noise=True,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25
):
    """
    Self-play with optional forced opening moves for both sides.
    We now store only the argmax move index as the policy label.
    """
    board = chess.Board()
    forced_line = select_forced_opening(prob=opening_prob)
    forced_index = 0

    positions = []
    while not board.is_game_over() and len(board.move_stack) < 600:
        # If we still have forced moves left:
        if forced_line and forced_index < len(forced_line):
            next_san = forced_line[forced_index]
            try:
                forced_move = board.parse_san(next_san)
            except ValueError:
                forced_move = None

            if forced_move and forced_move in board.legal_moves:
                enc = encode_node_4frames(MCTSNode(board=board), max_frames=4)
                # Single-integer label:
                forced_move_idx = move_to_index(forced_move)

                side_is_white = board.turn
                positions.append((enc, forced_move_idx, side_is_white))

                board.push(forced_move)
                forced_index += 1
                continue
            else:
                forced_line = None

        # If no forced move => run MCTS normally
        node = MCTSNode(board=board)
        enc = encode_node_4frames(node, max_frames=4)
        best_move, _policy_dist = run_mcts_batched(
            model=model,
            root_board=board,
            n_simulations=n_mcts_sims,
            batch_size=mcts_batch_size,
            temperature=0.4,
            add_dirichlet_noise=add_dirichlet_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon
        )
        if best_move is None:
            break

        best_move_idx = move_to_index(best_move)
        side_to_move_is_white = board.turn
        positions.append((enc, best_move_idx, side_to_move_is_white))

        board.push(best_move)

    # Final outcome from White's perspective
    result_str = board.result()
    if result_str == "1-0":
        outcome_for_white = 1.0
    elif result_str == "0-1":
        outcome_for_white = -1.0
    else:
        outcome_for_white = 0.0

    print(f"self-play game result: {result_str}")

    # Build training data
    training_data = []
    for (enc, move_idx, side_is_white) in positions:
        val = outcome_for_white if side_is_white else -outcome_for_white
        training_data.append((enc, move_idx, val))

    return training_data


def play_game_vs_stockfish_batched(
    model,
    stockfish_path,
    stockfish_params=None,
    n_mcts_sims=400,
    mcts_batch_size=200,
    opening_prob=0.5,
    add_dirichlet_noise=True,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25
):
    """
    Model vs. Stockfish, forced opening moves only for the MODEL side.
    Single-integer labeling: each position is tagged with the best_move_idx.
    """
    if stockfish_params is None:
        stockfish_params = {"Skill Level": 15}

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure(stockfish_params)

    board = chess.Board()
    model_is_white = bool(random.getrandbits(1))

    forced_line = select_forced_opening(prob=opening_prob)
    forced_index = 0

    model_positions = []
    sf_positions = []

    while not board.is_game_over() and len(board.move_stack) < 600:
        if (board.turn == chess.WHITE and model_is_white) or \
           (board.turn == chess.BLACK and not model_is_white):

            # Possibly forced move
            forced_move = None
            if forced_line and forced_index < len(forced_line):
                san = forced_line[forced_index]
                try:
                    parsed = board.parse_san(san)
                    if parsed in board.legal_moves:
                        forced_move = parsed
                    else:
                        forced_line = None
                except ValueError:
                    forced_line = None

            if forced_move:
                enc = encode_node_4frames(MCTSNode(board=board), max_frames=4)
                forced_move_idx = move_to_index(forced_move)

                side_white = board.turn
                model_positions.append((enc, forced_move_idx, side_white))

                board.push(forced_move)
                forced_index += 1

            else:
                # MCTS
                node = MCTSNode(board=board)
                enc = encode_node_4frames(node, max_frames=4)
                best_move, _policy_dist = run_mcts_batched(
                    model=model,
                    root_board=board,
                    n_simulations=n_mcts_sims,
                    batch_size=mcts_batch_size,
                    temperature=0.2,
                    add_dirichlet_noise=add_dirichlet_noise,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_epsilon=dirichlet_epsilon
                )
                if best_move is None:
                    break

                best_move_idx = move_to_index(best_move)
                side_white = board.turn
                model_positions.append((enc, best_move_idx, side_white))
                board.push(best_move)

        else:
            # Stockfish's turn
            sf_node = MCTSNode(board=board)
            sf_enc = encode_node_4frames(sf_node, max_frames=4)

            sf_move = engine.play(board, chess.engine.Limit(depth=1)).move
            if sf_move is None or not board.is_legal(sf_move):
                print("Stockfish returned an illegal move or None.")
                break

            sf_move_idx = move_to_index(sf_move)
            side_white = board.turn
            sf_positions.append((sf_enc, sf_move_idx, side_white))

            board.push(sf_move)

    engine.quit()

    # Final outcome
    result_str = board.result()
    if result_str == "1-0":
        white_score = 1.0
    elif result_str == "0-1":
        white_score = -1.0
    else:
        white_score = 0.0

    if model_is_white:
        outcome_for_model = white_score
    else:
        outcome_for_model = -white_score

    if outcome_for_model > 0:
        print("Model WON against Stockfish.")
    elif outcome_for_model < 0:
        print("Model LOST against Stockfish.")
    else:
        print("Model DREW against Stockfish.")

    # Combine positions
    training_data = []
    # Model moves (label from the model's perspective):
    for (enc, move_idx, side_is_white) in model_positions:
        val = outcome_for_model if (side_is_white == model_is_white) else -outcome_for_model
        training_data.append((enc, move_idx, val))

    # Stockfish moves (label from White's perspective):
    for (enc, move_idx, side_is_white) in sf_positions:
        val_sf = white_score if side_is_white == chess.WHITE else -white_score
        training_data.append((enc, move_idx, val_sf))

    return training_data


###############################################################################
# 2) Single-game worker function
###############################################################################
def run_one_game_in_worker(
    game_index,
    stockfish_games_ratio,
    stockfish_path,
    stockfish_params,
    n_mcts_sims,
    mcts_batch_size
):
    global MODEL
    if MODEL is None:
        raise RuntimeError("Global MODEL was not initialized in the worker!")

    r = random.random()
    if stockfish_path and r < stockfish_games_ratio:
        # Game vs Stockfish
        training_data = play_game_vs_stockfish_batched(
            model=MODEL,
            stockfish_path=stockfish_path,
            stockfish_params=stockfish_params,
            n_mcts_sims=n_mcts_sims,
            mcts_batch_size=mcts_batch_size,
            opening_prob=0,
            add_dirichlet_noise=False,
            dirichlet_alpha=0.1,
            dirichlet_epsilon=0.1
        )
        game_type = "Stockfish"
    else:
        # Self-Play
        training_data = self_play_game_batched(
            model=MODEL,
            n_mcts_sims=n_mcts_sims,
            mcts_batch_size=mcts_batch_size,
            opening_prob=0.7,
            add_dirichlet_noise=True,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.3
        )
        game_type = "Self-Play"

    print(f"Game {game_index} completed: {game_type}, got {len(training_data)} samples.")
    return training_data


###############################################################################
# 3) Training the Model
###############################################################################
def train_model(model, training_data, batch_size=64, epochs=3):
    """
    Training data is: (board_enc, single_move_idx, value).
    We'll store single_move_idx in y_pol, so we use sparse_categorical_crossentropy.
    """
    boards = []
    policy_moves = []
    value_labels = []

    for (enc, move_idx, val) in training_data:
        boards.append(enc)
        policy_moves.append(move_idx)
        value_labels.append(val)

    x = np.array(boards, dtype=np.float32)           # shape (N, 64, 14*frames) or similar
    y_pol = np.array(policy_moves, dtype=np.int32)   # shape (N,); single integer label
    y_val = np.array(value_labels, dtype=np.float32).reshape(-1, 1)  # shape (N,1)

    hist = model.fit(
        x=x,
        y=[y_pol, y_val],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    return hist


###############################################################################
# 4) Main Loop - Parallel Version
###############################################################################
def main_self_play_loop_parallel(
    num_iterations,
    games_per_iteration,
    model_path,
    replay_buffer_file=REPLAY_BUFFER_FILE,
    stockfish_games_ratio=0.1,
    stockfish_path=None,
    stockfish_params=None,
    n_mcts_sims=400,
    mcts_batch_size=200,
    num_workers=6
):
    """
    High-level repeated procedure (parallelized):
      1) Load model once in main process (for logging, summary).
      2) Load or create replay buffer.
      3) For each iteration:
          a) Use ProcessPoolExecutor to run games in parallel (init_worker).
          b) Gather all training data, add to replay buffer.
          c) Sample from replay buffer & train the main model.
          d) Save model & replay buffer.
    """
    # 1) Load model for logging in main process
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No model found at: {model_path}")

    print(f"Loading initial model from: {model_path}")
    init_model = tf.keras.models.load_model(model_path, compile=False)
    # Use sparse CE here as well:
    init_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=["sparse_categorical_crossentropy", "mean_squared_error"]
    )
    init_model.summary()

    # 2) Load or create replay buffer
    if os.path.isfile(replay_buffer_file):
        with open(replay_buffer_file, "rb") as f:
            replay_buffer = pickle.load(f)
        print(f"Loaded replay buffer from {replay_buffer_file}, size={len(replay_buffer)}")
    else:
        replay_buffer = deque(maxlen=MAX_BUFFER_SIZE)
        print(f"No existing replay buffer found. Created new one (max={MAX_BUFFER_SIZE}).")

    # 3) Main loop
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===\n")
        iteration_data = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(model_path,)
        ) as executor:
            futures = [
                executor.submit(
                    run_one_game_in_worker,
                    g_idx,
                    stockfish_games_ratio,
                    stockfish_path,
                    stockfish_params,
                    n_mcts_sims,
                    mcts_batch_size
                )
                for g_idx in range(games_per_iteration)
            ]
            for f in concurrent.futures.as_completed(futures):
                data = f.result()  # list of (enc, move_idx, val)
                iteration_data.extend(data)

        # Add new data to replay buffer
        for item in iteration_data:
            replay_buffer.append(item)
        print(f"Replay buffer size is now {len(replay_buffer)}")

        # Sample from replay buffer
        if len(replay_buffer) < 50000:
            sample_data = list(replay_buffer)
            print(f"Using all {len(sample_data)} samples (buffer < 50k).")
        else:
            sample_data = random.sample(replay_buffer, 50000)
            print(f"Sampled 50,000 positions from buffer of size {len(replay_buffer)}.")

        # Reload model & train
        print("Reloading model in main process for training.")
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=["sparse_categorical_crossentropy", "mean_squared_error"]
        )
        print(f"Training on {len(sample_data)} samples...")
        train_model(model, sample_data, batch_size=256, epochs=3)

        # Save model & replay buffer
        model.save(model_path)
        print(f"Model saved to {model_path}")

        with open(replay_buffer_file, "wb") as f:
            pickle.dump(replay_buffer, f)
        print(f"Replay buffer saved to {replay_buffer_file}")

    print("All iterations complete!")


###############################################################################
# 5) Example Entry Point
###############################################################################
if __name__ == "__main__":
    main_self_play_loop_parallel(
        num_iterations=10,
        games_per_iteration=50,
        model_path="/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/small_model/transformer/my_engine_eval_model_100GB_of_parsed_games_depht_8_copy.keras",
        replay_buffer_file="replay_buffer.pkl",
        stockfish_games_ratio=0.2,
        stockfish_path="/Users/mateuszzieba/Desktop/dev/chess/stockfish/src/stockfish",
        stockfish_params={"Skill Level": 15},
        n_mcts_sims=50,
        mcts_batch_size=25,
        num_workers=6
    )