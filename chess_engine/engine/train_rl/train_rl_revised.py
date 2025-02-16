import os
import pickle
import random
import chess
import chess.engine
import numpy as np
import tensorflow as tf

################################################################################
# Settings for replay buffer on disk
################################################################################
REPLAY_BUFFER_FILE = "replay_buffer.pkl"
MAX_REPLAY_BUFFER_SIZE = 100000  # Maximum number of positions to keep

################################################################################
# Replay buffer utilities
################################################################################
def load_replay_buffer(path: str):
    """Load the replay buffer from a pickle file, or return empty if not found."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_replay_buffer(path: str, data):
    """Save the replay buffer to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(data, f)

def trim_replay_buffer(data, max_size):
    """Keep only the most recent 'max_size' items from 'data'."""
    if len(data) > max_size:
        data = data[-max_size:]
    return data

################################################################################
# 1. Board encoding
################################################################################
def encode_board(board: chess.Board) -> np.ndarray:
    encoded = np.zeros((64, 14), dtype=np.float32)

    piece_type_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            base_channel = 0 if piece.color == chess.WHITE else 6
            channel_offset = piece_type_to_channel[piece.piece_type]
            encoded[sq, base_channel + channel_offset] = 1.0

    side_to_move_val = 1.0 if board.turn == chess.WHITE else 0.0
    encoded[:, 12] = side_to_move_val

    castling_val = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_val += 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_val += 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_val += 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_val += 8
    encoded[:, 13] = float(castling_val)

    return encoded

################################################################################
# 2. Move indexing
################################################################################
def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq   = move.to_square
    base    = from_sq * 64 + to_sq
    # +4096 if promotion to a queen
    if move.promotion == chess.QUEEN:
        return 4096 + base
    else:
        return base

NUM_MOVES = 8192  # total moves in our representation

################################################################################
# 3. ChessModelWrapper
################################################################################
class ChessModelWrapper:
    """Wraps a tf.keras.Model with [policy_head, value_head] outputs."""
    def __init__(self, model_path: str):
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.model.summary()
        else:
            raise ValueError(f"Model path not found: {model_path}")

    def predict(self, board_batch: np.ndarray):
        """
        board_batch: shape [batch_size, 64, 14]
        Returns (policy_logits_batch, value_batch):
          - policy_logits_batch: [batch_size, 8192]
          - value_batch:         [batch_size]
        """
        policy_logits, value = self.model.predict(board_batch, verbose=0)
        return policy_logits, value[:, 0]

################################################################################
# 4. MCTS Node
################################################################################
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, prior=0.0):
        self.board       = board
        self.parent      = parent
        self.children    = {}  # move -> MCTSNode
        self.visit_count = 0
        self.value_sum   = 0.0
        self.prior       = prior
        self.is_expanded = False

    def expanded(self) -> bool:
        return self.is_expanded

    @property
    def q_value(self) -> float:
        return self.value_sum / (1 + self.visit_count)

    @property
    def u_value(self) -> float:
        if self.parent is None:
            return 0
        c_puct = 1.5
        return (c_puct * self.prior *
                np.sqrt(self.parent.visit_count + 1) /
                (1 + self.visit_count))

    def best_child(self):
        """Select the move->child with highest (Q + U)."""
        best_score = -float('inf')
        best_move  = None
        best_node  = None
        for move, child in self.children.items():
            score = child.q_value + child.u_value
            if score > best_score:
                best_score = score
                best_move  = move
                best_node  = child
        return best_move, best_node

    def expand(self, policy_probs: np.ndarray):
        """
        policy_probs: shape [8192], distribution over move indices
        """
        self.is_expanded = True
        for move in self.board.legal_moves:
            idx   = move_to_index(move)
            prior = policy_probs[idx]
            next_board = self.board.copy()
            next_board.push(move)
            self.children[move] = MCTSNode(next_board, parent=self, prior=prior)

    def backpropagate(self, value_estimate: float):
        """Propagate value up the tree, flipping sign at each parent step."""
        self.visit_count += 1
        self.value_sum   += value_estimate
        if self.parent is not None:
            self.parent.backpropagate(-value_estimate)

################################################################################
# 5. MCTS Search with batched expansions
################################################################################
def mcts_search(root: MCTSNode,
                model_wrapper: ChessModelWrapper,
                num_simulations: int = 100,
                batch_size: int = 16):
    """
    Perform MCTS from the given root node, but do NN expansions in batches.
    Returns (mcts_pi, root_q) where:
     - mcts_pi is shape [8192], the final policy distribution from root.
     - root_q  is the root Q-value estimate after the search.
    """
    # If root is already terminal, no moves to search.
    if root.board.is_game_over():
        mcts_pi = np.zeros((NUM_MOVES,), dtype=np.float32)
        outcome = root.board.outcome()
        if outcome is None:
            root_value = 0.0
        else:
            if outcome.winner is None:
                root_value = 0.0  # draw
            else:
                root_value = 1.0 if outcome.winner == root.board.turn else -1.0
        return mcts_pi, root_value

    expand_queue = []

    def run_expansion_batch():
        """Process all queued leaf nodes with one NN inference."""
        if not expand_queue:
            return
        boards = np.stack([item[1] for item in expand_queue], axis=0)
        policy_logits_batch, value_batch = model_wrapper.predict(boards)

        for i, (node, _) in enumerate(expand_queue):
            policy_logits = policy_logits_batch[i]
            value         = value_batch[i]
            policy_probs  = tf.nn.softmax(policy_logits).numpy()
            node.expand(policy_probs)
            node.backpropagate(value)

        expand_queue.clear()

    for sim in range(num_simulations):
        node = root
        # (a) Selection
        while node.expanded() and len(node.children) > 0:
            best_move, node = node.best_child()
            if node.board.is_game_over():
                break

        # (b) Queue for expansion if leaf & not terminal
        if not node.expanded() and not node.board.is_game_over():
            board_enc = encode_board(node.board)
            expand_queue.append((node, board_enc))

        # (c) Possibly run expansions
        if len(expand_queue) >= batch_size or sim == (num_simulations - 1):
            run_expansion_batch()
        else:
            if node.board.is_game_over():
                outcome = node.board.outcome()
                if outcome is not None:
                    if outcome.winner is None:
                        node.backpropagate(0.0)  # draw
                    else:
                        value = 1.0 if outcome.winner == node.board.turn else -1.0
                        node.backpropagate(value)
                else:
                    node.backpropagate(0.0)

    # Final flush of any leftover expansions
    run_expansion_batch()

    # Build final distribution
    child_moves = list(root.children.keys())
    if len(child_moves) == 0:
        # Terminal or no expansions occurred, fallback to zeros
        mcts_pi = np.zeros((NUM_MOVES,), dtype=np.float32)
    else:
        visits = np.array([child.visit_count for child in root.children.values()],
                          dtype=np.float32)
        total_visits = visits.sum()
        if total_visits > 1e-12:
            visits /= total_visits
        else:
            # fallback: uniform
            visits = np.ones_like(visits) / len(visits)
        mcts_pi = np.zeros((NUM_MOVES,), dtype=np.float32)
        for i, m in enumerate(child_moves):
            mcts_pi[move_to_index(m)] = visits[i]

    return mcts_pi, root.q_value

################################################################################
# 6. Self-play
################################################################################
def self_play_one_game(
    model_wrapper: ChessModelWrapper,
    num_mcts_sims: int = 100,
    mcts_batch_size: int = 16
):
    board = chess.Board()
    game_history = []

    while not board.is_game_over():
        root = MCTSNode(board)
        mcts_pi, root_value = mcts_search(
            root,
            model_wrapper,
            num_simulations=num_mcts_sims,
            batch_size=mcts_batch_size
        )

        # Safety check: if pi sums to ~0, fallback to uniform over legal moves
        sum_pi = mcts_pi.sum()
        if sum_pi < 1e-12:
            legal_moves = list(board.legal_moves)
            if len(legal_moves) == 0:
                # No moves => game over, but let's forcibly break
                break
            else:
                uniform_pi = np.zeros_like(mcts_pi, dtype=np.float32)
                for move in legal_moves:
                    idx = move_to_index(move)
                    uniform_pi[idx] = 1.0
                uniform_pi /= uniform_pi.sum()
                mcts_pi = uniform_pi

        board_enc = encode_board(board)
        game_history.append([board_enc, mcts_pi, None])

        move_idx = np.random.choice(np.arange(NUM_MOVES), p=mcts_pi)
        from_sq  = move_idx // 64
        to_sq    = move_idx % 64
        promo    = None
        if move_idx >= 4096:
            base = move_idx - 4096
            from_sq = base // 64
            to_sq   = base % 64
            promo   = chess.QUEEN

        move = chess.Move(from_sq, to_sq, promotion=promo)
        if move not in board.legal_moves:
            # fallback if something invalid
            legal_moves = list(board.legal_moves)
            if len(legal_moves) == 0:
                break
            move = np.random.choice(legal_moves)
        board.push(move)

    outcome = board.outcome()
    if outcome is not None:
        if outcome.winner is None:
            game_result = 0.0  # draw
        else:
            game_result = 1.0 if outcome.winner == chess.WHITE else -1.0
    else:
        game_result = 0.0

    # Fill final result in game_history
    current_color = chess.WHITE
    for record in game_history:
        if current_color == chess.WHITE:
            record[2] = game_result
        else:
            record[2] = -game_result
        current_color = not current_color

    # Debug prints
    if game_result == 0:
        print("Self-play game ended in a draw.")
    elif game_result > 0:
        print("Self-play game ended: White (model) won.")
    else:
        print("Self-play game ended: Black (model) won.")

    return game_history

################################################################################
# 7. Play vs. Stockfish
################################################################################
def play_against_stockfish(
    model_wrapper: ChessModelWrapper,
    stockfish_path: str,
    num_games: int = 5,
    num_mcts_sims: int = 50,
    mcts_batch_size: int = 16,
    stockfish_skill: int = 20,
    stockfish_depth: int = 15
):

    if not os.path.exists(stockfish_path):
        raise ValueError(f"Stockfish not found: {stockfish_path}")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": stockfish_skill})

    all_data = []
    for game_idx in range(num_games):
        board = chess.Board()
        our_color = chess.WHITE if (game_idx % 2 == 0) else chess.BLACK
        game_history = []

        while not board.is_game_over():
            if board.turn == our_color:
                # MCTS move
                root = MCTSNode(board)
                mcts_pi, root_value = mcts_search(
                    root,
                    model_wrapper,
                    num_simulations=num_mcts_sims,
                    batch_size=mcts_batch_size
                )
                sum_pi = mcts_pi.sum()
                if sum_pi < 1e-12:
                    legal_moves = list(board.legal_moves)
                    if len(legal_moves) == 0:
                        break
                    else:
                        # fallback uniform
                        uniform_pi = np.zeros_like(mcts_pi, dtype=np.float32)
                        for mv in legal_moves:
                            uniform_pi[move_to_index(mv)] = 1.0
                        uniform_pi /= uniform_pi.sum()
                        mcts_pi = uniform_pi

                board_enc = encode_board(board)
                game_history.append([board_enc, mcts_pi, None])

                move_idx = np.random.choice(np.arange(NUM_MOVES), p=mcts_pi)
                from_sq  = move_idx // 64
                to_sq    = move_idx % 64
                promo    = None
                if move_idx >= 4096:
                    real_base = move_idx - 4096
                    from_sq   = real_base // 64
                    to_sq     = real_base % 64
                    promo     = chess.QUEEN
                move = chess.Move(from_sq, to_sq, promotion=promo)
                if move not in board.legal_moves:
                    legal_moves = list(board.legal_moves)
                    if len(legal_moves) == 0:
                        break
                    move = np.random.choice(legal_moves)
                board.push(move)
            else:
                # Stockfish move
                sf_result = engine.play(board, chess.engine.Limit(depth=stockfish_depth))
                board.push(sf_result.move)

        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                game_result = 0.0
            else:
                game_result = 1.0 if outcome.winner == chess.WHITE else -1.0
        else:
            game_result = 0.0

        final_result = game_result if our_color == chess.WHITE else -game_result
        for record in game_history:
            record[2] = final_result
        all_data.extend(game_history)

        color_str = "White" if our_color == chess.WHITE else "Black"
        if final_result == 0:
            print(f"Stockfish game {game_idx+1} ended in a draw (model as {color_str}).")
        elif final_result > 0:
            print(f"Stockfish game {game_idx+1} ended: model (as {color_str}) won.")
        else:
            print(f"Stockfish game {game_idx+1} ended: model (as {color_str}) lost.")

    engine.quit()
    return all_data

################################################################################
# 8. Training
################################################################################
def train_on_data(model, data, batch_size=32, epochs=1):
    """
    data: list of [board_enc, policy_vec, outcome_float].
    """
    boards = np.array([d[0] for d in data], dtype=np.float32)   # [N, 64, 14]
    pis    = np.array([d[1] for d in data], dtype=np.float32)   # [N, 8192]
    zs     = np.array([d[2] for d in data], dtype=np.float32)   # [N]
    model.fit(
        boards,
        [pis, zs],
        batch_size=batch_size,
        epochs=epochs,
        verbose=0
    )

################################################################################
# 9. Main RL loop
################################################################################
if __name__ == "__main__":
    MODEL_PATH       = "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/train_supervised/alphazero_full_policy_model_kopia.keras"
    STOCKFISH_PATH   = "/Users/mateuszzieba/Desktop/dev/chess/stockfish/src/stockfish"
    ITERATIONS       = 3
    NUM_GAMES_TOTAL  = 150
    RATIO_VS_SF      = 0.6
    NUM_MCTS_SIMS    = 4
    MCTS_BATCH_SIZE  = 2
    STOCKFISH_SKILL  = 0
    STOCKFISH_DEPTH  = 1
    EPOCHS_PER_ITER  = 2

    # 1) Load model
    model_wrapper = ChessModelWrapper(MODEL_PATH)
    model = model_wrapper.model

    # 2) Re-compile (fine-tuning)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=["categorical_crossentropy", "mean_squared_error"]
    )

    # 3) Load replay buffer
    replay_buffer = load_replay_buffer(REPLAY_BUFFER_FILE)
    print(f"Loaded replay buffer with {len(replay_buffer)} samples.")

    # 4) RL loop
    for iteration in range(ITERATIONS):
        print(f"\n--- Iteration {iteration+1}/{ITERATIONS} ---")
        # Decide how many self-play vs. stockfish games
        num_vs_sf = int(NUM_GAMES_TOTAL * RATIO_VS_SF)
        num_vs_sf = max(num_vs_sf, 0)
        num_self  = NUM_GAMES_TOTAL - num_vs_sf

        all_data = []

        # 4a) Self-play
        if num_self > 0:
            print(f" Generating {num_self} self-play games...")
            for _ in range(num_self):
                game_data = self_play_one_game(
                    model_wrapper,
                    num_mcts_sims=NUM_MCTS_SIMS,
                    mcts_batch_size=MCTS_BATCH_SIZE
                )
                all_data.extend(game_data)

        # 4b) vs. Stockfish
        if num_vs_sf > 0:
            print(f" Generating {num_vs_sf} games vs. Stockfish (skill={STOCKFISH_SKILL}, depth={STOCKFISH_DEPTH})...")
            sf_data = play_against_stockfish(
                model_wrapper,
                STOCKFISH_PATH,
                num_games=num_vs_sf,
                num_mcts_sims=NUM_MCTS_SIMS,
                mcts_batch_size=MCTS_BATCH_SIZE,
                stockfish_skill=STOCKFISH_SKILL,
                stockfish_depth=STOCKFISH_DEPTH
            )
            all_data.extend(sf_data)

        # 4c) Update replay buffer
        if all_data:
            print(f" Acquired {len(all_data)} new samples.")
            replay_buffer.extend(all_data)
            replay_buffer = trim_replay_buffer(replay_buffer, MAX_REPLAY_BUFFER_SIZE)
            save_replay_buffer(REPLAY_BUFFER_FILE, replay_buffer)
            print(f" Replay buffer now has {len(replay_buffer)} samples.")

            # Train on entire replay buffer
            print(f" Training on {len(replay_buffer)} positions ...")
            train_on_data(model, replay_buffer, batch_size=64, epochs=EPOCHS_PER_ITER)
        else:
            print(" No new data was generated this iteration.")

        # 4d) Save checkpoint
        chkpt_name = f"mcts_checkpoint_iter_{iteration+1}.keras"
        model.save(chkpt_name)
        print(f" Saved checkpoint: {chkpt_name}")
    model.save(MODEL_PATH)
    print("RL + Stockfish training loop complete!")