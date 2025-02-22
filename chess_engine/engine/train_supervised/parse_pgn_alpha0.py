import os
import random
import chess
import chess.pgn
import chess.engine
import numpy as np
import tensorflow as tf

# If these are in another file, import them:
# from environment import encode_board, move_to_index

###############################################################################
# Utility / Hyperparameters
###############################################################################
NUM_MOVES = 64 * 64 * 5  # from your scheme: from_sq * 64 * 5 + to_sq * 5 + promo_code
MAX_TOTAL_BYTES = 100 * 1024**3  # 100 GB, adjust as you prefer

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Example placeholder. Replace with your real board encoder that returns
    a shape (64, 14) or similar.
    Here we do a trivial encoding that won't be meaningful for real training.
    """
    # Just a dummy encoding: one-hot of piece type, ignoring color/side to move, etc.
    # In real code, you have your existing encode_board logic.
    enc = np.zeros((64, 14), dtype=np.float32)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            piece_idx = {
                chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
            }.get(piece.piece_type, -1)
            color_offset = 0 if piece.color == chess.WHITE else 6
            if piece_idx >= 0:
                enc[sq, piece_idx + color_offset] = 1.0

    # For demonstration, let's put side-to-move in last plane
    if board.turn == chess.WHITE:
        enc[:, -1] = 1.0
    else:
        enc[:, -1] = 0.0
    return enc


def move_to_index(move: chess.Move) -> int:
    """
    Convert chess.Move to an integer in [0, 20480).
    from_sq * 64 * 5 + to_sq * 5 + promo_code
    """
    promo_code = 0
    if move.promotion:  # If it's a promotion
        if move.promotion == chess.KNIGHT:
            promo_code = 1
        elif move.promotion == chess.BISHOP:
            promo_code = 2
        elif move.promotion == chess.ROOK:
            promo_code = 3
        elif move.promotion == chess.QUEEN:
            promo_code = 4

    return move.from_square * 64 * 5 + move.to_square * 5 + promo_code


def convert_cp_to_value(score_cp: float, clamp=1000.0) -> float:
    """
    Convert centipawn score in [-∞, +∞] to [-1.0..+1.0], saturating at ±clamp.
    E.g. clamp=1000 => if engine says +1200 cp, we treat it as +1.0
    """
    if score_cp > clamp:
        return 1.0
    elif score_cp < -clamp:
        return -1.0
    else:
        return score_cp / clamp


###############################################################################
# TFRecord Example creation
###############################################################################
def create_example(board_array, move_idx, value):
    """
    Creates a TFRecord Example from (board, move_idx, value).
    """
    board_bytes = board_array.tobytes()
    features = {
        "board": tf.train.Feature(bytes_list=tf.train.BytesList(value=[board_bytes])),
        "move": tf.train.Feature(int64_list=tf.train.Int64List(value=[move_idx])),
        "value": tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


###############################################################################
# Shard writing logic
###############################################################################
def write_shard(
    buffer_data,
    shard_index,
    output_dir,
    val_split=0.2,
    total_bytes_written=0,
    max_total_bytes=100 * 1024**3
):
    """
    Takes buffer_data, shuffles it, splits into train/val TFRecord files,
    writes them out. Returns updated total_bytes_written plus flags.
    """
    random.shuffle(buffer_data)
    val_count = int(len(buffer_data) * val_split)
    val_chunk = buffer_data[:val_count]
    train_chunk = buffer_data[val_count:]

    wrote_anything = False
    stop_parsing = False

    # Shard filenames
    train_shard_path = os.path.join(output_dir, f"train-shard-{shard_index:03d}.tfrecord")
    val_shard_path   = os.path.join(output_dir, f"val-shard-{shard_index:03d}.tfrecord")

    # Write train shard
    with tf.io.TFRecordWriter(train_shard_path) as train_writer:
        for (board_array, move_idx, value_f) in train_chunk:
            ex = create_example(board_array, move_idx, value_f)
            serialized = ex.SerializeToString()
            if total_bytes_written + len(serialized) > max_total_bytes:
                stop_parsing = True
                break
            train_writer.write(serialized)
            total_bytes_written += len(serialized)
            wrote_anything = True

    # If we already hit the limit, skip val writing
    if not stop_parsing:
        with tf.io.TFRecordWriter(val_shard_path) as val_writer:
            for (board_array, move_idx, value_f) in val_chunk:
                ex = create_example(board_array, move_idx, value_f)
                serialized = ex.SerializeToString()
                if total_bytes_written + len(serialized) > max_total_bytes:
                    stop_parsing = True
                    break
                val_writer.write(serialized)
                total_bytes_written += len(serialized)
                wrote_anything = True

    print(f"Shard {shard_index:03d} -> train: {len(train_chunk)} (some partial if limit reached)")
    print(f"Shard {shard_index:03d} -> val:   {len(val_chunk)}   (some partial if limit reached)")
    print(f"Total bytes so far: {total_bytes_written} / {max_total_bytes} (limit)")

    return total_bytes_written, wrote_anything, stop_parsing


###############################################################################
# Main PGN parsing with engine evaluation
###############################################################################
def parse_pgn_files_sharded(
    pgn_paths,
    output_dir="data_tfrecords_sharded",
    shard_buffer_size=50000,
    val_split=0.2,
    limit=None,
    max_total_bytes=MAX_TOTAL_BYTES,
    engine_path="/Users/mateuszzieba/Desktop/dev/chess/stockfish/src/stockfish",
    engine_depth=5
):
    """
    Reads PGN files, uses engine to evaluate each position,
    then writes TFRecords in train/val shards.

    :param pgn_paths: List of .pgn file paths
    :param output_dir: Where to save TFRecords
    :param shard_buffer_size: # of samples to buffer before writing a shard
    :param val_split: Fraction for validation in each shard
    :param limit: Max # of games to parse (optional)
    :param max_total_bytes: Byte limit
    :param engine_path: Path to your UCI engine (e.g. Stockfish)
    :param engine_depth: Depth for engine.search
    """
    os.makedirs(output_dir, exist_ok=True)

    sample_buffer = []
    games_processed = 0
    shard_index = 0
    total_bytes_written = 0
    stop_parsing = False

    # 1) Launch engine
    print(f"Initializing engine from: {engine_path}")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine_limit = chess.engine.Limit(depth=engine_depth, time=0.05)

    def flush_buffer():
        """
        Writes out the current sample buffer to TFRecords.
        Clears buffer and updates shard index.
        Returns True if should continue, False if we hit limit.
        """
        nonlocal shard_index, sample_buffer, total_bytes_written, stop_parsing
        if not sample_buffer:
            return True  # Nothing to write, just continue

        total_bytes_written_local, wrote_any, new_stop = write_shard(
            buffer_data=sample_buffer,
            shard_index=shard_index,
            output_dir=output_dir,
            val_split=val_split,
            total_bytes_written=total_bytes_written,
            max_total_bytes=max_total_bytes
        )

        total_bytes_written = total_bytes_written_local
        shard_index += 1
        sample_buffer.clear()

        if new_stop:
            stop_parsing = True
            return False
        return True

    # 2) Iterate pgn files
    for pgn_file in pgn_paths:
        if stop_parsing:
            break

        print(f"Reading {pgn_file} ...")
        with open(pgn_file, "r", errors="ignore") as f:
            while True:
                if stop_parsing:
                    break
                game = chess.pgn.read_game(f)
                if game is None:
                    break  # no more games

                # optional: skip partial or unknown results
                result_str = game.headers.get("Result", "*")
                if result_str not in ["1-0", "0-1", "1/2-1/2"]:
                    continue

                board = game.board()
                for move in game.mainline_moves():

                    # Evaluate the current position with the engine
                    try:
                        info = engine.analyse(board, engine_limit)
                        # info["score"] is a PovScore, can be mate or cp
                        pov_score = info["score"].pov(board.turn)
                        if pov_score.is_mate():
                            # If mate is found, saturate score
                            mate_in = pov_score.mate()
                            eval_cp = 1000 if mate_in > 0 else -1000
                        else:
                            eval_cp = pov_score.score()

                        value_label = convert_cp_to_value(eval_cp, clamp=1000.0)

                    except Exception as e:
                        print(f"Engine analysis error: {e}")
                        # fallback
                        value_label = 0.0

                    # Convert move to index
                    idx = move_to_index(move)
                    if 0 <= idx < NUM_MOVES:
                        # Encode board to array
                        board_arr = encode_board(board)
                        sample_buffer.append((board_arr, idx, value_label))

                        # if buffer is large, flush
                        if len(sample_buffer) >= shard_buffer_size:
                            keep_going = flush_buffer()
                            if not keep_going:
                                break

                    board.push(move)

                games_processed += 1
                if limit is not None and games_processed >= limit:
                    break

    # 3) Flush leftover samples
    if sample_buffer and not stop_parsing:
        flush_buffer()

    # 4) Clean up
    engine.quit()
    print(f"\nDone. Parsed {games_processed} games total.")
    print(f"Total bytes written: {total_bytes_written}")
    if stop_parsing:
        print("Stopped early because we reached the byte limit.")
    else:
        print("All data written without hitting byte limit.")


###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    pgn_files = [
        '/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/games_data/lichess_elite_2023-11.pgn',
    ]

    parse_pgn_files_sharded(
        pgn_paths=pgn_files,
        output_dir="engine_eval_tfrecords",
        shard_buffer_size=20000,    # e.g., flush after 20k samples
        val_split=0.2,             # 20% for validation
        limit=None,                # or specify a game-limit
        max_total_bytes=20 * 1024**3,  # 20 GB limit
        engine_path="/Users/mateuszzieba/Desktop/dev/chess/stockfish/src/stockfish",
        engine_depth=5
    )