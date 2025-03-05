import os
import random

import chess
import chess.pgn
import chess.engine
import numpy as np
import tensorflow as tf

# Utility / Hyperparameters
NUM_MOVES = 64 * 64 * 5  # from_sq * 64 * 5 + to_sq * 5 + promo_code
MAX_TOTAL_BYTES = 100 * 1024**3  # 100 GB limit (adjust as needed)


def encode_single_board(board: chess.Board) -> np.ndarray:
    """
    Returns a float32 array of shape (64, 17).

    Plane layout (plane indices):
      0  : White Pawn
      1  : White Knight
      2  : White Bishop
      3  : White Rook
      4  : White Queen
      5  : White King
      6  : Black Pawn
      7  : Black Knight
      8  : Black Bishop
      9  : Black Rook
      10 : Black Queen
      11 : Black King
      12 : Castling rights (integer 0..15) scaled to [0..1], replicated
      13 : Side to move (1.0 = White, 0.0 = Black), replicated
      14 : En-passant square (1.0 where the EP capture square is, else 0.0)
      15 : Halfmove clock (scaled/clamped to [0..1]), replicated
      16 : Fullmove number (scaled/clamped to [0..1]), replicated
    """
    enc = np.zeros((64, 17), dtype=np.float32)

    # 1) Fill piece planes
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            piece_type = piece.piece_type  # 1..6
            color_offset = 0 if piece.color == chess.WHITE else 6
            plane_idx = color_offset + (piece_type - 1)
            enc[sq, plane_idx] = 1.0

    # 2) Castling rights in plane 12
    castling_code = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_code |= 1  # bit 0
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_code |= 2  # bit 1
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_code |= 4  # bit 2
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_code |= 8  # bit 3
    enc[:, 12] = castling_code / 15.0

    # 3) Side to move in plane 13
    enc[:, 13] = float(board.turn == chess.WHITE)

    # 4) En-passant square in plane 14
    if board.ep_square is not None:
        enc[board.ep_square, 14] = 1.0

    # 5) Halfmove clock in plane 15 (clamp at 100 for scaling)
    halfmove_val = min(board.halfmove_clock, 100) / 100.0
    enc[:, 15] = halfmove_val

    # 6) Fullmove number in plane 16 (clamp at 1000)
    fullmove_val = min(board.fullmove_number, 1000) / 1000.0
    enc[:, 16] = fullmove_val

    return enc


class BoardHistory:
    """
    Maintains a rolling window of up to N board encodings.
    Each call to push(board) appends the encoded board state.
    get_encoded() returns a (64, 17*N) array, padding with zeros if fewer than N.
    """

    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = []

    def push(self, board: chess.Board):
        enc = encode_single_board(board)
        self.frames.append(enc)
        # Keep only the last num_frames boards
        while len(self.frames) > self.num_frames:
            self.frames.pop(0)

    def get_encoded(self) -> np.ndarray:
        """
        Returns shape (64, 17 * num_frames).
        If the game hasn't had that many moves yet, zero-pad the missing frames.
        """
        current_count = len(self.frames)
        missing = self.num_frames - current_count
        padded = []

        # Zero-fill for missing frames
        for _ in range(missing):
            padded.append(np.zeros((64, 17), dtype=np.float32))

        # Then the actual frames, from oldest to newest
        padded.extend(self.frames)

        # Concatenate along the plane dimension (axis=1)
        # So we get shape (64, 17 * num_frames)
        return np.concatenate(padded, axis=1)


def move_to_index(move: chess.Move) -> int:
    """
    Convert chess.Move to an integer in [0, 20480).
    from_sq * 64 * 5 + to_sq * 5 + promo_code
    """
    promo_code = 0
    if move.promotion:
        if move.promotion == chess.KNIGHT:
            promo_code = 1
        elif move.promotion == chess.BISHOP:
            promo_code = 2
        elif move.promotion == chess.ROOK:
            promo_code = 3
        elif move.promotion == chess.QUEEN:
            promo_code = 4

    return (
        move.from_square * 64 * 5
        + move.to_square * 5
        + promo_code
    )


def convert_cp_to_value(score_cp: float, clamp=1000.0) -> float:
    """
    Convert centipawn score in [-∞, +∞] to [-1.0..+1.0],
    saturating at ±clamp.
    """
    if score_cp > clamp:
        return 1.0
    elif score_cp < -clamp:
        return -1.0
    else:
        return score_cp / clamp


def create_example(board_array, move_idx, value):
    """
    TFRecord Example creation.
    board_array is a (64, 17 * N) float array if using N frames.
    """
    board_bytes = board_array.tobytes()
    features = {
        "board": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[board_bytes])
        ),
        "move": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[move_idx])
        ),
        "value": tf.train.Feature(
            float_list=tf.train.FloatList(value=[value])
        ),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_shard(
    buffer_data,
    shard_index,
    output_dir,
    val_split=0.2,
    total_bytes_written=0,
    max_total_bytes=MAX_TOTAL_BYTES
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
    train_shard_path = os.path.join(
        output_dir, f"train-shard-{shard_index:03d}.tfrecord"
    )
    val_shard_path = os.path.join(
        output_dir, f"val-shard-{shard_index:03d}.tfrecord"
    )

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

    print(
        f"Shard {shard_index:03d} -> "
        f"train: {len(train_chunk)} (partial if limit reached)"
    )
    print(
        f"Shard {shard_index:03d} -> "
        f"val: {len(val_chunk)} (partial if limit reached)"
    )
    print(
        f"Total bytes so far: {total_bytes_written} / {max_total_bytes} (limit)"
    )

    return total_bytes_written, wrote_anything, stop_parsing


def parse_pgn_files_sharded(
    pgn_paths,
    output_dir="data_tfrecords_sharded",
    shard_buffer_size=50000,
    val_split=0.2,
    limit=None,
    max_total_bytes=MAX_TOTAL_BYTES,
    engine_path="/path/to/stockfish",
    engine_depth=5
):
    """
    Reads PGN files, but uses Stockfish as "teacher":
      - For each position, we query Stockfish for:
          (a) best_move for the policy label
          (b) numeric eval for the value label
      - Then we store (position, best_move_idx, Stockfish_eval) in TFRecords.
      - We now collect 4 frames of board history for each position.
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
    engine_limit = chess.engine.Limit(depth=engine_depth)

    def flush_buffer():
        nonlocal shard_index, sample_buffer, total_bytes_written, stop_parsing
        if not sample_buffer:
            return True  # Nothing to write, just continue

        (
            total_bytes_written_local,
            wrote_any,
            new_stop
        ) = write_shard(
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

                # Skip incomplete or weird results if you like
                result_str = game.headers.get("Result", "*")
                if result_str not in ["1-0", "0-1", "1/2-1/2"]:
                    continue

                board = game.board()
                board_history = BoardHistory(num_frames=4)

                # Step through the mainline moves
                # but use Stockfish for the policy label
                for move in game.mainline_moves():
                    # 1) Add the current board to the history
                    board_history.push(board)

                    # 2) Evaluate the current position with the engine
                    try:
                        info = engine.analyse(board, engine_limit)
                        pov_score = info["score"].pov(board.turn)

                        if pov_score.is_mate():
                            mate_in = pov_score.mate()
                            eval_cp = 1000 if mate_in > 0 else -1000
                        else:
                            eval_cp = pov_score.score()

                        value_label = convert_cp_to_value(eval_cp, clamp=1000.0)
                        best_move = engine.play(board, engine_limit).move

                        if best_move is None:
                            # In rare cases, engine might fail
                            continue

                        best_move_idx = move_to_index(best_move)
                        if 0 <= best_move_idx < NUM_MOVES:
                            board_arr_4frames = board_history.get_encoded()
                            sample_buffer.append(
                                (board_arr_4frames, best_move_idx, value_label)
                            )

                            # If the buffer is large, flush
                            if len(sample_buffer) >= shard_buffer_size:
                                keep_going = flush_buffer()
                                if not keep_going:
                                    break
                    except Exception as e:
                        print(f"Engine analysis error: {e}")
                        continue

                    # 5) Push the PGN move to proceed
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


if __name__ == "__main__":
    pgn_files = [
        "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/"
        "chess_engine/engine/games_data/lichess_elite_2023-01.pgn"
    ]
    parse_pgn_files_sharded(
        pgn_paths=pgn_files,
        output_dir="engine_eval_tfrecords",
        shard_buffer_size=20000,
        val_split=0.2,
        limit=None,
        max_total_bytes=100 * 1024**3,
        engine_path=(
            "/Users/mateuszzieba/Desktop/dev/chess/stockfish/src/stockfish"
        ),
        engine_depth=3
    )