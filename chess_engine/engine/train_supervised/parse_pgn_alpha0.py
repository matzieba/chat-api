import os
import random
import chess
import chess.pgn
import numpy as np
import tensorflow as tf

# Adjust if needed (must contain encode_board() and move_to_index()).
from chess_engine.engine.small_model.transformer.environment import (
    encode_board, move_to_index
)


def create_example(board_array, move_idx, value):
    """
    Convert (board_array, move_idx, value) into a TF Example with
    features 'board', 'move', and 'value'.
      board_array: float32 [64,14]
      move_idx: int in [0..4095]
      value: float in [-1.0, 0.0, +1.0] (from perspective of side to move)
    """
    board_bytes = board_array.tobytes()
    feature = {
        "board": tf.train.Feature(bytes_list=tf.train.BytesList(value=[board_bytes])),
        "move": tf.train.Feature(int64_list=tf.train.Int64List(value=[move_idx])),
        "value": tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def get_final_result(game_result, side_to_move_is_white):
    """
    Convert a PGN result string + which side is moving into a float in [-1,0,+1].
      - "1-0": White wins
      - "0-1": Black wins
      - "1/2-1/2": Draw

    For the side to move:
      +1 if that side eventually wins, -1 if it loses, 0 if draw.
    """
    if game_result == "1-0":
        base_value = +1.0
    elif game_result == "0-1":
        base_value = -1.0
    else:
        base_value = 0.0
    return base_value if side_to_move_is_white else -base_value


def write_shard(
        buffer_data,
        shard_index,
        output_dir,
        val_split=0.2,
        total_bytes_written=0,
        max_total_bytes=100 * 1024 ** 3
):
    """
    Shuffle and write out the positions from 'buffer_data' into two shard files:
       train-shard-XXX.tfrecord
       val-shard-XXX.tfrecord
    in the given output_dir, up to max_total_bytes total.

    buffer_data: list of (board_array, move_idx, value)
    shard_index: integer index for shard numbering
    total_bytes_written: how many bytes have been written so far
    max_total_bytes: total budget in bytes for train+val combined

    Returns:
      (new_total_bytes, wrote_anything, stop_parsing)
      - new_total_bytes: updated total bytes after writing
      - wrote_anything : True if at least one record was written, else False
      - stop_parsing   : True if we've reached the limit and should stop all further parsing
    """
    # Shuffle data for this shard
    random.shuffle(buffer_data)
    val_count = int(len(buffer_data) * val_split)
    val_chunk = buffer_data[:val_count]
    train_chunk = buffer_data[val_count:]

    wrote_anything = False
    stop_parsing = False

    # Build shard filenames
    train_shard_path = os.path.join(output_dir, f"train-shard-{shard_index:03d}.tfrecord")
    val_shard_path = os.path.join(output_dir, f"val-shard-{shard_index:03d}.tfrecord")

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
        # Write val shard
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

    # Log how many we actually wrote
    print(f"Shard {shard_index:03d} -> train: wrote up to {len(train_chunk)} (some may be partial if limit reached)")
    print(f"Shard {shard_index:03d} -> val:   wrote up to {len(val_chunk)}   (some may be partial if limit reached)")
    print(f"Total bytes so far: {total_bytes_written} / {max_total_bytes} (limit)")

    return total_bytes_written, wrote_anything, stop_parsing


def parse_pgn_files_sharded(
        pgn_paths,
        output_dir="data_tfrecords_sharded",
        skip_promotion=True,
        shard_buffer_size=50000,
        val_split=0.2,
        limit=None,
        max_total_bytes=100 * 1024 ** 3
):
    """
    Reads PGN files, produces (board, move, value) samples in memory up to
    'shard_buffer_size' at a time, then writes them out to new shard pairs:
       train-shard-XXX.tfrecord / val-shard-XXX.tfrecord

    Also enforces a size limit for combined train+val data across all shards.

    Arguments:
      pgn_paths         : List of PGN file paths
      output_dir        : Directory to write shard files
      skip_promotion    : If True, skip any moves with promotion
      shard_buffer_size : #samples to hold in memory before writing a shard
      val_split         : fraction of data for validation (rest is train)
      limit             : If set, stops after parsing this many GAMES
      max_total_bytes   : Byte budget for entire dataset (train+val combined).

    The process stops early if we reach the size limit.
    """
    os.makedirs(output_dir, exist_ok=True)

    sample_buffer = []
    games_processed = 0
    shard_index = 0
    total_bytes_written = 0
    stop_parsing = False

    def flush_buffer():
        """
        Attempts to write out the current buffer to a new shard pair,
        updates local shard_index & total_bytes_written globally.
        Returns True if we should continue parsing, False if we've hit the limit.
        """
        nonlocal shard_index, sample_buffer, total_bytes_written, stop_parsing

        if not sample_buffer:
            return True  # nothing to flush, just continue

        if stop_parsing:
            return False  # we've already decided no more writing

        # Write one shard pair
        total_bytes_written, wrote_any, new_stop = write_shard(
            buffer_data=sample_buffer,
            shard_index=shard_index,
            output_dir=output_dir,
            val_split=val_split,
            total_bytes_written=total_bytes_written,
            max_total_bytes=max_total_bytes
        )

        shard_index += 1
        sample_buffer.clear()

        if new_stop:
            stop_parsing = True
            return False

        return True

    # --------------------------------------------
    # 1) Read PGNs, fill up buffer, flush to shards
    # --------------------------------------------
    for pgn_file in pgn_paths:
        if stop_parsing:
            break

        with open(pgn_file, "r", errors="ignore") as f:
            while True:
                if stop_parsing:
                    break

                game = chess.pgn.read_game(f)
                if game is None:
                    break  # No more games in this file

                result_str = game.headers.get("Result", "*")
                if result_str not in ["1-0", "0-1", "1/2-1/2"]:
                    # skip partial or unknown results
                    continue

                board = game.board()
                for move in game.mainline_moves():
                    if skip_promotion and move.promotion is not None:
                        board.push(move)
                        continue

                    idx = move_to_index(move)
                    if not (0 <= idx < 4096):
                        board.push(move)
                        continue

                    side_white = (board.turn == chess.WHITE)
                    final_val = get_final_result(result_str, side_white)

                    sample_buffer.append((encode_board(board), idx, final_val))

                    # If we exceed buffer, flush to new shard
                    if len(sample_buffer) >= shard_buffer_size:
                        keep_going = flush_buffer()
                        if not keep_going:
                            break

                    board.push(move)

                games_processed += 1
                if limit is not None and games_processed >= limit:
                    break

    # --------------------------------------------
    # 2) Flush any leftover buffer data to final shard
    # --------------------------------------------
    if not stop_parsing and sample_buffer:
        flush_buffer()

    print(f"\nDone. Parsed {games_processed} games total.")
    print(f"Total bytes written: {total_bytes_written} (limit={max_total_bytes})")
    if stop_parsing:
        print("Stopped early due to size limit.")
    else:
        print("Size limit not reached; all data written.")


if __name__ == "__main__":
    # Example usage:
    #  - parse games from a list of PGN files
    #  - create shards of 50k positions each
    #  - keep total dataset (train+val) under 100 GB
    pgn_files = [
        "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/games_data_1/lichess_elite_2021-02.pgn"
    ]
    parse_pgn_files_sharded(
        pgn_paths=pgn_files,
        output_dir="data_tfrecords_sharded",
        skip_promotion=True,
        shard_buffer_size=50000,
        val_split=0.2,
        limit=None,
        max_total_bytes=100 * 1024 ** 3  # example: 100 GB
    )