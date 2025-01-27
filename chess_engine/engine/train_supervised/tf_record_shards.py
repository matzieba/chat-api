import os
import chess
import chess.pgn
import numpy as np
import tensorflow as tf

from chess_engine.engine.small_model.transformer.environment import (
    encode_board, move_to_index
)

def create_example(board_array, move_idx):
    """Convert board_array (shape [64,14]) to TF Example with features 'board' and 'move'."""
    board_bytes = board_array.tobytes()
    feature = {
        "board": tf.train.Feature(bytes_list=tf.train.BytesList(value=[board_bytes])),
        "move":  tf.train.Feature(int64_list=tf.train.Int64List(value=[move_idx]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_and_write_tfrecords(
    pgn_paths,
    shard_size=50000,
    output_dir="data_tfrecords_1",
    skip_promotion=True,
    max_total_bytes=100 * 1024**3  # 100 GB
):
    """
    Reads PGN files, writes positions into TFRecord shards, each up to 'shard_size' positions.
    Stops altogether once total size exceeds 'max_total_bytes'.
    """
    os.makedirs(output_dir, exist_ok=True)

    shard_index = 0
    sample_count_in_shard = 0
    total_bytes = 0  # track how many bytes we've written so far

    writer = tf.io.TFRecordWriter(
        os.path.join(output_dir, f"train_shard_{shard_index}.tfrecord")
    )

    positions_written = 0
    stop_parsing = False  # if we hit 100GB, we'll set this to True

    for pgn_file in pgn_paths:
        if stop_parsing:
            # Already reached 100GB in a previous file
            break

        with open(pgn_file, "r", errors="ignore") as f:
            while True:
                if stop_parsing:
                    break

                game = chess.pgn.read_game(f)
                if game is None:
                    break  # no more games in this file

                board = game.board()
                for move in game.mainline_moves():
                    # Skip promotions if desired
                    if skip_promotion and move.promotion is not None:
                        board.push(move)
                        continue

                    board_array = encode_board(board)  # shape [64,14], float32
                    idx = move_to_index(move)

                    # Skip if out of range (0..4095)
                    if idx < 0 or idx >= 4096:
                        board.push(move)
                        continue

                    # Create TF Example
                    example = create_example(board_array, idx)
                    serialized_example = example.SerializeToString()

                    # Update total_bytes
                    total_bytes += len(serialized_example)

                    # If we have exceeded the limit, stop parsing
                    if total_bytes >= max_total_bytes:
                        print(f"Reached {max_total_bytes} bytes. Stopping.")
                        stop_parsing = True
                        break

                    # Write to TFRecord
                    writer.write(serialized_example)
                    positions_written += 1
                    sample_count_in_shard += 1

                    # Advance board
                    board.push(move)

                    # If we reached shard_size, open a new shard
                    if sample_count_in_shard >= shard_size:
                        writer.close()
                        shard_index += 1
                        sample_count_in_shard = 0
                        writer = tf.io.TFRecordWriter(
                            os.path.join(output_dir, f"train_shard_{shard_index}.tfrecord")
                        )

                # End of this game
            # End of pgn_file
        # if pgn_file

    # Close last writer
    writer.close()
    print(f"Done. Wrote {positions_written} positions total across {shard_index + 1} shards.")
    print(f"Approx total bytes written: {total_bytes}")

if __name__ == "__main__":
    pgn_files = [
        "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/games_data_1/lichess_elite_2021-02.pgn"
    ]
    parse_and_write_tfrecords(
        pgn_files,
        shard_size=50000,
        output_dir="data_tfrecords_1",
        skip_promotion=True,
        max_total_bytes=100 * 1024**3  # 100 GB
    )