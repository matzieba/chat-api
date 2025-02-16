import os
import random
import chess
import chess.pgn
import numpy as np
import tensorflow as tf

def move_to_index(move):
    from_sq = move.from_square
    to_sq = move.to_square
    base = from_sq * 64 + to_sq
    if move.promotion == chess.QUEEN:
        return 4096 + base
    else:
        return base

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

def get_final_result(result, side_to_move_is_white):
    if result == "1-0":
        base_val = 1.0
    elif result == "0-1":
        base_val = -1.0
    else:
        base_val = 0.0
    return base_val if side_to_move_is_white else -base_val


def create_example(board_array, policy_vector, value):
    board_array_16 = board_array.astype(np.float16)
    policy_vec_16  = policy_vector.astype(np.float16)

    board_bytes  = board_array_16.tobytes()
    policy_bytes = policy_vec_16.tobytes()

    feature = {
        "board": tf.train.Feature(bytes_list=tf.train.BytesList(value=[board_bytes])),
        "policy": tf.train.Feature(bytes_list=tf.train.BytesList(value=[policy_bytes])),
        "value": tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_shard(buffer_data, shard_index, output_dir, max_size_bytes, total_bytes_written):
    shard_path = os.path.join(output_dir, f"shard-{shard_index:03d}.tfrecord")

    with tf.io.TFRecordWriter(shard_path) as writer:
        for (board_arr, policy_vec, val) in buffer_data:
            ex = create_example(board_arr, policy_vec, val)
            serialized = ex.SerializeToString()

            if total_bytes_written + len(serialized) > max_size_bytes:
                return total_bytes_written, True  # done

            writer.write(serialized)
            total_bytes_written += len(serialized)

    return total_bytes_written, False


def parse_pgn_files(
    pgn_paths,
    output_dir="data_tfrecords_full_policy",
    skip_promotion=False,
    shard_buffer_size=50000,
    max_total_bytes=1024**3,
    limit_games=None
):
    os.makedirs(output_dir, exist_ok=True)

    sample_buffer = []
    shard_index = 0
    total_bytes_written = 0
    games_parsed = 0

    def flush_buffer():
        nonlocal shard_index, total_bytes_written
        if not sample_buffer:
            return
        total_bytes_written, reached_limit = write_shard(
            buffer_data=sample_buffer,
            shard_index=shard_index,
            output_dir=output_dir,
            max_size_bytes=max_total_bytes,
            total_bytes_written=total_bytes_written
        )
        shard_index += 1
        sample_buffer.clear()
        return reached_limit

    for pgn_file in pgn_paths:
        print(f"Reading PGN: {pgn_file}")
        with open(pgn_file, "r", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                result_str = game.headers.get("Result", "*")
                if result_str not in ["1-0", "0-1", "1/2-1/2"]:
                    continue  # skip incomplete or unknown
                board = game.board()
                for move in game.mainline_moves():
                    if skip_promotion and move.promotion is not None:
                        board.push(move)
                        continue

                    move_idx = move_to_index(move)
                    if not (0 <= move_idx < 8192):
                        board.push(move)
                        continue

                    # Build a one-hot policy vector
                    policy_vec = np.zeros((8192,), dtype=np.float32)
                    policy_vec[move_idx] = 1.0

                    side_is_white = (board.turn == chess.WHITE)
                    value_label = get_final_result(result_str, side_is_white)

                    board_encoding = encode_board(board)
                    sample_buffer.append((board_encoding, policy_vec, value_label))
                    if len(sample_buffer) >= shard_buffer_size:
                        if flush_buffer():

                            return

                    board.push(move)

                games_parsed += 1
                if limit_games is not None and games_parsed >= limit_games:
                    break

    # flush leftover
    flush_buffer()
    print(f"Done. Parsed {games_parsed} games total.")
    print(f"Output shards in {output_dir}. Total bytes written ~{total_bytes_written}.")


if __name__ == "__main__":
    # Example usage:
    pgn_files = [
        # Put your PGN file paths here
        "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/games_data/lichess_elite_2023-11.pgn",
    ]
    parse_pgn_files(
        pgn_paths=pgn_files,
        output_dir="data_tfrecords_full_policy",
        skip_promotion=False,
        shard_buffer_size=20000,
        max_total_bytes=100 * 1024**3,
        limit_games=None
    )