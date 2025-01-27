import os
import random
import chess
import chess.pgn
import numpy as np

from chess_engine.engine.small_model.transformer.environment import encode_board, move_to_index


# Assumes environment.py is in the same directory or importable; adjust as needed:


def parse_pgn_files(
    pgn_paths,
    output_path="data_prepared",
    skip_promotion=True,
    limit=None,
    val_split=0.2
):
    """
    Parses PGN files to build a (board -> move index) dataset.
    Each board is encoded via 'encode_board'; each move is turned into a 0..4095 index via 'move_to_index'.
    Skips promotion moves if skip_promotion=True.
    Shuffles, splits into train/val, and saves {train,val}.npz in 'output_path'.
    """
    X = []
    Y = []
    games_processed = 0

    for pgn_file in pgn_paths:
        with open(pgn_file, "r", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break  # No more games in this file

                board = game.board()
                for move in game.mainline_moves():
                    # Optionally skip promotion moves
                    if skip_promotion and move.promotion is not None:
                        board.push(move)
                        continue

                    # Encode current board, store in X
                    X.append(encode_board(board))

                    # Convert move to index [0..4095]. If outside range (unlikely), skip.
                    idx = move_to_index(move)
                    if idx < 0 or idx >= 4096:
                        board.push(move)
                        continue

                    Y.append(idx)
                    board.push(move)

                games_processed += 1
                if limit is not None and games_processed >= limit:
                    break
        if limit is not None and games_processed >= limit:
            break

    # Shuffle the dataset
    combined = list(zip(X, Y))
    random.shuffle(combined)
    X, Y = zip(*combined)
    X = np.array(X, dtype=np.float32)   # shape (N, 64, 14)
    Y = np.array(Y, dtype=np.int32)     # shape (N,)

    # Split train/val
    val_size = int(len(X) * val_split)
    X_val, Y_val = X[:val_size], Y[:val_size]
    X_train, Y_train = X[val_size:], Y[val_size:]

    os.makedirs(output_path, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_path, "train.npz"),
        X_train=X_train,
        Y_train=Y_train
    )
    np.savez_compressed(
        os.path.join(output_path, "val.npz"),
        X_val=X_val,
        Y_val=Y_val
    )

    print(f"Processed {games_processed} games total.")
    print(f"Train set: {len(X_train)} samples, Val set: {len(X_val)} samples.")

if __name__ == "__main__":
    # Example usage: parse up to 1000 games from two PGN files, skip promotion moves,
    # produce a 80/20 train/val split, and store results in 'data_prepared/'.
    pgn_files = ["/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/games_data/lichess_elite_2020-06.pgn"]  # Change to your own PGN paths.
    parse_pgn_files(
        pgn_paths=pgn_files,
        output_path="data_prepared",
        skip_promotion=True,
        limit=100,
        val_split=0.2
    )