import random
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# Adjust import to match your model’s location:
from chess_engine.engine.small_model.model import create_small_chess_model
from utils import encode_move, board_to_planes, get_legal_moves_mask, NUM_MOVES

class TrainingProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}: loss = {logs.get('loss', 0):.4f}, "
              f"policy_accuracy = {logs.get('policy_output_accuracy', 0):.4f}, "
              f"val_loss = {logs.get('val_loss', 0):.4f}, "
              f"val_policy_accuracy = {logs.get('val_policy_output_accuracy', 0):.4f}")

def data_generator(file_path, batch_size=256, max_games=None, is_training=True):
    """
    Yields input_dict, output_dict in a form suitable for
    model.compile(loss={"policy_output": "categorical_crossentropy", ...}).
    The policy is returned as a one-hot vector of shape (NUM_MOVES,).
    """
    def generator():
        try:
            with open(file_path, "r") as pgn_file:
                game_counter = 0
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break  # End of file reached

                    # Control which games go to training vs validation:
                    if is_training and random.random() < 0.1:
                        continue  # Skip this game for training set
                    if not is_training and random.random() >= 0.1:
                        continue  # Skip this game for validation set

                    result = game.headers.get("Result", "1/2-1/2")
                    if result == "1-0":
                        value_label = 1.0
                    elif result == "0-1":
                        value_label = -1.0
                    else:
                        value_label = 0.0

                    board = game.board()

                    for move in game.mainline_moves():
                        state = board_to_planes(board)

                        try:
                            action_index = encode_move(move)
                        except (ValueError, IndexError) as e:
                            # If the move cannot be encoded properly, skip it
                            print(f"Skipping move due to encoding error: {e}")
                            board.push(move)
                            continue

                        # Create one-hot for policy
                        policy_one_hot = np.zeros((NUM_MOVES,), dtype=np.float32)
                        policy_one_hot[action_index] = 1.0

                        # Get the legal moves mask
                        mask = get_legal_moves_mask(board).astype(np.float32)

                        # Yield the batch element
                        yield (
                            {
                                'input_board': state,
                                'mask_input': mask,
                            },
                            {
                                'policy_output': policy_one_hot,
                                'value_output': value_label,
                            },
                        )

                        board.push(move)

                    game_counter += 1
                    if max_games and game_counter >= max_games:
                        break
        except Exception as e:
            print(f"Exception in data generator: {e}")
            raise e

    return generator

def create_dataset(file_path, batch_size, max_games, is_training):
    output_types = (
        {'input_board': tf.float32, 'mask_input': tf.float32},
        {'policy_output': tf.float32, 'value_output': tf.float32},
    )
    output_shapes = (
        {'input_board': (8, 8, 14), 'mask_input': (NUM_MOVES,)},
        {'policy_output': (NUM_MOVES,), 'value_output': ()},
    )
    dataset = tf.data.Dataset.from_generator(
        data_generator(file_path, batch_size, max_games, is_training),
        output_types=output_types,
        output_shapes=output_shapes
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_supervised(file_path, max_games=None, existing_model_path=None):
    print("Starting training...")
    if existing_model_path:
        print(f"Loading existing model from {existing_model_path}...")
        model = tf.keras.models.load_model(existing_model_path)
    else:
        print("Creating new model...")
        # Make sure to import or define create_small_chess_model in your environment
        model = create_small_chess_model()

    # Create datasets
    batch_size = 64
    print("Creating training dataset...")
    train_dataset = create_dataset(file_path, batch_size, max_games, is_training=True)
    print("Creating validation dataset...")
    val_dataset = create_dataset(file_path, batch_size, max_games, is_training=False)

    # Callbacks
    checkpoint = ModelCheckpoint('best_trained_model.keras',
                                 monitor='val_policy_output_accuracy',  # matches model metric name
                                 save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_policy_output_accuracy',
                                   mode='max', patience=10,
                                   restore_best_weights=True)
    training_progress = TrainingProgressCallback()

    callbacks = [checkpoint, early_stopping, training_progress]

    print("Starting model training...")
    epochs = 40  # Adjust as needed
    steps_per_epoch = 500  # Adjust as needed
    validation_steps = steps_per_epoch // 10

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    print("Training completed.")
    # Save the model to a file
    model_save_path = 'best_trained_model_small_cnn_l2.keras'
    model.save(model_save_path, include_optimizer=True)
    print(f"Best model saved to {model_save_path}")
    return model

# Example usage
if __name__ == "__main__":
    model = train_supervised(
        "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/games_data/lichess_elite_2020-07.pgn",
        max_games=30000,
        existing_model_path=None,  # or a path to an existing model
    )