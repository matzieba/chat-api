import random
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

from utils import NUM_SQUARES, NUM_PROMOTION_OPTIONS, NUM_MOVES
from utils import encode_move, convert_board_to_sequence, get_legal_moves_mask

class TrainingProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: loss = {logs['loss']:.4f}, accuracy = {logs['accuracy']:.4f}, "
              f"val_loss = {logs['val_loss']:.4f}, val_accuracy = {logs['val_accuracy']:.4f}")

def data_generator(file_path, batch_size=256, max_games=None, is_training=True):
    def generator():
        with open(file_path, "r") as pgn_file:
            game_counter = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  # End of file reached

                # Skip some games for validation
                if is_training and random.random() < 0.1:
                    continue  # Skip this game for training set
                if not is_training and random.random() >= 0.1:
                    continue  # Skip this game for validation set

                board = game.board()
                for move in game.mainline_moves():
                    state = convert_board_to_sequence(board)
                    try:
                        action_index = encode_move(move)
                    except (ValueError, IndexError):
                        board.push(move)
                        continue
                    # Get legal moves mask
                    mask = get_legal_moves_mask(board)
                    yield {'input_seq': state, 'mask_input': mask.astype(np.float32)}, action_index
                    board.push(move)
                game_counter += 1
                if max_games and game_counter >= max_games:
                    break  # Stop after max_games
    return generator

def create_dataset(file_path, batch_size, max_games, is_training):
    output_types = ({'input_seq': tf.int32, 'mask_input': tf.float32}, tf.int32)
    output_shapes = ({'input_seq': (64,), 'mask_input': (NUM_MOVES,)}, ())
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
        if model.output_shape[-1] != NUM_MOVES:
            raise ValueError(f"Number of moves ({NUM_MOVES}) does not match the model's output layer size ({model.output_shape[-1]}).")
    else:
        print("Creating new model...")
        from model import create_model  # Ensure this import works correctly
        model = create_model()

    # Create datasets
    batch_size = 64
    print("Creating training dataset...")
    train_dataset = create_dataset(file_path, batch_size, max_games, is_training=True)
    print("Creating validation dataset...")
    val_dataset = create_dataset(file_path, batch_size, max_games, is_training=False)

    # Callbacks
    checkpoint = ModelCheckpoint('best_transformer_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    training_progress = TrainingProgressCallback()
    callbacks = [checkpoint, early_stopping, reduce_lr, training_progress]

    print("Starting model training...")
    # Train model
    epochs = 5  # Adjust as needed
    steps_per_epoch = 15  # Adjust as needed
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
    model_save_path = 'best_trained_transformer_model.keras'
    model.save(model_save_path, include_optimizer=True)
    print(f"Best model saved to {model_save_path}")
    return model

# Start training
model = train_supervised(
    '/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/games_data/lichess_elite_2020-06.pgn',
    max_games=20,
    existing_model_path="/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/models/best_trained_transformer_model.keras"
)