import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint  # <-- Import the checkpoint callback


###############################################################################
# 1. TFRecord Parsing
###############################################################################
def parse_tfrecord(serialized_example):
    """
    Parse a single TFRecord Example. Expects:
      'board': raw bytes storing float32 [64,14]
      'move':  int64 index in [0..4095]
      'value': float in [-1..+1]

    Returns: (board_tensor), (move_idx, value_label)
      so we can feed it to a two‐head model: outputs = [policy_head, value_head].
    """
    feature_desc = {
        'board': tf.io.FixedLenFeature([], tf.string),
        'move': tf.io.FixedLenFeature([], tf.int64),
        'value': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(serialized_example, feature_desc)

    # Decode board
    board_bytes = tf.io.decode_raw(parsed['board'], tf.float32)
    board_tensor = tf.reshape(board_bytes, (64, 14))

    move_idx = parsed['move']
    value_label = parsed['value']

    # Return (features, labels) with multi‐output model
    return board_tensor, (move_idx, value_label)


###############################################################################
# 2. CNN with Two Heads: Policy (4096) & Value (1)
###############################################################################
def create_alphazero_cnn():
    """
    Builds a CNN trunk with two heads:
      - policy_head: 4096‐way softmax for moves
      - value_head: scalar in [-1..+1]
    """
    inputs = Input(shape=(64, 14), name="board_input")

    x = Reshape((8, 8, 14))(inputs)  # Convert to 8x8 with 14 channels

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)  # -> shape (None,4,4,64)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)  # -> shape (None,2,2,128)

    x = Flatten()(x)  # shape ~ (None, 512)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Policy head: output shape (4096), softmax
    policy_head = Dense(4096, activation='softmax', name='policy_head')(x)

    # Value head: output shape (1), tanh to keep in [-1..+1]
    value_head = Dense(1, activation='tanh', name='value_head')(x)

    model = Model(inputs=inputs, outputs=[policy_head, value_head])

    # Two separate losses:
    #   policy_head -> sparse_categorical_crossentropy
    #   value_head  -> mean squared error
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={
            'policy_head': 'sparse_categorical_crossentropy',
            'value_head': 'mse'
        },
        loss_weights={
            'policy_head': 1.0,
            'value_head': 1.0
        },
        metrics={
            'policy_head': 'accuracy',
            'value_head': 'mse'
        }
    )

    return model


###############################################################################
# 3. Main Training Logic
###############################################################################
def build_dataset_from_shards(file_pattern, batch_size, shuffle_buffer=50000, is_training=True):
    """
    Build a tf.data.Dataset from multiple TFRecord shards specified by file_pattern.
    Example: file_pattern="data_tfrecords_sharded/train-*.tfrecord"

    If is_training=True, we shuffle; otherwise we skip shuffle.
    """
    files_dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

    dataset = files_dataset.interleave(
        lambda fname: tf.data.TFRecordDataset(fname),
        cycle_length=4,  # how many files to read concurrently
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training
    )

    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def main():
    train_pattern = "data_tfrecords_sharded/train-*.tfrecord"
    val_pattern = "data_tfrecords_sharded/val-*.tfrecord"

    batch_size = 128
    epochs = 5

    print("Building training dataset from shards:", train_pattern)
    train_dataset = build_dataset_from_shards(
        file_pattern=train_pattern,
        batch_size=batch_size,
        shuffle_buffer=50000,
        is_training=True
    )

    print("Building validation dataset from shards:", val_pattern)
    val_dataset = build_dataset_from_shards(
        file_pattern=val_pattern,
        batch_size=batch_size,
        shuffle_buffer=1,  # minimal or no shuffle for validation
        is_training=False
    )

    model_path = ""
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating a fresh AlphaZero‐style CNN...")
        model = create_alphazero_cnn()

    model.summary()

    # ----------------------------------------------------------------------
    # 4. Create a checkpoint callback to save the model after each epoch
    # ----------------------------------------------------------------------
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "epoch_{epoch:02d}_valLoss_{val_loss:.4f}.keras"),
        save_freq='epoch',
        save_weights_only=False,  # set to True if you only want weights
        monitor='val_loss',  # or you could monitor 'val_policy_head_loss' if you prefer
        mode='min',
        save_best_only=False  # set to True if you only want to save the best epoch
    )

    # ----------------------------------------------------------------------
    # 5. Train with callbacks
    # ----------------------------------------------------------------------
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback]  # Pass checkpoint to callbacks
    )

    # Save final model (if you still want a single file at end)
    save_model_path = "./alphazero_cnn_model.keras"
    model.save(save_model_path)
    print(f"Model fully saved to {save_model_path}")


if __name__ == "__main__":
    main()