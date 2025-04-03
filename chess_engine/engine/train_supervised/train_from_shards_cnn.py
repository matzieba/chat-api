import json
import os
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    Flatten,
    Dropout,
    Reshape,
)
from tensorflow.keras.optimizers import Adam

NUM_MOVES = 64 * 64 * 5
BATCH_SIZE = 512
EPOCHS = 2

TRAIN_PATTERN = "my_refactored_tfrecords/train-*.tfrecord"
VAL_PATTERN = "my_refactored_tfrecords/val-*.tfrecord"


def parse_tfrecord(serialized_example):
    feat_desc = {
        "board": tf.io.FixedLenFeature([], tf.string),
        "move": tf.io.FixedLenFeature([], tf.int64),
        "value": tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(serialized_example, feat_desc)

    # Decode the board bytes as float32 and reshape to (64, 17)
    board_bytes = tf.io.decode_raw(parsed["board"], tf.float32)
    board_tensor = tf.reshape(board_bytes, (64, 17))

    move_idx = parsed["move"]
    value_label = parsed["value"]
    return board_tensor, (move_idx, value_label)


def residual_block(x, filters=128):
    shortcut = x
    x = Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation("relu")(x)
    return x


def create_alphazero_cnn(
        input_shape=(64, 17),
        policy_size=NUM_MOVES,
        num_filters=64,
        num_res_blocks=8,
        dense_units=512,
):
    """
    An AlphaZero‐style CNN with smaller default settings for demonstration.
    Adjust as needed (more blocks, bigger dense layer, etc.).

    input_shape: (64, 17) for a single board frame of 64 squares × 17 planes.
    This is reshaped internally to (8, 8, 17).
    """
    inputs = Input(shape=input_shape, name="board_input")

    # Reshape (64, 17) -> (8, 8, 17)
    x = Reshape((8, 8, input_shape[-1]))(inputs)

    # Initial convolution
    x = Conv2D(num_filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Residual blocks
    for _ in range(num_res_blocks):
        x = residual_block(x, filters=num_filters)

    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(0.2)(x)

    # Policy head: choose among 20,480 moves
    policy_head = Dense(policy_size, activation="softmax", name="policy_head")(x)

    # Value head: single float in [-1,1]
    value_head = Dense(1, activation="tanh", name="value_head")(x)

    model = Model(inputs=inputs, outputs=[policy_head, value_head])
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={
            "policy_head": "sparse_categorical_crossentropy",
            "value_head": "mse",
        },
        loss_weights={
            "policy_head": 1.0,
            "value_head": 1.0,
        },
        metrics={
            "policy_head": "accuracy",
            "value_head": "mse",
        },
    )
    return model


def build_dataset_from_shards(file_pattern, batch_size, is_training=True):
    files_dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

    dataset = files_dataset.interleave(
        lambda fname: tf.data.TFRecordDataset(fname),
        cycle_length=1,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training,
    )
    shuffle_buffer = 10_000
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


def count_tfrecord_examples(file_pattern):
    num_examples = 0
    for filename in tf.io.gfile.glob(file_pattern):
        for _ in tf.data.TFRecordDataset(filename):
            num_examples += 1
    return num_examples


def get_or_compute_counts(train_pattern, val_pattern, cache_file="count.json"):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            if "train_count" in data and "val_count" in data:
                print(f"Using cached counts from {cache_file}...")
                return data["train_count"], data["val_count"]
        except Exception as e:
            print(f"Warning: could not read '{cache_file}', will re-compute. Reason: {e}")

    print("Counting TFRecord examples from scratch...")
    train_count = count_tfrecord_examples(train_pattern)
    val_count = count_tfrecord_examples(val_pattern)
    print(f"train_count = {train_count}, val_count = {val_count}")

    data = {"train_count": train_count, "val_count": val_count}
    with open(cache_file, "w") as f:
        json.dump(data, f)
    print(f"Counts saved to {cache_file}")

    return train_count, val_count


def main():
    train_count, val_count = get_or_compute_counts(
        train_pattern=TRAIN_PATTERN,
        val_pattern=VAL_PATTERN,
        cache_file="counts.json",
    )
    print("Train count:", train_count)
    print("Val count:", val_count)

    steps_per_epoch = train_count // BATCH_SIZE
    val_steps = val_count // BATCH_SIZE

    print(f"Steps per epoch (train) = {steps_per_epoch}")
    print(f"Steps per epoch (val) = {val_steps}")

    print("Building training dataset from:", TRAIN_PATTERN)
    train_dataset = (
        build_dataset_from_shards(
            file_pattern=TRAIN_PATTERN,
            batch_size=BATCH_SIZE,
            is_training=True,
        )
        .repeat()
    )

    print("Building validation dataset from:", VAL_PATTERN)
    val_dataset = (
        build_dataset_from_shards(
            file_pattern=VAL_PATTERN,
            batch_size=BATCH_SIZE,
            is_training=False,
        )
        .repeat()
    )

    # Create or load model
    model_path = (
        "/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/"
        "chess_engine/engine/train_supervised/"
        "my_engine_eval_model_100GB_of_parsed_games_pure_argmax.keras"
    )
    if os.path.exists(model_path):
        print(f"Loading existing model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating a fresh AlphaZero‐style CNN (single‐frame).")
        model = create_alphazero_cnn(
            input_shape=(64, 17),  # single board frame
            policy_size=NUM_MOVES,
            num_filters=128,
            num_res_blocks=8,
            dense_units=512,
        )
        model.summary()

    checkpoint_dir = "./checkpoints_engine_eval"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(
            checkpoint_dir,
            "ckpt_epoch_{epoch:02d}_valLoss_{val_loss:.4f}.keras"
        ),
        save_freq="epoch",
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=False,
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback],
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
    )

    model.save(model_path)
    print(f"Final model saved to: {model_path}")


if __name__ == "__main__":
    main()