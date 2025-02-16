import json
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, Add,
    Flatten, Dense, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

BOARD_SHAPE = (64, 14)
POLICY_SIZE = 8192
BATCH_SIZE  = 32
EPOCHS      = 2

def count_tfrecord_examples_cached(tfrecord_pattern, cache_file="tfrecord_count_cache.json"):
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
    else:
        cache_data = {}

    if tfrecord_pattern in cache_data:
        return cache_data[tfrecord_pattern]

    total_count = 0
    for file_path in tf.io.gfile.glob(tfrecord_pattern):
        dataset = tf.data.TFRecordDataset(file_path)
        for _ in dataset:
            total_count += 1

    cache_data[tfrecord_pattern] = total_count
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)

    return total_count

def parse_tfrecord_fn(example_proto):
    feature_desc = {
        "board":  tf.io.FixedLenFeature([], tf.string),
        "policy": tf.io.FixedLenFeature([], tf.string),
        "value":  tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_desc)

    board_raw  = tf.io.decode_raw(parsed["board"],  tf.float16)
    board_f32  = tf.cast(board_raw, tf.float32)
    board_f32  = tf.reshape(board_f32, BOARD_SHAPE)

    policy_raw = tf.io.decode_raw(parsed["policy"], tf.float16)
    policy_f32 = tf.cast(policy_raw, tf.float32)
    policy_f32 = tf.reshape(policy_f32, [POLICY_SIZE])

    value_f32  = tf.reshape(parsed["value"], [1])  # shape [1]
    return board_f32, (policy_f32, value_f32)


def build_dataset(tfrecord_pattern, batch_size, shuffle_buffer=10000, is_training=True):

    files = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=is_training)
    # Interleave
    ds = files.interleave(
        lambda fname: tf.data.TFRecordDataset(fname, compression_type=""),
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training
    )
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def residual_block(x, filters=64, kernel_size=3):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def create_model(num_res_blocks=4, filters=64, policy_dim=POLICY_SIZE):
    inputs = Input(shape=BOARD_SHAPE, name="board_input")
    x = Reshape((8, 8, 14))(inputs)

    x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for _ in range(num_res_blocks):
        x = residual_block(x, filters=filters, kernel_size=3)

    x = Flatten()(x)
    policy_out = Dense(policy_dim, activation='softmax', name='policy_head')(x)
    value_out  = Dense(1, activation='tanh', name='value_head')(x)

    model = Model(inputs=inputs, outputs=[policy_out, value_out])
    model.compile(
        optimizer=Adam(1e-4),
        loss={
            'policy_head': 'categorical_crossentropy',
            'value_head': 'mse'
        },
        loss_weights={
            'policy_head': 1.0,
            'value_head': 1.0
        },
        metrics={
            'policy_head': ['categorical_accuracy'],
            'value_head': ['mse']
        }
    )
    return model


def main(model_path):
    train_pattern = "data_tfrecords_full_policy/shard-*.tfrecord"
    val_pattern   = "data_tfrecords_full_policy/shard-*.tfrecord"
    train_count = count_tfrecord_examples_cached(train_pattern)
    val_count = count_tfrecord_examples_cached(val_pattern)
    print(f"Training examples: {train_count}")
    print(f"Validation examples: {val_count}")
    steps_per_epoch = train_count // BATCH_SIZE
    validation_steps = val_count // BATCH_SIZE // 10

    print(f"steps_per_epoch: {steps_per_epoch}")
    print(f"validation_steps: {validation_steps}")
    train_ds = build_dataset(
        tfrecord_pattern=train_pattern,
        batch_size=BATCH_SIZE,
        shuffle_buffer=10000,
        is_training=True
    )

    val_ds = build_dataset(
        tfrecord_pattern=val_pattern,
        batch_size=BATCH_SIZE,
        shuffle_buffer=1,
        is_training=False
    )

    if model_path is not None and os.path.isfile(model_path):
        print(f"Loading existing model from {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("No valid model path provided, building a fresh model.")
        model = create_model(num_res_blocks=4, filters=64, policy_dim=POLICY_SIZE)
    model.summary()

    ckpt_dir = "checkpoints_full_policy"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, "epoch_{epoch:02d}_valLoss_{val_loss:.4f}.keras"),
        save_freq='epoch',
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=False
    )

    model.fit(
        train_ds.repeat(),
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    model.save("alphazero_full_policy_model.keras")
    print("Model saved to alphazero_full_policy_model.keras")


if __name__ == "__main__":
    main('/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/train_supervised/alphazero_full_policy_model.keras')