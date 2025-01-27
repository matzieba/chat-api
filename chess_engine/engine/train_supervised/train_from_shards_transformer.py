import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D

###############################################################################
# 1. TFRecord Parsing
###############################################################################
def parse_tfrecord(serialized_example):
    """
    Parse a single TFRecord Example. Expects:
      'board': raw bytes storing float32 [64, 14]
      'move':  int64 index in [0..4095]
    """
    feature_description = {
        'board': tf.io.FixedLenFeature([], tf.string),
        'move':  tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)

    # Decode board bytes into a float32 tensor [64,14]
    board_bytes = tf.io.decode_raw(parsed_example['board'], tf.float32)
    board_tensor = tf.reshape(board_bytes, (64, 14))

    # The label is the move index
    move_idx = parsed_example['move']
    return board_tensor, move_idx

def count_tfrecord_samples(filenames):
    """
    Counts total number of samples in the given TFRecord files.
    """
    count = 0
    for fname in filenames:
        for _ in tf.data.TFRecordDataset(fname):
            count += 1
    return count

###############################################################################
# 2. A Minimal Transformer Encoder in Keras
###############################################################################
def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout_rate=0.1):
    """
    A single Transformer encoder block with:
      - MultiHeadAttention (self-attention)
      - Dropout + Residual + LayerNorm
      - Feed-forward block (two Dense layers)
      - Dropout + Residual + LayerNorm
    """
    # First layer normalization + Multi‐head self‐attention
    x_norm = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
    )(x_norm, x_norm)
    x_res = inputs + attention_output  # Residual connection

    # Second layer normalization + Feed‐forward block
    x_norm2 = LayerNormalization(epsilon=1e-6)(x_res)
    ffn = Sequential([
        Dense(ff_dim, activation='relu'),
        Dense(d_model),
    ])
    x_ffn = ffn(x_norm2)
    return x_res + x_ffn  # Another residual connection

def create_transformer_model(num_layers=4, d_model=256, num_heads=4, ff_dim=1024, dropout_rate=0.1):
    """
    Creates a small Transformer model for move classification:
      Input shape: (64, 14)
      Output shape: (4096) with a softmax
    """
    inputs = Input(shape=(64, 14), name="board_input")

    # Project 14 features -> d_model dimension
    x = Dense(d_model)(inputs)

    # Simple trainable positional embeddings for the 64 "tokens" (one per square)
    positions = tf.range(start=0, limit=64, delta=1)
    pos_embed_layer = Embedding(input_dim=64, output_dim=d_model)
    pos_embedding = pos_embed_layer(positions)  # shape (64, d_model)
    # Broadcast these embeddings to the batch dimension
    x = x + tf.expand_dims(pos_embedding, axis=0)  # shape (batch_size, 64, d_model)

    # Stack multiple Transformer encoder blocks
    for _ in range(num_layers):
        x = transformer_encoder(x, d_model, num_heads, ff_dim, dropout_rate)

    # Pool the final sequence output. Options:
    #  1) GlobalAveragePooling1D
    #  2) Flatten
    #  3) Take the [CLS] token if implementing BERT‐style
    x = GlobalAveragePooling1D()(x)

    # Output layer: 4096 possible moves → softmax
    outputs = Dense(4096, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

###############################################################################
# 3. Main Training Logic
###############################################################################
def main():
    # -------------------------------------------------------------------------
    # 3a. Define your training parameters statically here
    # -------------------------------------------------------------------------
    tfrecord_dir = "data_tfrecords"      # Directory with *.tfrecord shards
    model_path   = ""                      # If set and valid, load an existing model
    save_path    = "./trained_chess_move_model.keras"  # Where to save final model
    batch_size   = 128
    epochs       = 10

    # -------------------------------------------------------------------------
    # 3b. Gather TFRecord files and count total number of samples
    # -------------------------------------------------------------------------
    tfrecord_files = sorted(glob.glob(os.path.join(tfrecord_dir, '*.tfrecord')))
    if not tfrecord_files:
        raise ValueError(f"No .tfrecord files found in {tfrecord_dir}")

    total_samples = count_tfrecord_samples(tfrecord_files)
    print(f"Found {total_samples} total samples in TFRecords.")
    steps_per_epoch = total_samples // batch_size
    print(f"Computed steps_per_epoch={steps_per_epoch} for batch_size={batch_size}.")

    # -------------------------------------------------------------------------
    # 3c. Build the Dataset (only one shard at a time in memory)
    # -------------------------------------------------------------------------
    # We'll read shards one-by-one (cycle_length=1) to minimize memory usage.
    file_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)

    # Interleave so we read one file at a time
    dataset = file_dataset.interleave(
        lambda fname: tf.data.TFRecordDataset(fname),
        cycle_length=1,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    # Decode, shuffle, batch, prefetch
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # -------------------------------------------------------------------------
    # 3d. Create or load the model
    # -------------------------------------------------------------------------
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating a fresh Transformer model...")
        model = create_transformer_model(
            num_layers=4,    # Increase as needed
            d_model=256,      # Increase for more capacity
            num_heads=4,     # Must divide d_model
            ff_dim=1024,      # Dimension of the feed‐forward layer
            dropout_rate=0.1
        )

    model.summary()

    # -------------------------------------------------------------------------
    # 3e. Train
    # -------------------------------------------------------------------------
    print(f"Training for {epochs} epoch(s). Reading entire dataset each epoch.")
    model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )

    # -------------------------------------------------------------------------
    # 3f. Save the trained model
    # -------------------------------------------------------------------------
    model.save(save_path)
    print(f"Model saved to {save_path}")

###############################################################################
# 4. Entry Point
###############################################################################
if __name__ == "__main__":
    main()