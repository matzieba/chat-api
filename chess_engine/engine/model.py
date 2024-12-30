import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

from chess_engine.engine.utils import NUM_MOVES


def create_model():
    # Input layer: Sequence of token IDs
    input_seq = tf.keras.Input(shape=(64,), dtype=tf.int32, name='input_seq')
    # Positional encoding
    embedding_dim = 64
    vocab_size = 13  # 12 pieces + empty square

    # Token embedding
    token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='token_embedding')(input_seq)

    # Add positional encoding
    position_indices = tf.range(64)
    position_embedding = layers.Embedding(input_dim=64, output_dim=embedding_dim, name='position_embedding')(position_indices)
    x = token_embedding + position_embedding

    # Transformer Encoder
    num_layers = 4
    num_heads = 8
    ffn_dim = 256
    dropout_rate = 0.1

    for _ in range(num_layers):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward network
        ffn = layers.Dense(ffn_dim, activation='relu')(x)
        ffn_output = layers.Dense(embedding_dim)(ffn)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Flatten the output
    x = layers.Flatten()(x)

    # Final dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output layer (logits)
    logits = layers.Dense(NUM_MOVES, name='logits')(x)

    # Masking illegal moves
    mask_input = tf.keras.Input(shape=(NUM_MOVES,), name='mask_input')
    negative_inf = -1e9
    masked_logits = logits + (1.0 - mask_input) * negative_inf

    # Apply softmax to masked logits
    output = tf.keras.layers.Softmax(name='output')(masked_logits)

    # Create the model
    model = tf.keras.Model(inputs=[input_seq, mask_input], outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy')
    return model

def policy_gradient(model, optimizer, states, actions, rewards, gamma=0.99):
    # Convert actions to tensor
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    # Compute discounted rewards
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    cumulative_reward = 0
    for t in reversed(range(len(rewards))):
        cumulative_reward = rewards[t] + gamma * cumulative_reward
        discounted_rewards[t] = cumulative_reward

    # Normalize rewards
    mean_reward = np.mean(discounted_rewards)
    std_reward = np.std(discounted_rewards) + 1e-8  # Avoid division by zero
    discounted_rewards = (discounted_rewards - mean_reward) / std_reward

    with tf.GradientTape() as tape:
        # Forward pass
        logits = model(states, training=True)

        # Compute log probabilities
        action_masks = tf.one_hot(actions, NUM_MOVES)
        log_probabilities = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)

        # Compute loss
        loss = -tf.reduce_mean(discounted_rewards * log_probabilities)

    # Compute gradients
    grads = tape.gradient(loss, model.trainable_variables)
    # Apply gradients
    optimizer.apply_gradients(zip(grads, model.trainable_variables))