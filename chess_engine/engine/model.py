import numpy as np

import tensorflow as tf
from keras.src.optimizers import Adam
from tensorflow.keras import layers
from chess_engine.engine.utils import NUM_MOVES


def create_model():
    # Input layer: Board representation as 8x8x14 tensor
    input_board = tf.keras.Input(shape=(8, 8, 14), dtype=tf.float32, name='input_board')

    # Initial convolutional layer
    x = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(input_board)

    # Residual blocks
    for _ in range(20):  # Increase depth as necessary
        residual = x
        x = layers.Conv2D(filters=256, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters=256, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)

    # Policy head
    policy_conv = layers.Conv2D(filters=2, kernel_size=1, activation='relu')(x)
    policy_flat = layers.Flatten()(policy_conv)
    logits = layers.Dense(NUM_MOVES, name='logits')(policy_flat)

    # Masking illegal moves
    mask_input = tf.keras.Input(shape=(NUM_MOVES,), name='mask_input')
    negative_inf = -1e9
    masked_logits = logits + (1.0 - mask_input) * negative_inf

    # Softmax output
    policy_output = layers.Softmax(name='policy_output')(masked_logits)

    # Value head
    value_conv = layers.Conv2D(filters=1, kernel_size=1, activation='relu')(x)
    value_flat = layers.Flatten()(value_conv)
    value_dense = layers.Dense(256, activation='relu')(value_flat)
    value_output = layers.Dense(1, activation='tanh', name='value_output')(value_dense)

    # Create model
    model = tf.keras.Model(inputs=[input_board, mask_input], outputs=[policy_output, value_output])
    model.compile(
        optimizer=Adam(learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3,
            decay_steps=100000
        )),
        loss={
            'policy_output': 'sparse_categorical_crossentropy',
            'value_output': 'mean_squared_error'
        },
        loss_weights={
            'policy_output': 1.0,
            'value_output': 1.0  # Adjust weight as needed
        },
        metrics={
            'policy_output': [
                tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
            ],
            # Optionally, add metrics for 'value_output' if desired
        }
    )

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