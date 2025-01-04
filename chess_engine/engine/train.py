import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from environment import ChessEnvironment
from utils import NUM_SQUARES, NUM_PROMOTION_OPTIONS, NUM_MOVES
from utils import encode_move, decode_action, board_to_planes, get_legal_moves_mask
from model import create_model

def compute_returns(rewards, gamma=0.99):
    returns = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + gamma * cumulative
        returns[i] = cumulative
    # Normalize returns
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    return returns

def policy_value_update(model, optimizer, states, masks, actions, returns):
    # Convert to tensors
    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    masks_tensor = tf.convert_to_tensor(masks, dtype=tf.float32)
    actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
    returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Forward pass
        policy_outputs, value_outputs = model({'input_board': states_tensor, 'mask_input': masks_tensor}, training=True)
        value_outputs = tf.squeeze(value_outputs, axis=1)

        # Compute advantages
        advantages = returns_tensor - value_outputs

        # Compute policy loss
        action_masks = tf.one_hot(actions_tensor, NUM_MOVES)
        log_probs = tf.math.log(policy_outputs + 1e-8)
        selected_log_probs = tf.reduce_sum(log_probs * action_masks, axis=1)
        policy_loss = -tf.reduce_mean(selected_log_probs * advantages)

        # Compute value loss
        value_loss = tf.reduce_mean(tf.square(advantages))

        # Total loss
        total_loss = policy_loss + 0.01 * value_loss  # Adjust value loss weight as needed

    # Compute gradients and apply them
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def train(episodes, save_path, existing_model_path=None):
    if existing_model_path:
        model = tf.keras.models.load_model(existing_model_path)
        optimizer = model.optimizer  # Use the optimizer from the loaded model
        print(f"Loaded model from {existing_model_path}")
    else:
        model = create_model()
        optimizer = model.optimizer  # Use the optimizer from the compiled model
        print("Initialized a new model.")
    opponent_model = tf.keras.models.load_model("/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/best_trained_model_cnn_opponent.keras")
    env = ChessEnvironment(agent_color=chess.WHITE)

    for episode in range(episodes):
        # Initialize the environment and storage arrays
        state = env.reset()
        done = False
        states, masks, actions, rewards = [], [], [], []

        while not done:
            # Agent's move
            state_input = board_to_planes(state)
            mask = get_legal_moves_mask(state)

            # Predict action probabilities and value
            state_input_expanded = np.expand_dims(state_input, axis=0)
            mask_input_expanded = np.expand_dims(mask, axis=0)
            model_inputs = {'input_board': state_input_expanded, 'mask_input': mask_input_expanded}
            policy_output, value_output = model.predict(model_inputs, verbose=0)
            action_probs = policy_output.ravel()
            state_value = value_output[0][0]

            # Choose an action
            action_index = np.random.choice(NUM_MOVES, p=action_probs)
            action_move = decode_action(action_index)

            # Agent performs the move
            next_state, reward, done, info = env.move(action_move)

            # Store experiences
            states.append(state_input)
            masks.append(mask)
            actions.append(action_index)
            rewards.append(reward)

            # Update the state
            state = env.board

            if not done:
                # Opponent's move
                if episode % 2 == 0:
                    env.opponent_random_move()
                else:
                    env.opponent_move(opponent_model)  # You may implement a random opponent or use a weak engine

                # Check if game is over after opponent's move
                if env.board.is_game_over():
                    result = env.board.result()
                    final_reward = env.get_game_result_reward(result)
                    rewards[-1] += final_reward  # Add final reward
                    done = True
                    info['result'] = env.get_result_string(result)

        # Compute returns and perform policy-value update
        returns = compute_returns(rewards, gamma=0.99)
        policy_value_update(model, optimizer, states, masks, actions, returns)

        # Calculate total reward and episode length
        total_reward = sum(rewards)
        episode_length = len(rewards)
        average_reward = total_reward / episode_length if episode_length > 0 else 0

        # Print episode info
        game_result = info.get('result', 'unknown')
        print(f"Episode {episode + 1}/{episodes} completed. Result: {game_result}, "
              f"Total Reward: {total_reward:.2f}, Episode Length: {episode_length}, "
              f"Average Reward per Step: {average_reward:.4f}")

        # Save the model periodically
        if (episode + 1) % 100 == 0:
            model.save(save_path, include_optimizer=True)
            print(f"Model saved to {save_path} at episode {episode + 1}")

    # Save the final model
    model.save(save_path, include_optimizer=True)
    print(f"Model saved to {save_path}")

# Start training
train(
    episodes=10000,
    save_path='best_trained_model_cnn_1.keras',
    existing_model_path='best_trained_model_cnn_1.keras'  # Set to None if training from scratch
)