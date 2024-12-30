import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from environment import ChessEnvironment
from utils import NUM_SQUARES, NUM_PROMOTION_OPTIONS, NUM_MOVES
from utils import encode_move, decode_action, convert_board_to_sequence, get_legal_moves_mask
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

def policy_gradient_update(model, optimizer, states, masks, actions, returns):
    actions = np.array(actions)
    returns = np.array(returns)
    with tf.GradientTape() as tape:
        # Get probabilities for all actions given the states
        logits = model({'input_seq': states, 'mask_input': masks}, training=True)
        log_probs = tf.math.log(logits + 1e-8)  # Add epsilon to avoid log(0)

        # Gather the log probabilities corresponding to the actions taken
        actions_indices = tf.stack([tf.range(len(actions)), actions], axis=1)
        selected_log_probs = tf.gather_nd(log_probs, actions_indices)

        # Compute the policy loss
        loss = -tf.reduce_mean(selected_log_probs * returns)

    # Compute gradients
    grads = tape.gradient(loss, model.trainable_variables)
    # Apply gradients
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

    env = ChessEnvironment(agent_color=chess.WHITE)
    opponent_model = tf.keras.models.clone_model(model)
    opponent_model.set_weights(model.get_weights())
    opponent_model.compile(optimizer=Adam(learning_rate=1e-4))  # Recompile to initialize optimizer

    for episode in range(episodes):
        # Initialize the environment and storage arrays
        state = env.reset()
        done = False
        states, masks, actions, rewards = [], [], [], []

        while not done:
            # Agent's move
            state_input = convert_board_to_sequence(state)
            mask = get_legal_moves_mask(state)

            # Predict action probabilities
            state_input_expanded = np.expand_dims(state_input, axis=0)
            mask_input_expanded = np.expand_dims(mask, axis=0)
            model_inputs = {'input_seq': state_input_expanded, 'mask_input': mask_input_expanded}
            action_probs = model.predict(model_inputs, verbose=0).ravel()

            # Choose an action
            action_index = np.random.choice(NUM_MOVES, p=action_probs)
            action_move = decode_action(action_index)

            # Agent performs the move
            next_state, reward, done, info = env.move(action_move)

            # Check if the game is over after the agent's move
            if not done:
                # Opponent's move
                env.opponent_move(opponent_model)
                # Check if the game is over after the opponent's move
                if env.board.is_game_over():
                    result = env.board.result()
                    final_reward = env.get_game_result_reward(result)
                    reward += final_reward  # Add final reward to agent's reward
                    done = True
            else:
                # The game ended after the agent's move
                pass

            # Store experiences
            states.append(state_input)
            masks.append(mask)
            actions.append(action_index)
            rewards.append(reward)

            # Update the state
            state = env.board
            if done:
                break  # Exit the loop when the game is over

        # Compute returns and perform policy gradient update
        returns = compute_returns(rewards, gamma=0.99)
        states_array = np.array(states)
        masks_array = np.array(masks)
        actions_array = np.array(actions)
        returns_array = np.array(returns)
        policy_gradient_update(model, optimizer, states_array, masks_array, actions_array, returns_array)

        # Calculate total reward and episode length
        total_reward = sum(rewards)
        episode_length = len(rewards)
        average_reward = total_reward / episode_length if episode_length > 0 else 0

        # Print episode info
        print(f"Episode {episode + 1}/{episodes} completed. "
              f"Total Reward: {total_reward:.2f}, "
              f"Episode Length: {episode_length}, "
              f"Average Reward per Step: {average_reward:.4f}")

        # Save the entire model periodically (e.g., every 100 episodes)
        if (episode + 1) % 100 == 0:
            model.save(save_path, include_optimizer=True)
            print(f"Model saved to {save_path} at episode {episode + 1}")

    # Save the final model
    model.save(save_path, include_optimizer=True)
    print(f"Model saved to {save_path}")

# Start training
train(episodes=10, save_path="/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/models/best_trained_transformer_model.keras", existing_model_path="/Users/mateuszzieba/Desktop/dev/cvt/chat-api/chat-api/chess_engine/engine/models/best_trained_transformer_model.keras")