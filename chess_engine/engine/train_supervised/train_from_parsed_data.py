import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# We assume environment.py is present and that your MCTS code expects a model with two outputs:
#  1) policy_head: shape (None, 4096), for the move probabilities
#  2) value_head:  shape (None, 1), for the board value (used by MCTS)

def create_chess_model():
    """
    Creates a simple feed-forward model that takes two inputs:
      1) 'input_board': shape (64,14)
      2) 'mask_input':  shape (4096,) - a mask for legal moves, typically used in inference

    Outputs two heads:
      - policy_head: shape (4096,) with a softmax activation (move distribution)
      - value_head:  shape (1,) with tanh activation

    For pure supervised training, we only train the policy_head (value_head is set to loss weight=0).
    """
    board_input = keras.Input(shape=(64, 14), name="input_board")
    mask_input = keras.Input(shape=(4096,), name="mask_input")  # MCTS usage expects to pass this

    # Flatten the board encoding
    x = layers.Flatten()(board_input)  # shape (None, 64*14) = (None, 896)

    # A couple of dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)

    # Policy head: 4096 possible moves
    policy_head = layers.Dense(4096, activation='softmax', name="policy_head")(x)

    # Value head: single scalar in [-1, 1]
    value_head = layers.Dense(1, activation='tanh', name="value_head")(x)

    model = keras.Model(
        inputs=[board_input, mask_input],
        outputs=[policy_head, value_head]
    )
    return model


def train_supervised(data_path="data_prepared",
                     batch_size=64,
                     epochs=5,
                     lr=1e-3):
    """
    Loads the data from data_path (train.npz, val.npz),
    trains a two-head model (policy + value), but only
    uses the policy head for supervised next-move prediction.
    """
    # 1) Load NumPy data
    train_data = np.load(os.path.join(data_path, "train.npz"))
    val_data = np.load(os.path.join(data_path, "val.npz"))

    X_train = train_data["X_train"]  # shape (N, 64, 14)
    Y_train = train_data["Y_train"]  # shape (N,) move index
    X_val = val_data["X_val"]
    Y_val = val_data["Y_val"]

    # 2) Create dummy mask data for training (shape (N, 4096)), since the model has two inputs
    #    At inference time or during MCTS, we pass a real move mask.
    train_mask = np.ones((len(X_train), 4096), dtype=np.float32)
    val_mask = np.ones((len(X_val), 4096), dtype=np.float32)

    # 3) Build the model
    model = create_chess_model()

    # We'll only train the policy head. The value head is present for MCTS usage,
    # but we have no labeled values in plain supervised PGN data. So we set its weight to 0.
    losses = {
        "policy_head": "sparse_categorical_crossentropy",  # supervised label = next move index
        "value_head": "mse"  # dummy
    }
    loss_weights = {
        "policy_head": 1.0,
        "value_head": 0.0
    }

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=losses,
        loss_weights=loss_weights,
        metrics={"policy_head": "accuracy"}
    )

    model.summary()

    # 4) Train
    # Note that we provide zeros for value_head, which is effectively ignored.
    model.fit(
        x={"input_board": X_train, "mask_input": train_mask},
        y={
            "policy_head": Y_train,
            "value_head": np.zeros((len(X_train), 1), dtype=np.float32)
        },
        validation_data=(
            {"input_board": X_val, "mask_input": val_mask},
            {
                "policy_head": Y_val,
                "value_head": np.zeros((len(X_val), 1), dtype=np.float32)
            }
        ),
        batch_size=batch_size,
        epochs=epochs
    )

    # 5) Save
    model_save_path = os.path.join(data_path, "chess_model.keras")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    # Example usage of training
    train_supervised(
        data_path="data_prepared",
        batch_size=8,
        epochs=5,
        lr=1e-3
    )