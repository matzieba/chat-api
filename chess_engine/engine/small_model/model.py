import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2  # <--- add this

NUM_MOVES = 20480

def create_small_chess_model(
    board_shape=(8, 8, 14),
    num_moves=NUM_MOVES,
    num_filters=64,
    num_res_blocks=4,
    learning_rate=1e-3,
    l2_reg=1e-5       # <--- add a parameter for L2 strength
):
    input_board = layers.Input(shape=board_shape, name="input_board")
    mask_input = layers.Input(shape=(num_moves,), name="mask_input")

    # Initial Convolution with L2 regularization
    x = layers.Conv2D(
        filters=num_filters,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_regularizer=l2(l2_reg)   # <--- L2 regularization
    )(input_board)

    # Residual Blocks
    for _ in range(num_res_blocks):
        residual = x
        x = layers.Conv2D(
            num_filters, 3, padding='same',
            use_bias=False,
            kernel_regularizer=l2(l2_reg)  # <--- L2 reg
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(
            num_filters, 3, padding='same',
            use_bias=False,
            kernel_regularizer=l2(l2_reg)  # <--- L2 reg
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)

    # Policy Head
    policy_conv = layers.Conv2D(
        filters=2,
        kernel_size=1,
        activation='relu',
        kernel_regularizer=l2(l2_reg)   # <--- L2 reg
    )(x)
    policy_flat = layers.Flatten()(policy_conv)
    logits = layers.Dense(
        num_moves,
        name='logits',
        kernel_regularizer=l2(l2_reg)   # <--- L2 reg
    )(policy_flat)

    negative_inf = -1e9
    masked_logits = logits + (1.0 - mask_input) * negative_inf
    policy_output = layers.Softmax(name='policy_output')(masked_logits)

    # Value Head
    value_conv = layers.Conv2D(
        filters=1,
        kernel_size=1,
        activation='relu',
        kernel_regularizer=l2(l2_reg)   # <--- L2 reg
    )(x)
    value_flat = layers.Flatten()(value_conv)
    value_dense = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=l2(l2_reg)   # <--- L2 reg
    )(value_flat)
    value_output = layers.Dense(1, activation='tanh', name='value_output')(value_dense)

    # Define Model
    model = Model(inputs=[input_board, mask_input], outputs=[policy_output, value_output])

    # Compile with 'categorical_crossentropy' for the policy head
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            'policy_output': 'categorical_crossentropy',   # <--- changed here
            'value_output': 'mean_squared_error'
        },
        loss_weights={
            'policy_output': 1.0,
            'value_output': 1.0
        },
        metrics={
            'policy_output': [
                tf.keras.metrics.CategoricalAccuracy(name='policy_accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='policy_top5_accuracy')
            ]
        }
    )

    return model