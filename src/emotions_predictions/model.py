from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Dropout, Flatten, Input


def create_model(
    n_timesteps: int,
    num_classes: int,
    *,
    gru_units: int = 256,
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """
    Build an uncompiled GRU classifier. Call ``compile`` in the training module.
    """
    inputs = Input(shape=(n_timesteps, 1))
    x = GRU(gru_units, return_sequences=True)(inputs)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
