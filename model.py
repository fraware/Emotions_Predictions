import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(input_shape, num_classes=3, gru_units=256, dropout_rate=0.2, learning_rate=0.001):
    """
    Creates a GRU-based neural network model for classification.

    Parameters:
    input_shape (tuple): Shape of the input data.
    num_classes (int): Number of classes for classification (default is 3).
    gru_units (int): Number of units in the GRU layer (default is 256).
    dropout_rate (float): Dropout rate for regularization (default is 0.2).
    learning_rate (float): Learning rate for the optimizer (default is 0.001).

    Returns:
    tf.keras.Model: A compiled Keras model.
    """
    inputs = Input(shape=(input_shape, 1))
    gru = GRU(gru_units, return_sequences=True)(inputs)
    flat = Flatten()(gru)
    dropout = Dropout(dropout_rate)(flat)
    outputs = Dense(num_classes, activation='softmax')(dropout)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
