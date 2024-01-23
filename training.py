from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

def train_model(model, x_train, y_train, x_test, y_test, save_path, epochs, learning_rate=0.001, batch_size=32, patience=10):
    """
    Trains a given Keras model with the provided training and validation data.

    Parameters:
    model: Keras model to be trained.
    x_train, y_train: Training data and labels.
    x_test, y_test: Validation data and labels.
    save_path (str): Path to save the best model.
    epochs (int): Number of epochs to train.
    learning_rate (float): Initial learning rate for training. Default is 0.001.
    batch_size (int): Batch size for training. Default is 32.
    patience (int): Patience for EarlyStopping. Default is 10.

    Returns:
    Tuple of the best Keras model after training and the training history.
    """
    # Callbacks
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint(f'{save_path}_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    lr_schedule = LearningRateScheduler(lambda epoch: learning_rate * np.exp(-epoch / 10.))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Model Training
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[es, mc, lr_schedule])

    # Load the best model
    best_model = load_model(f'{save_path}_best_model.h5')

    return best_model, history
