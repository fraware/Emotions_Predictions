from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    save_dir: str | Path,
    epochs: int,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    patience: int = 10,
    tensorboard_logdir: str | Path | None = None,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train with validation on ``x_val`` / ``y_val``.

    Persists the best checkpoint by validation accuracy.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "best_model.keras"

    callbacks: list[tf.keras.callbacks.Callback] = [
        EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            save_best_only=True,
        ),
        LearningRateScheduler(lambda epoch: learning_rate * np.exp(-epoch / 10.0)),
    ]
    if tensorboard_logdir is not None:
        tb_path = Path(tensorboard_logdir)
        tb_path.mkdir(parents=True, exist_ok=True)
        callbacks.append(TensorBoard(log_dir=str(tb_path), histogram_freq=0))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    best_model = load_model(str(checkpoint_path))
    return best_model, history
