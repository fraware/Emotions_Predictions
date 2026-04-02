from __future__ import annotations

import numpy as np
import pytest

from emotions_predictions.model import create_model


@pytest.mark.slow
def test_model_compile_and_fit_one_step() -> None:
    n_samples, n_timesteps, n_classes = 8, 5, 3
    x = np.random.randn(n_samples, n_timesteps, 1).astype(np.float32)
    y = np.eye(n_classes, dtype=np.float32)[np.random.randint(0, n_classes, size=n_samples)]

    model = create_model(n_timesteps, n_classes, gru_units=16, dropout_rate=0.1)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x, y, batch_size=4, epochs=1, verbose=0)
    preds = model.predict(x, verbose=0)
    assert preds.shape == (n_samples, n_classes)
