from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emotions_predictions.data import load_and_process_data


def test_load_and_process_stratified_shapes(tiny_eeg_csv: Path) -> None:
    processed = load_and_process_data(
        tiny_eeg_csv,
        test_fraction=0.15,
        val_fraction=0.2,
        random_state=42,
    )
    n_classes = 3
    assert processed.X_train.shape[-1] == 1
    assert processed.n_features == 3
    assert processed.y_train.shape[1] == n_classes
    assert processed.y_val.shape[1] == n_classes
    assert processed.y_test.shape[1] == n_classes

    assert processed.X_train.shape[0] + processed.X_val.shape[0] + processed.X_test.shape[0] == 60

    # One-hot rows sum to 1
    assert np.allclose(processed.y_train.sum(axis=1), 1.0)
    assert np.allclose(processed.y_val.sum(axis=1), 1.0)
    assert np.allclose(processed.y_test.sum(axis=1), 1.0)

    assert processed.class_names == ["negative", "neutral", "positive"]


def test_missing_label_column(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="label"):
        load_and_process_data(bad, test_fraction=0.2, val_fraction=0.2, random_state=0)
