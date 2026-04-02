from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize


@dataclass(frozen=True)
class ProcessedData:
    """Train / validation / test tensors with aligned one-hot labels."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    n_features: int
    label_encoder: LabelEncoder


def load_and_process_data(
    file_path: str | Path,
    *,
    test_fraction: float,
    val_fraction: float,
    random_state: int,
) -> ProcessedData:
    """
    Load CSV, stratify into train / val / test, and one-hot encode labels with a fixed class order.

    The label column must be named ``label``. All other columns are treated as features.
    """
    path = Path(file_path)
    if not path.is_file():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    data = pd.read_csv(path)
    if "label" not in data.columns:
        raise ValueError("Missing 'label' column in the dataset")

    y_raw = data["label"].astype(str)
    X = data.drop(columns=["label"])

    le = LabelEncoder()
    y_int = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X,
        y_int,
        test_size=test_fraction,
        stratify=y_int,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv,
        y_tv,
        test_size=val_fraction,
        stratify=y_tv,
        random_state=random_state,
    )

    def _reshape_features(frame: pd.DataFrame) -> np.ndarray:
        arr = frame.to_numpy(dtype=np.float32)
        return arr.reshape(arr.shape[0], arr.shape[1], 1)

    X_train_r = _reshape_features(X_train)
    X_val_r = _reshape_features(X_val)
    X_test_r = _reshape_features(X_test)

    classes = np.arange(n_classes)
    y_train_oh = label_binarize(y_train, classes=classes).astype(np.float32)
    y_val_oh = label_binarize(y_val, classes=classes).astype(np.float32)
    y_test_oh = label_binarize(y_test, classes=classes).astype(np.float32)

    return ProcessedData(
        X_train=X_train_r,
        X_val=X_val_r,
        X_test=X_test_r,
        y_train=y_train_oh,
        y_val=y_val_oh,
        y_test=y_test_oh,
        class_names=list(le.classes_),
        n_features=X_train.shape[1],
        label_encoder=le,
    )
