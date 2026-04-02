from __future__ import annotations

import csv
from pathlib import Path

import pytest


@pytest.fixture()
def tiny_eeg_csv(tmp_path: Path) -> Path:
    """Minimal CSV: 3 classes, enough rows per class for stratified splits."""
    path = tmp_path / "eeg.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["f1", "f2", "f3", "label"])
        w.writeheader()
        for label in ("negative", "neutral", "positive"):
            for i in range(20):
                w.writerow(
                    {
                        "f1": str(float(i)),
                        "f2": str(float(i) * 0.1),
                        "f3": str(float(label == "neutral")),
                        "label": label,
                    }
                )
    return path
