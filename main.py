"""Backward-compatible entry point: run the default training and evaluation pipeline."""

from emotions_predictions.config import Settings
from emotions_predictions.pipeline import run_pipeline


def main() -> None:
    run_pipeline(Settings())


if __name__ == "__main__":
    main()
