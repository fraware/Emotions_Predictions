from __future__ import annotations

from pathlib import Path

import typer

from emotions_predictions.config import Settings
from emotions_predictions.pipeline import run_pipeline

app = typer.Typer(no_args_is_help=True, help="EEG emotion recognition — train and evaluate.")


@app.command("train")
def train_cmd(
    data_path: Path | None = typer.Option(
        None,
        "--data-path",
        help=(
            "CSV with feature columns and a `label` column "
            "(default: env or EEG-emotions.csv in cwd)."
        ),
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory for checkpoints and figures.",
    ),
    epochs: int | None = typer.Option(None, "--epochs", help="Training epochs."),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Batch size."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    no_plots: bool = typer.Option(
        False,
        "--no-plots",
        help="Do not open interactive matplotlib windows.",
    ),
) -> None:
    """Train the model and evaluate on the held-out test split."""
    cfg = Settings()
    if data_path is not None:
        cfg = cfg.model_copy(update={"data_path": data_path})
    if output_dir is not None:
        cfg = cfg.model_copy(update={"output_dir": output_dir})
    if epochs is not None:
        cfg = cfg.model_copy(update={"epochs": epochs})
    if batch_size is not None:
        cfg = cfg.model_copy(update={"batch_size": batch_size})
    if seed is not None:
        cfg = cfg.model_copy(update={"random_seed": seed})
    run_pipeline(cfg, show_plots=not no_plots)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
