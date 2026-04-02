from __future__ import annotations

from emotions_predictions.config import Settings
from emotions_predictions.data import load_and_process_data
from emotions_predictions.evaluation import evaluate_model, plot_history
from emotions_predictions.model import create_model
from emotions_predictions.seeds import set_random_seeds
from emotions_predictions.training import train_model


def run_pipeline(settings: Settings | None = None, *, show_plots: bool = True) -> None:
    """End-to-end train and evaluate using held-out test data."""
    cfg = settings or Settings()
    set_random_seeds(cfg.random_seed)

    processed = load_and_process_data(
        cfg.data_path,
        test_fraction=cfg.test_fraction,
        val_fraction=cfg.val_fraction,
        random_state=cfg.random_seed,
    )

    model = create_model(
        processed.n_features,
        len(processed.class_names),
        gru_units=cfg.gru_units,
        dropout_rate=cfg.dropout_rate,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trained, history = train_model(
        model,
        processed.X_train,
        processed.y_train,
        processed.X_val,
        processed.y_val,
        save_dir=cfg.output_dir,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        patience=cfg.early_stopping_patience,
        tensorboard_logdir=cfg.tensorboard_logdir,
    )

    history_path = cfg.output_dir / "training_history.png"
    plot_history(
        history,
        show=show_plots,
        save_path=history_path,
    )

    evaluate_model(
        trained,
        processed.X_test,
        processed.y_test,
        class_labels=processed.class_names,
        show_plots=show_plots,
        output_dir=cfg.output_dir,
    )
