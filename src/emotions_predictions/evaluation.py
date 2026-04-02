from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def plot_history(
    history: Any,
    *,
    metric: str = "accuracy",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    metric_values = history.history[metric]
    val_metric_values = history.history["val_" + metric]
    epochs = range(1, len(metric_values) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, metric_values, "b-", label=f"Training {metric}")
    axes[0].plot(epochs, val_metric_values, "r-", label=f"Validation {metric}")
    axes[0].set_title(f"Training and Validation {metric.capitalize()}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(metric.capitalize())
    axes[0].legend()

    axes[1].plot(epochs, history.history["loss"], "b-", label="Training Loss")
    axes[1].plot(epochs, history.history["val_loss"], "r-", label="Validation Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    class_labels: list[str] | None = None,
    show_plots: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate on held-out test data; optionally save figures to ``output_dir``.
    Returns a dict of scalar metrics and report strings for programmatic use.
    """
    y_true = np.argmax(y_test, axis=1)
    loss, model_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    names = class_labels if class_labels is not None else [str(i) for i in range(y_test.shape[1])]

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=names, zero_division=0)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {model_acc * 100:.3f}%")
    print(f"Balanced accuracy: {balanced_acc * 100:.3f}%")
    print(f"Macro F1: {macro_f1:.4f}")

    n_classes = y_test.shape[1]
    fpr: dict[int, np.ndarray] = {}
    tpr: dict[int, np.ndarray] = {}
    roc_auc: dict[int, float] = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
        roc_auc[i] = float(auc(fpr[i], tpr[i]))

    try:
        macro_auc_ovr = float(
            roc_auc_score(y_test, y_pred_probs, average="macro", multi_class="ovr")
        )
    except ValueError:
        macro_auc_ovr = float("nan")

    out: Path | None = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
    colors = ["blue", "red", "green", "purple", "orange", "brown"]
    for i, color in zip(range(n_classes), cycle(colors), strict=False):
        ax_roc.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"Class {names[i]} (AUC = {roc_auc[i]:.2f})",
        )
    ax_roc.plot([0, 1], [0, 1], "k--", lw=2)
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.05)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Multi-class ROC (one-vs-rest)")
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    if out is not None:
        fig_roc.savefig(out / "roc.png", dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig_roc)

    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        xticklabels=names,
        yticklabels=names,
        ax=ax_cm,
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    plt.tight_layout()
    if out is not None:
        fig_cm.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig_cm)

    print("Classification Report:\n----------------------\n", report)

    return {
        "loss": float(loss),
        "accuracy": float(model_acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "macro_auc_ovr": macro_auc_ovr,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
