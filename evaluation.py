import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.utils import to_categorical
import numpy as np
import seaborn as sns

def plot_history(history, metric='accuracy'):
    """
    Plots the training history for a given metric and its validation metric.

    Parameters:
    history: Training history object from a Keras model.
    metric (str): Metric to be plotted (default: 'accuracy').
    """
    metric_values = history.history[metric]
    val_metric_values = history.history['val_' + metric]
    epochs = range(1, len(metric_values) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metric_values, 'b-', label=f'Training {metric}')
    plt.plot(epochs, val_metric_values, 'r-', label=f'Validation {metric}')
    plt.title(f'Training and Validation {metric.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, class_labels=None):
    """
    Evaluates the model on the test data and prints out the confusion matrix, classification report, and ROC curves.

    Parameters:
    model: Trained Keras model.
    X_test: Test features.
    y_test: Test labels (should be one-hot encoded for multi-class).
    class_labels (list): List of class labels (optional).
    """
    # Evaluate the model
    model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Test Accuracy: {model_acc * 100:.3f}%")

    # Predict probabilities for each class
    y_pred_probs = model.predict(X_test)

    # Compute ROC curve and ROC area for each class
    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_labels[i] if class_labels else i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    # Confusion Matrix and Classification Report
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    clr = classification_report(y_true, y_pred, target_names=class_labels)

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:\n----------------------\n", clr)
