import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

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
    Evaluates the model on the test data and prints out the confusion matrix and classification report.

    Parameters:
    model: Trained Keras model.
    X_test: Test features.
    y_test: Test labels.
    class_labels (list): List of class labels (optional).
    """
    model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Test Accuracy: {model_acc * 100:.3f}%")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_test_labels, y_pred)
    clr = classification_report(y_test_labels, y_pred, target_names=class_labels)

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:\n----------------------\n", clr)
