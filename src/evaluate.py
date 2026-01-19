"""
evaluate.py
------------
Evaluation metrics and confusion matrix for the trained model.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import tensorflow as tf

from src.data_loader import get_data_generators


def main():
    # Load test data
    _, _, test_gen = get_data_generators()

    # Load trained model
    model = tf.keras.models.load_model("models/resnet50_finetuned.h5")

    # Predict
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = (y_pred_probs > 0.5).astype(int).ravel()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Cat", "Dog"]))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Cat", "Dog"]
    )

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - ResNet50")
    plt.savefig("gradcam/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
