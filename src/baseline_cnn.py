"""
baseline_cnn.py
----------------
A simple CNN trained from scratch.
Used as a baseline to compare against transfer learning.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

from src.data_loader import get_data_generators


def build_baseline_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    return model


def main():
    # Load data
    train_gen, val_gen, test_gen = get_data_generators()

    # Build model
    model = build_baseline_model()

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train model (few epochs only)
    history = model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen
    )

    # Evaluate on test data
    test_loss, test_acc = model.evaluate(test_gen)

    print("\nBaseline CNN Test Accuracy:", round(test_acc * 100, 2), "%")

    # Save model
    model.save("models/baseline_cnn.h5")
    print("Baseline model saved to models/baseline_cnn.h5")


if __name__ == "__main__":
    main()
