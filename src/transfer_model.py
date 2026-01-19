"""
transfer_model.py
------------------
Transfer learning using ResNet50 with two-phase training:
1) Feature extraction (frozen base)
2) Fine-tuning (top layers unfrozen)
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data_loader import get_data_generators


def build_transfer_model(input_shape=(224, 224, 3)):
    # Load pre-trained ResNet50
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze the convolutional base
    base_model.trainable = False

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model, base_model


def main():
    # Load data
    train_gen, val_gen, test_gen = get_data_generators()

    # Build model
    model, base_model = build_transfer_model()

    # Compile (Phase 1)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(
            "models/resnet50_phase1.h5",
            save_best_only=True
        )
    ]

    print("\n🔹 Phase 1: Training classification head")
    model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # ---------------- PHASE 2: FINE-TUNING ---------------- #

    print("\n🔹 Phase 2: Fine-tuning top layers of ResNet50")

    # Unfreeze top layers of the base model
    for layer in base_model.layers[-10:]:
    	layer.trainable = True



    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_finetune = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(
            "models/resnet50_finetuned.h5",
            save_best_only=True
        )
    ]

    model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        callbacks=callbacks_finetune
    )

    # Final evaluation
    test_loss, test_acc = model.evaluate(test_gen)
    print("\nResNet50 Test Accuracy:", round(test_acc * 100, 2), "%")


if __name__ == "__main__":
    main()