"""
data_loader.py
----------------
Handles image loading, preprocessing, and data augmentation.
Used by both baseline CNN and transfer learning models.
"""

from tensorflow.keras.applications.resnet50 import preprocess_input

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_generators(
    data_dir="data",
    img_size=(224, 224),
    batch_size=32
):
    """
    Creates train, validation, and test data generators.

    Parameters:
    - data_dir: base dataset directory
    - img_size: target image size
    - batch_size: number of images per batch

    Returns:
    - train_gen, val_gen, test_gen
    """

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Data augmentation (TRAIN ONLY)
    train_datagen = ImageDataGenerator(
    	preprocessing_function=preprocess_input,
    	rotation_range=20,
    	width_shift_range=0.1,
    	height_shift_range=0.1,
    	zoom_range=0.2,
    	horizontal_flip=True
    )


    # Validation & Test (NO augmentation)
    val_test_datagen = ImageDataGenerator(
    	preprocessing_function=preprocess_input
    )


    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_generator, val_generator, test_generator
