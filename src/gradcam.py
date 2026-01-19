"""
gradcam.py
----------
Generates Grad-CAM heatmaps for model interpretability.
"""

import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def main():
    model = tf.keras.models.load_model("models/resnet50_finetuned.h5")

    img_path = "data/test/dogs/" + os.listdir("data/test/dogs")[0]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name="conv5_block3_out"
    )

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    os.makedirs("gradcam", exist_ok=True)
    cv2.imwrite("gradcam/gradcam_sample.png", superimposed)

    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Grad-CAM Visualization")
    plt.show()


if __name__ == "__main__":
    main()
