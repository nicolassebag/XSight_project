import os

import numpy as np
import tensorflow as tf
import keras
from typing import Literal, Tuple

# Display
from IPython.display import display, Image
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import joblib

from XSight.params import PATHO_COLUMNS


def make_gradcam_heatmap(X_img,
                         X_tab,
                         model,
                         last_conv_layer_name,
                         pred_index=None,
                         threshold = 0.7):

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    # threshold (float): Seuil entre 0 et 1. Seules les valeurs supérieures seront conservées.
    #                   0.4 = garde les 60% des zones les plus importantes
    #                   0.6 = garde les 40% des zones les plus importantes
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model([X_img, X_tab])
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    heatmap_thresholded = tf.where(heatmap > threshold, heatmap, tf.zeros_like(heatmap))
    return heatmap_thresholded.numpy()

def apply_heatmap(img_array, heatmap, alpha=0.4):
    """Fonction helper pour appliquer la heatmap à l'image originale"""
    # Convertit la heatmap en couleurs
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Redimensionne et superpose
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = np.array(jet_heatmap)

    # Superposition
    superimposed_img = jet_heatmap * alpha + img_array
    return keras.utils.array_to_img(superimposed_img)

#########################################################
##################### TO DELETE ? #######################
#########################################################

def save_and_display_gradcam(image_path, heatmap, alpha=0.4):
    # Load the original image
    img = Image.open(image_path)
    img = keras.utils.img_to_array(img)

    # Convert to grey scale
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # # Save the superimposed image
    # superimposed_img.save(cam_path)

    # # Display Grad CAM
    # display(Image.open(cam_path))
    display(superimposed_img)
