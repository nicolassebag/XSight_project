import pandas as pd
import numpy as np
import os
from PIL import Image

import tensorflow as tf



def resize_all_images(df, img_dir, target_phys_size=(256, 256), final_size=(64, 64)):
    """
    Redimensionne et normalise toutes les images d'un dossier en fonction de leur pixel spacing et leur size,
    en se basant sur les data d'un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'Image Index', 'OriginalImage[Width',
                           'OriginalImagePixelSpacing[x', 'Height]', 'y]'.
        img_dir (str): Chemin vers le dossier contenant les images.
        target_phys_size (tuple): Taille physique cible (en mm).
        final_size (tuple): Taille finale pour le modèle (en pixels).

    Returns:
        list of tf.Tensor: Liste des images prétraitées.
    """
    processed_images = []

    for idx, row in df.iterrows():
        img_filename = row['Image Index']
        img_path = os.path.join(img_dir, img_filename)

        # Chargement et forcage de l’image en grayscale
        image = tf.io.read_file(img_path)
        image = tf.image.decode_image(image, channels=1)
        image.set_shape([None, None, 1])

        # Taille physique cible (en pixels)
        pixel_spacing_x = row['OriginalImagePixelSpacing[x']
        pixel_spacing_y = row['y]']

        new_width = int(target_phys_size[0] / pixel_spacing_x)
        new_height = int(target_phys_size[1] / pixel_spacing_y)

        # Redimensionnement à taille physique homogène
        image = tf.image.resize(image, (new_height, new_width))

        # Resize final pour le modèle
        image = tf.image.resize(image, final_size)

        # Normalisation des pixels
        image = image / 255.0

        processed_images.append(image)

    return processed_images
