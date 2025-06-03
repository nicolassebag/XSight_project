##### IMPORTS #####

from typing import Tuple
import pandas as pd
import numpy as np
import os
from PIL import Image

import tensorflow as tf


##### FUNCTIONS #####

def load_data(filepath: str) -> pd.DataFrame:
    """Load a csv file into a Dataframe."""
    return pd.read_csv(filepath)

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not needed for model training."""
    columns_to_drop = [
        'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 'Follow-up #', 'Patient ID'
    ]
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

def encode_labels(df: pd.DataFrame, label_column: str = 'Finding Labels') -> pd.DataFrame:
    """OneHotEncode the labels and drop the original column."""
    if label_column in df.columns:
        dummies = df[label_column].str.get_dummies(sep='|')
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[label_column])
    return df

                ############################################################

def preprocess_1(filepath: str) -> pd.DataFrame:
    """Complete preprocessing the pipeline."""
    df = load_data(filepath)
    df = drop_unnecessary_columns(df)
    df = encode_labels(df)
    return df


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
