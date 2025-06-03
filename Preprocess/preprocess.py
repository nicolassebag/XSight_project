##### IMPORTS #####

from typing import Tuple
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

from data import fetch_images_to_memory

import tensorflow as tf




### VARIABLE GLOBAL ###

gcp_project= "xsight-project-le-wagon"
bucket_name= "cxr8_images_bucket"
prefix= "all_images/"



##### FUNCTIONS PREPROC DATA #####

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



##### --------- FUNCTIONS PREPROC IMAGE --------- #####


def resize_all_images(df, image_list, final_size=(64, 64)):
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


    # -- Appel Retour un dict avec le nom de l'image et ses datas -- #

    """
    Fetch images from GCS into memory as BytesIO objects
    Returns: {image_name: tf.io.decode_image(img_bytes) object}
    """
    images = fetch_images_to_memory(
                                    gcp_project= gcp_project,
                                    bucket_name= bucket_name,
                                    prefix= prefix,
                                    image_list= image_list
                                )


    processed_images = []


    for img_name, img in images.items():

        # Recuperation de l'index de la photo dans data_entry
        row = df[df["Image Index"] == img_name]

        # Taille physique cible (en pixels)
        pixel_spacing_x = row['OriginalImagePixelSpacing[x']
        pixel_spacing_y = row['y]']

        target_phys_size=(256, 256)

        new_width = int(target_phys_size[0] / pixel_spacing_x)
        new_height = int(target_phys_size[1] / pixel_spacing_y)

        # Redimensionnement à taille physique homogène
        img = tf.image.resize(img, (new_height, new_width))

        # Resize final pour le modèle
        img = tf.image.resize(img, final_size)

        # Normalisation des pixels
        img = img / 255.0

        processed_images.append(img)

    processed_images = np.array(processed_images)

    return processed_images






# ------------------ TEST ------------------ #

# df = pd.read_csv('raw_data/CXR8/CXR8/Data_Entry_2017_v2020.csv')
# image_list=["00000001_000.png", "00000002_000.png"]

# data_img = resize_all_images(df, target_phys_size=(256, 256), final_size=(64, 64))

# print(len(data_img))
# plt.imshow(data_img[0], cmap='gray')
# plt.axis('off')
# plt.show()
