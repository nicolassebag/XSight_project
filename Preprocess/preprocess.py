##### IMPORTS #####

from typing import Tuple
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

from data import fetch_images_to_memory

import tensorflow as tf
from sklearn.preprocessing import StandardScaler



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
    df['Patient Sex M'] = df['Patient Sex'].map({'M': 1, 'F':0})
    df['View Position PA'] = df['View Position'].map({'PA': 1, 'AP':0})
    scaler = StandardScaler()
    scaler.fit(df[['Patient Age']])
    df['Patient Age'] = scaler.transform(df[['Patient Age']])
    return df

                ############################################################

def preprocess_basic(filepath: str) -> pd.DataFrame:
    """Complete preprocessing the pipeline.
    1- load data raw
    2- drop les columns relatives aux images
    3- encode les maladies en columns separees
    """
    df = load_data(filepath)
    df = drop_unnecessary_columns(df)
    df = encode_labels(df)
    return df

def preprocess_one_target(filepath: str) -> pd.DataFrame:
    """
    Preprocess data and retain only specific columns:
    'Image Index', 'Patient Age','Patient Sex M',
    'View Position PA', 'patient ID maladie'
    """
    df = load_data(filepath)
    df = drop_unnecessary_columns(df)
    df = encode_labels(df)

    df['maladie'] = (df['No Finding'] == 0).astype(int)

    columns_to_keep = [
        'Image Index',
        'Patient Age',
        'Patient Sex M',
        'View Position PA',
        'patient ID',
        'maladie'
    ]

    df = df[[col for col in columns_to_keep if col in df.columns]]

    return df

def preprocess_6cat(df: pd.DataFrame,filepath: str) -> pd.DataFrame:
    """
    Group multilabel pathology columns into 6 broader categories and return a cleaned DataFrame
    with one-hot encoded category columns.
    """

    df = load_data(filepath)
    df = drop_unnecessary_columns(df)
    df = encode_labels(df)

    cardio_pleurale      = ['Cardiomegaly', 'Edema', 'Effusion', 'Pleural_Thickening']
    pulmonaire_diffuse   = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumonia']
    pulmonaire_chronique = ['Emphysema', 'Fibrosis']
    tumeur               = ['Mass', 'Nodule']
    autres               = ['Hernia', 'Pneumothorax']

    def assigner_groupe(row):
        if row.get('No Finding', 0) == 1:
            return 'No_finding'
        if any(row.get(col, 0) == 1 for col in cardio_pleurale):
            return 'Cardio_Pleurale'
        if any(row.get(col, 0) == 1 for col in pulmonaire_diffuse):
            return 'Pulmonaire_Diffuse'
        if any(row.get(col, 0) == 1 for col in pulmonaire_chronique):
            return 'Pulmonaire_Chronique'
        if any(row.get(col, 0) == 1 for col in tumeur):
            return 'Tumeur'
        if any(row.get(col, 0) == 1 for col in autres):
            return 'Autres'
        return 'Inconnu'

    # Assign category
    df['categorie_6'] = df.apply(assigner_groupe, axis=1)

    # One-hot encode the new category column
    df_onehot = pd.get_dummies(df['categorie_6'], prefix='cat6', dtype=int)
    df = pd.concat([df, df_onehot], axis=1)

    # Drop old pathology columns + temporary category column
    columns_to_drop = cardio_pleurale + pulmonaire_diffuse + pulmonaire_chronique + tumeur + autres
    columns_to_drop += ['No Finding', 'categorie_6', 'maladie']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

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
