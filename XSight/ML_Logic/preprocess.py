##### IMPORTS #####

from typing import Tuple
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import shutil
import random

from XSight.ML_Logic.data import fetch_png_images
from sklearn.model_selection import train_test_split

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from typing import List
from XSight.params import PATHO_COLUMNS, PATIENT_ID_COL, RANDOM_STATE, GCP_PROJECT, BUCKET_NAME, PREFIX
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


import random



##### --------- FUNCTIONS PREPROC DATA --------- #####


def load_data(filepath: str) -> pd.DataFrame:
    """Load a csv file into a Dataframe."""
    return pd.read_csv(filepath)

def df_structure(df: pd.DataFrame,label_column: str = 'Finding Labels') -> pd.DataFrame:
    if label_column in df.columns:
        dummies = df[label_column].str.get_dummies(sep='|')
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[label_column])
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not needed for model training."""
    columns_to_drop = [
        'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 'Follow-up #'
    ]
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

def encode_labels(df: pd.DataFrame, save_scaler_path: str) -> pd.DataFrame:
    """OneHotEncode the labels and drop the original column."""

    # Filtrer les lignes avec une seule maladie (pas de '|')
    df = df[~df["Finding Labels"].str.contains("\|")]

    # Supprimer certaines pathologies rares
    labels_to_remove = ['Emphysema', 'Fibrosis', 'Edema', 'Pneumonia', 'Hernia']
    df = df[~df["Finding Labels"].isin(labels_to_remove)]

    # Séparer "No Finding" des autres
    no_finding_df = df[df["Finding Labels"] == "No Finding"]
    other_df = df[df["Finding Labels"] != "No Finding"]

    # Sous-échantillonner "No Finding" à 20 000
    no_finding_sampled = no_finding_df.sample(n=20000, random_state=42)

    # Recombiner
    df = pd.concat([no_finding_sampled, other_df], ignore_index=True)

    label_column =  'Finding Labels'

    if label_column in df.columns:
        dummies = df[label_column].str.get_dummies(sep='|')
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[label_column])
    df['Patient Sex M'] = df['Patient Sex'].map({'M': 1, 'F':0})
    df['View Position PA'] = df['View Position'].map({'PA': 1, 'AP':0})
    df = df.drop(columns=['Patient Sex','View Position'])
    scaler = StandardScaler()
    scaler.fit(df[['Patient Age']])
    joblib.dump(scaler, save_scaler_path)


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
    # df = drop_unnecessary_columns(df)
    df = encode_labels(df,save_scaler_path="XSight/ML_Logic/scaler.joblib")
    return df

def preprocess_one_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take already preprocessed/encoded data and retain only specific columns,
    while creating a binary 'maladie' column.
    """
    df = df.copy()

    df['maladie'] = (df['No Finding'] == 0).astype(int)

    columns_to_keep = [
        'Image Index',
        'Patient Age',
        'Patient Sex M',
        'View Position PA',
        'Patient ID',
        'maladie'
    ]
    df = df[[col for col in columns_to_keep if col in df.columns]]

    return df

def preprocess_6cat(df: pd.DataFrame) -> pd.DataFrame:
    """
    From already encoded data, create 6-category classification and return cleaned DataFrame.
    """
    df = df.copy()

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

    df['categorie_6'] = df.apply(assigner_groupe, axis=1)

    # One-hot encode the 6 categories
    df_cat6 = pd.get_dummies(df['categorie_6'], prefix='cat6', dtype=int)
    df = pd.concat([df, df_cat6], axis=1)

    # Drop old pathology and helper columns
    drop_cols = cardio_pleurale + pulmonaire_diffuse + pulmonaire_chronique + tumeur + autres
    drop_cols += ['No Finding', 'categorie_6', 'maladie']  # maladie may or may not be there
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    return df


def stratified_chunk_split(df: pd.DataFrame, chunk_sizes: List[int], patho_columns: List[str], random_state: int = 42) -> List[pd.DataFrame]:
    """
    Split the full DataFrame into stratified, non-overlapping chunks by image-level rows,
    preserving label distribution based on pathology columns.
    """

    X = df.index.values.reshape(-1, 1)
    y = df[patho_columns].values

    chunks = []
    seen_indices = set()

    for size in chunk_sizes:
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=size, random_state=random_state)
        _, chunk_idx = next(msss.split(X, y))

        # Remove already used indices
        chunk_idx = [idx for idx in chunk_idx if idx not in seen_indices]
        seen_indices.update(chunk_idx)

        # Append chunk
        chunk_df = df.iloc[chunk_idx].reset_index(drop=True)
        chunks.append(chunk_df)

    return chunks[0]





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
    images = fetch_png_images(
                                    gcp_project= GCP_PROJECT,
                                    bucket_name= BUCKET_NAME,
                                    prefix= PREFIX,
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

    return processed_images, final_size


#########################################################################
###           YOLO PREPROCESS          ###

def df_generation():
    """
    Renvoie un dataframe pret à pour organiser les dossiers d'images pour yolo
    """

    data = load_data(filepath)
    data = data[~data["Finding Labels"].str.contains("\|")]

    group_mapping = {
    # Classes supprimées
    "Hernia": None,
    "Pneumonia": None,
    "Fibrosis": None,

    # Groupes
    "Atelectasis": "Opacités pulmonaires",
    "Consolidation": "Opacités pulmonaires",
    "Edema": "Opacités pulmonaires",
    "Infiltration": "Opacités pulmonaires",
    "Effusion": "Anomalies pleurales",
    "Pleural_Thickening": "Anomalies pleurales",
    "Mass": "Nodules/masses",
    "Nodule": "Nodules/masses",
    "Emphysema": "Parenchyme chronique",

    # Classes conservées
    "Cardiomegaly": "Cardiomegaly",
    "Pneumothorax": "Pneumothorax",
    "No Finding": "No Finding"
}

    # Application du mapping
    data['Finding Labels'] = data['Finding Labels'].map(group_mapping)

    # Suppression des classes exclues (Hernia, Pneumonia, Fibrosis)
    # data = data.dropna(subset=['Groupe'])
    data = data.dropna(subset=['Finding Labels'])

    # Séparer No Finding des autres classes
    no_findings = data[data["Finding Labels"] == "No Finding"]
    others      = data[data["Finding Labels"] != "No Finding"]

    # Sous-échantillonner No Finding
    no_findings_sample = no_findings.sample(n=18000, random_state=42)

    # Séparer Opacités pulmonaires des autres pathologies
    opacites = others[others["Finding Labels"] == "Opacités pulmonaires"]
    rest     = others[others["Finding Labels"] != "Opacités pulmonaires"]

    # Sous-échantillonner Opacités pulmonaires
    opacites_sample = opacites.sample(n=6000, random_state=42)

    # Reconstituer le DataFrame équilibré
    balanced_df = pd.concat([rest, no_findings_sample, opacites_sample], axis=0)

    # Mélanger les lignes
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df



def create_directory_structure(base_path, classes, splits=['train', 'val']):
    """Create the directory structure for YOLO classification."""
    for split in splits:
        split_path = os.path.join(base_path, split)
        os.makedirs(split_path, exist_ok=True)
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            os.makedirs(class_path, exist_ok=True)
    print(f"[✓] Directory structure created for {len(classes)} classes in {splits}.")


def copy_images(data, split_name, images_dir, output_dir):
    """Copy images to the appropriate directory for each class."""
    copied_count = 0
    missing_count = 0

    for _, row in data.iterrows():
        image_name = row['image_name']
        class_name = row['class']

        src_path = os.path.join(images_dir, image_name)
        dst_path = os.path.join(output_dir, split_name, class_name, image_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            missing_count += 1
            if missing_count <= 5:
                print(f"[!] Missing image: {src_path}")

    print(f"[{split_name.upper()}] Copied {copied_count} images | Missing: {missing_count}")
    return copied_count, missing_count


def prepare_yolo_dataset(csv_path, images_dir, output_dir, test_size=0.2, random_state=42):
    """
    Prepares a folder structure and splits a labeled image dataset for YOLO classification training.
    - csv_path: path to a CSV file with 'Image Index' and 'Finding Labels' columns
    - images_dir: directory containing the source images
    - output_dir: destination folder for YOLO-ready dataset
    """
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} records from CSV")

    # Extract all labels from multi-label fields
    all_labels = []
    for labels in df['Finding Labels'].dropna():
        all_labels.extend([label.strip() for label in labels.split('|')])

    unique_classes = sorted(set(all_labels))
    print(f"[INFO] Detected {len(unique_classes)} unique classes:\n{unique_classes}")

    # Create directory structure
    create_directory_structure(output_dir, unique_classes)

    # Build simplified DataFrame for single-label training (use only first label)
    data_for_split = []
    for _, row in df.iterrows():
        if pd.notna(row['Finding Labels']):
            primary_class = row['Finding Labels'].split('|')[0].strip()
            data_for_split.append({
                'image_name': row['Image Index'],
                'class': primary_class
            })
    data_df = pd.DataFrame(data_for_split)
    print(f"[INFO] Prepared {len(data_df)} images for stratified splitting.")

    # Stratified train/val split
    train_data, val_data = train_test_split(
        data_df,
        test_size=test_size,
        random_state=random_state,
        stratify=data_df['class']
    )
    print(f"[INFO] Train/Val split → {len(train_data)} / {len(val_data)} images.")

    # Copy files
    train_copied, _ = copy_images(train_data, 'train', images_dir, output_dir)
    val_copied, _ = copy_images(val_data, 'val', images_dir, output_dir)

    # Print class distribution
    print("\n[TRAIN class distribution]")
    print(train_data['class'].value_counts().sort_index())
    print("\n[VAL class distribution]")
    print(val_data['class'].value_counts().sort_index())

    return unique_classes, train_copied, val_copied





# ------------------ TEST ------------------ #

# df = pd.read_csv('raw_data/CXR8/CXR8/Data_Entry_2017_v2020.csv')
# image_list=["00000001_000.png", "00000002_000.png"]

# data_img = resize_all_images(df, target_phys_size=(256, 256), final_size=(64, 64))

# print(len(data_img))
# plt.imshow(data_img[0], cmap='gray')
# plt.axis('off')
# plt.show()
