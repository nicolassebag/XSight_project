# Data manipulation
import numpy as np
import pandas as pd
from typing import Tuple
import os
from colorama import Fore, Style

from keras import Model, Sequential, Input, layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.metrics import AUC

from PIL import Image

import tensorflow as tf

from google.cloud import storage
from io import BytesIO

from XSight.ML_Logic.model import initialize_model, compile_model, train_model, evaluate_model
from XSight.ML_Logic.preprocess import preprocess_basic, preprocess_one_target, preprocess_6cat, resize_all_images, stratified_chunk_split
from XSight.params import *

### Variables ###


num_labels = 1 # 1, 6 ou 15 features
final_size = (64,64)

### ---------------- MAIN ---------------- ###


def main(num_labels = num_labels ):

    filepath = 'data/Data_Entry_2017_v2020.csv'

    print("Chargement des données...")
    df = preprocess_basic(filepath)
    df_chunk = stratified_chunk_split(df, chunk_sizes = [10], patho_columns= PATHO_COLUMNS)


    print("Prétraitement des données ...")

    if num_labels == 1 :
        df_chunk = preprocess_one_target(df_chunk)

    if num_labels == 6 :
        df_chunk = preprocess_6cat(df_chunk)

    image_list = df_chunk['Image Index']

    print("Prétraitement des images ...")
    processed_images, image_size = resize_all_images(df,
                                         image_list,
                                         final_size= final_size)
    image_size = (tuple(image_size),1)


    print("Initialisation du modèle...")
    model = initialize_model(image_size,
                     num_tabular_features = 3, #age, sexe, viewpoint
                     num_labels = num_labels)

    print("Compiling du modèle...")
    compile_model(model= model,
                  num_labels = num_labels,
                  loss= 'binary_crossentropy'
                  )

    print("Entraînement du modèle...")

    X_tab = df_chunk[['Patient Age', 'Patient Sex M','View Position PA']]
    y = df_chunk['maladie']

    model,history = train_model(
                                model= model,
                                X_img = processed_images,
                                X_tab= X_tab ,
                                y=y ,
                                batch_size=16,
                                patience=2,
                                epochs=100,
                                validation_data=None, # overrides validation_split
                                validation_split=0.3
                            )

    # metrics = evaluate_model(
    #     model= model,
    #     X_img: np.ndarray,
    #     X_tab: np.ndarray,
    #     y: np.ndarray,
    #     batch_size=64
    # )

    # print(metrics)



if __name__ == "__main__":
    main()
