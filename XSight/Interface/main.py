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




def main():

    filepath = 'data/Data_Entry_2017_v2020.csv'

    print("Chargement des données...")
    df = preprocess_basic(filepath)
    df_chunk = stratified_chunk_split(df, chunk_sizes = [1000], patho_columns= PATHO_COLUMNS)


    print("Prétraitement des données ...")

    df_clean_one = preprocess_one_target(df_chunk)
    df_clean_six = preprocess_6cat(df_chunk)

    print('Chargement des images...')


    print("Prétraitement des images ...")
    processed_images = resize_all_images(df,
                                         image_list,   #on fournit la liste complete?
                                         final_size=(64, 64))



    print("Initialisation du modèle...")
    model = initialize_model(image_size:int,  # ?
                     num_tabular_features:int,
                     num_labels:int)

    print("Compiling du modèle...")
    compile_model(model: Model,
                  num_labels: int,
                  loss:str ='binary_crossentropy'

    print("Entraînement du modèle...")
    train_model(
        model: Model,
        X_img: np.ndarray,
        X_tab: np.ndarray,
        y: np.ndarray,
        batch_size=16,
        patience=2,
        epochs=100,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    )

    metrics = evaluate_model(
        model: Model,
        X_img: np.ndarray,
        X_tab: np.ndarray,
        y: np.ndarray,
        batch_size=64
    )

    print(metrics)



if __name__ == "__main__":
    main()
