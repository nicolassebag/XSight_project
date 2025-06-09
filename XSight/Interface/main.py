# Data manipulation
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os
import yaml
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

from XSight.ML_Logic.model import initialize_model, compile_model, train_model, evaluate_model, make_dataset, initialize_resnet_model, configure_gpu
from XSight.ML_Logic.split_data import split_data
from XSight.ML_Logic.preprocess import preprocess_basic, preprocess_one_target, preprocess_6cat, resize_all_images, stratified_chunk_split
from XSight.params import *
from XSight.ML_Logic.registery import finalize_and_upload





### Variables ###


num_labels = 1 # 1, 6 ou 15 features
final_size = (64,64)
path = 'gs://cxr8_images_bucket/images_preprocessed_256/images_256x256'
### ---------------- MAIN ---------------- ###


def main(num_labels = num_labels ):

    filepath = IMAGE_LIST_PATH

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
    image_size = (image_size[0], image_size[1], 1)
    print(image_size)
    # image_size = image_size

    print("Initialisation du modèle...")
    model = initialize_model(image_size,
                     num_tabular_features = 3, #age, sexe, viewpoint
                     num_labels = num_labels)

    print("Compiling du modèle...")
    compile_model(model= model,
                  num_labels = num_labels,
                  loss= 'binary_focal_crossentropy'
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

    params = {
    "epochs": epochs,
    "batch_size": batch_size,
    "validation_split": validation_split,
    "patience" : patience
    }

    finalize_and_upload(
    model=model,
    history=history,
    params=params,
    model_name='model'
)

    # metrics = evaluate_model(
    #     model= model,
    #     X_img: np.ndarray,
    #     X_tab: np.ndarray,
    #     y: np.ndarray,
    #     batch_size=64
    # )

    # print(metrics)





def run_experiment(
    csv_path: str,
    image_folder: str,
    image_size: Tuple[int,int],
    sample_size: Optional[int],
    batch_size: int = 128,
    batch_size_val: int = 128,
    patience: int = 10,
    epochs: int = 200,
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42,
    output_base: str = "experiments"
):
    # GPU + mixed precision
    configure_gpu()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Load & sample
    df = preprocess_basic(csv_path)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)
    print(Fore.CYAN + f"Chargé {len(df)} ex." + Style.RESET_ALL)

    # Split stratifié patient-level
    df_train, df_val, df_test = split_data(
        data_encoded=df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    print(Fore.CYAN + f"Split train={len(df_train)} val={len(df_val)} test={len(df_test)}" + Style.RESET_ALL)

    # Build paths, X_tab, y (15 labels)

    def build(df_split, gcs_bucket_path):
        imgs = df_split["Image Index"].map(lambda fn: f"{gcs_bucket_path}/{fn}").tolist()
        X_tab = df_split[["Patient Age", "Patient Sex M", "View Position PA"]].values.astype(np.float32)
        y = df_split[PATHO_COLUMNS].values.astype(np.float32)
        return imgs, X_tab, y

    # def build(df_split):
    #     imgs = df_split["Image Index"].map(lambda fn: os.path.join(image_folder, fn)).tolist()
    #     X_tab = df_split[["Patient Age", "Patient Sex M", "View Position PA"]].values.astype(np.float32)
    #     y = df_split[PATHO_COLUMNS].values.astype(np.float32)
    #     return imgs, X_tab, y

    train_imgs, train_X, train_y = build(df_train,path)
    val_imgs,   val_X,   val_y   = build(df_val,path)
    test_imgs,  test_X,  test_y  = build(df_test,path)

    # Datasets
    train_ds = make_dataset(train_imgs, train_X, train_y, image_size, batch_size, shuffle=True)
    val_ds   = make_dataset(val_imgs,   val_X,   val_y,   image_size, batch_size_val, shuffle=False)
    test_ds  = make_dataset(test_imgs,  test_X,  test_y,  image_size, batch_size_val, shuffle=False)

    # Steps
    train_steps = len(train_imgs) // batch_size
    val_steps   = max(1, len(val_imgs) // batch_size_val)
    print(Fore.CYAN + f"Steps train={train_steps} val={val_steps}" + Style.RESET_ALL)

    # run_dir & config
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_base, f"run_{ts}_{image_size[0]}x{image_size[1]}_15lbl")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump({
            "csv": csv_path,
            "images": image_folder,
            "image_size": image_size,
            "batch_train": batch_size,
            "batch_val": batch_size_val,
            "patience": patience,
            "epochs": epochs,
            "test_size": test_size,
            "val_size": val_size,
            "random_state": random_state
        }, f)
    print(Fore.GREEN + "▶️  Run dir:", run_dir + Style.RESET_ALL)

    # 7) Model & compile
    model = initialize_resnet_model(image_size + (3,), num_tabular_features=3, num_labels=len(PATHO_COLUMNS))
    model = compile_model(model, initial_lr=1e-4, decay_steps=1000, decay_rate=0.9, use_focal=True)

    # 8) Callbacks & fit
    # cb_ckpt = ModelCheckpoint(
    #     os.path.join(run_dir, "ckpt_{epoch:02d}.h5"),
    #     monitor="val_loss", mode="min", save_best_only=True
    # )
    cb_es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[cb_es],
        verbose=2
    )

    # 9) Save final & metrics
    model.save(os.path.join(run_dir, "model_final.keras"))
    pd.DataFrame(history.history).to_csv(os.path.join(run_dir, "metrics.csv"), index_label="epoch")
    print(Fore.GREEN + "✅ Modèle et métriques sauvegardés dans", run_dir + Style.RESET_ALL)


#     params = {
#     "epochs": epochs,
#     "batch_size": batch_size,
#     "validation_split": 0.3,
#     "patience" : patience
#     }

#     finalize_and_upload(
#         model=model,
#         history=history,
#         params=params,
#         model_name='model'
# )

    return test_ds

if __name__ == "__main__":
    # main()
    test_ds = run_experiment(
        csv_path="data/Data_Entry_2017_v2020.csv",
        image_folder= path,
        image_size=(64, 64),
        sample_size=50,
        batch_size=256,
        patience=5,
        epochs=20,
        output_base="experiments"
    )
