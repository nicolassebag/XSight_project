# Data manipulation
import numpy as np
import pandas as pd
from typing import Tuple

from colorama import Fore, Style

# KERAS
from keras import Model, Sequential, Input, layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.metrics import AUC
from keras.losses import BinaryFocalCrossentropy
import tensorflow as tf
from tensorflow.keras import Model, layers, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import BinaryFocalCrossentropy

from ultralytics import YOLO


from ultralytics import YOLO

def initialize_yolo_model(model_name="yolo11x-cls"):
    """
    Initialise un modèle YOLO pour la classification.
    Par défaut : yolo11x-cls (pré-entraîné sur ImageNet), passer le chemin des poids (.pt) pour un autre
    modèle pré-entrainé.
    """
    model = YOLO(model_name)
    return model


def train_yolo_model(model, data_dir="yolo_dataset_groupé_nico_adjusted", epochs=30, batch_size=128):
    """
    Entraîne le modèle YOLO sur un dossier de classification (style ImageNet).

    Args:
        model: objet YOLO initialisé
        data_dir: dossier contenant /train et /val avec une sous-arborescence par classe
        epochs: nombre d'epochs
        batch_size: taille des batchs
    """
    results = model.train(
        data=data_dir,
        epochs=epochs,
        imgsz=224,
        degrees=10,
        translate=0.1,
        scale=0.1,
        shear=0.1,
        hsv_v=0.1,
        hsv_s=0.1,
        auto_augment="randaugment",
        batch=batch_size,
        device=0  # ou 'cpu' si pas de GPU
    )
    return results


def initialize_model(image_size:int,
                     num_tabular_features:int,
                     num_labels:int) -> Model:
    """
    Initialize a model for image + tabular data.

    Parameters:
    - image_size: int, e.g., 64 for 64x64 images
    - num_tabular_features: int, e.g., 3 for age, sex, view position
    - num_labels: int, number of target classes (multi-label)
    """

    ### IMAGE CNN
    img_input = Input(shape=image_size, name='img_input')
    # Convolution & MaxPooling
    cnn_model = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(img_input)
    cnn_model = layers.MaxPool2D(pool_size=(2,2))(cnn_model)
    cnn_model = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(cnn_model)
    cnn_model = layers.MaxPool2D(pool_size=(2,2))(cnn_model)
    cnn_model = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(cnn_model)
    cnn_model = layers.MaxPool2D(pool_size=(2,2))(cnn_model)
    # Flattening
    cnn_model = layers.Flatten()(cnn_model)


    ### TABULAR
    tab_input = Input(shape=(num_tabular_features,), name='tab_input')
    # Hidden layers
    dense_model = layers.Dense(16, activation='relu')(tab_input)
    dense_model = layers.Dropout(0.3)(dense_model)


    ### COMBINATION
    combined_input = layers.concatenate([cnn_model,dense_model])
    combined_model = layers.Dense(64, activation='relu',kernel_regularizer='l2')(combined_input)
    combined_model = layers.Dropout(0.5)(combined_model)

    ### OUTPUT
    # Use 'sigmoid' for multi-label classification (binary crossentropy loss)
    # as multiple labels can be true for each sample
    output = layers.Dense(num_labels,activation='sigmoid')(combined_model)
    model = Model(inputs=[img_input,tab_input], outputs=output, name = 'Full_Model')

    print("✅ Model initialized")
    return model


def initialize_model_old(input_shape: tuple, num_labels:int) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()

    model.add(Input(shape=(input_shape)))

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8, kernel_size=(4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    ### Flattening
    model.add(layers.Flatten())
    ### Last layer - Classification Layer with 15 outputs corresponding to 15 targets in dataset
    ### Use 'sigmoid' for multi-label classification (binary crossentropy loss) as multiple labels can be true for each sample
    model.add(layers.Dense(num_labels,activation='sigmoid'))

    print("✅ Model initialized")
    return model


def initialize_resnet_model(
    image_size: Tuple[int, int, int],
    num_tabular_features: int,
    num_labels: int,
    train_base: bool = False
) -> Model:
    """
    Modèle ResNet50 + tête fully-connected pour classification multi-label.
    Deux inputs : image et features tabulaires.
    """
    img_in = Input(shape=image_size, name="img_input")
    tab_in = Input(shape=(num_tabular_features,), name="tab_input")
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=img_in
    )
    base.trainable = train_base
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.15)(x)
    t = layers.Dense(32, activation=None)(tab_in)
    t = layers.BatchNormalization()(t)
    t = layers.Activation("relu")(t)
    t = layers.Dropout(0.15)(t)
    t = layers.Dense(32, activation=None)(t)
    t = layers.BatchNormalization()(t)
    t = layers.Activation("relu")(t)
    t = layers.Dropout(0.15)(t)
    combined = layers.concatenate([x, t])
    combined = layers.Dense(64, activation=None)(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Activation("relu")(combined)
    combined = layers.Dropout(0.15)(combined)
    out = layers.Dense(num_labels, activation="sigmoid", name="output")(combined)
    return Model(inputs=[img_in, tab_in], outputs=out, name="ResNet50_Tabular_ML")


def compile_model(
    model: Model,
    initial_lr: float = 1e-4,
    decay_steps: int = 500,
    decay_rate: float = 0.9,
    use_focal: bool = True
) -> Model:
    lr_sched = ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )
    opt = Adam(learning_rate=lr_sched)
    loss = (
        BinaryFocalCrossentropy(gamma=2.0)
        if use_focal
        else "binary_crossentropy"
    )
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            AUC(name="pr_auc", curve="PR"),
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    return model


def train_model(
        model: Model,
        X_img: np.ndarray,
        X_tab: np.ndarray,
        y: np.ndarray,
        batch_size=16,
        patience=10,
        epochs=100,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # X_img: (n_samples, 64, 64, 1)
    # X_tab: (n_samples, 3)
    # y: (n_samples, num_classes)

    history = model.fit(
        [X_img,X_tab],
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    # #print(history.history)
    # val_accuracy = np.array(history.history['val_accuracy'])
    # val_precision = np.array(history.history['val_precision'])
    # val_recall = np.array(history.history['val_recall'])
    # val_pr_auc = np.array(history.history['val_pr_auc'])

    # # Score combiné (peut être pondéré selon priorité)
    # combined_scores = val_accuracy + val_precision + val_recall + val_pr_auc
    # best_combined_idx = np.argmax(combined_scores)

    metrics = model.get_metrics_result()

    print(f"✅ Model trained on {len(X_img)} rows with best val scores:")
    print(f"validation loss: {round(metrics['loss'], 2)}")
    print(f"validation accuracy: {round(metrics['accuracy'], 2)}")
    print(f"validation precision: {round(metrics['precision'], 2)}")
    print(f"validation recall: {round(metrics['recall'], 2)}")
    print(f"validation PR-AUC: {round(metrics['pr_auc'], 2)}")

    return model, history

def evaluate_model(
        model: Model,
        X_img: np.ndarray,
        X_tab: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X_img)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=[X_img,X_tab],
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    pr_auc = metrics["pr_auc"]

    print(f"✅ Model evaluated")
    print(f"loss: {round(loss, 2)}")
    print(f"accuracy: {round(accuracy, 2)}")
    print(f"precision: {round(precision, 2)}")
    print(f"recall: {round(recall, 2)}")
    print(f"AUC: {round(pr_auc, 2)}")

    return metrics

def parse_image_function(
    img_path: tf.Tensor,
    tab_features: tf.Tensor,
    label: tf.Tensor,
    final_size: Tuple[int,int]
) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    img = tf.io.decode_png(tf.io.read_file(img_path), channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, final_size)
    img = tf.image.grayscale_to_rgb(img)
    img = tf.keras.applications.resnet.preprocess_input(img)
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        img = tf.cast(img, tf.float16)
    return (img, tab_features), label

def make_dataset(
    img_paths: list[str],
    X_tab: np.ndarray,
    y: np.ndarray,
    image_size: Tuple[int,int],
    batch_size: int,
    shuffle: bool = False
):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, X_tab, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_paths))
    def _map(p, t, l):
        return parse_image_function(p, t, l, final_size=image_size)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        def _aug(inputs, label):
            img, tab = inputs
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            return (img, tab), label
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(Fore.GREEN + f"GPU détectés ({len(gpus)}) – memory growth activé" + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "Aucun GPU détecté, entraînement sur CPU." + Style.RESET_ALL)
