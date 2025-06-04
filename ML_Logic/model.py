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
    # First Convolution & MaxPooling
    cnn_model = layers.Conv2D(8, kernel_size=(4,4), activation='relu', padding='same')(img_input)
    cnn_model = layers.MaxPool2D(pool_size=(2,2))(cnn_model)
    # Flattening
    cnn_model = layers.Flatten()(cnn_model)


    ### TABULAR
    tab_input = Input(shape=(num_tabular_features,), name='tab_input')
    # Hidden layers
    dense_model = layers.Dense(8, activation='relu')(tab_input)


    ### COMBINATION
    combined_input = layers.concatenate([cnn_model,dense_model])
    combined_model = layers.Dense(8, activation='relu')(combined_input)


    ### OUTPUT
    # Use 'sigmoid' for multi-label classification (binary crossentropy loss)
    # as multiple labels can be true for each sample
    output = layers.Dense(num_labels,activation='sigmoid')(combined_model)
    model = Model(inputs=[img_input,tab_input], outputs=output, name = 'full_model')

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


def compile_model(model: Model, num_labels, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = Adam(learning_rate=learning_rate)
    metrics = [AUC(name='pr_auc', multi_label=True, num_labels=num_labels, curve='PR'), 'precision', 'recall']

    # Use 'binary_crossentropy' for multi-label classification
    # as multiple labels can be true for each sample
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=metrics)
    print("✅ Model compiled")
    return model


def train_model(
        model: Model,
        X_img: np.ndarray,
        X_tab: np.ndarray,
        y: np.ndarray,
        batch_size=16,
        patience=2,
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
    print(history.history)
    print(f"✅ Model trained on {len(X_img)} rows with min val scores:")
    print(f"precision: {round(np.min(history.history['val_precision']), 2)}")
    print(f"recall: {round(np.min(history.history['val_recall']), 2)}")
    print(f"AUC: {round(np.min(history.history['val_pr_auc']), 2)}")

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
    precision = metrics["precision"]
    recall = metrics["recall"]
    pr_auc = metrics["pr_auc"]

    print(f"✅ Model evaluated")
    print(f"loss: {round(loss, 2)}")
    print(f"precision: {round(precision, 2)}")
    print(f"recall: {round(recall, 2)}")
    print(f"AUC: {round(pr_auc, 2)}")

    return metrics
