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


def initialize_model(input_shape: tuple, num_classes:int) -> Model:
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
    ### Last layer - Classification Layer with 20 outputs corresponding to 20 targets in dataset
    ### Use 'sigmoid' for multi-label classification (binary crossentropy loss) as multiple labels can be true for each sample
    model.add(layers.Dense(num_classes,activation='sigmoid'))

    print("✅ Model initialized")
    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = Adam(learning_rate=learning_rate)
    metrics = [AUC(name='pr_auc', multi_label=True, num_labels=20, curve='PR'), 'precision', 'recall']

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', # Use 'binary_crossentropy' for multi-label classification as multiple labels can be true for each sample
                  metrics=metrics)
    print("✅ Model compiled")
    return model


def train_model(
        model: Model,
        X: np.ndarray,
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

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )
    print(history.history)
    print(f"✅ Model trained on {len(X)} rows with min val scores:")
    print(f"precision: {round(np.min(history.history['val_precision']), 2)}")
    print(f"recall: {round(np.min(history.history['val_recall']), 2)}")
    print(f"AUC: {round(np.min(history.history['val_pr_auc']), 2)}")

    return model, history

def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
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
