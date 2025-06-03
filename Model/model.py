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


def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()

    model.add(Input(shape=(64,64,1)))

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8, kernel_size=(4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    ### Flattening
    model.add(layers.Flatten())
    ### Last layer - Classification Layer with 20 outputs corresponding to 20 targets in dataset
    model.add(layers.Dense(20,activation='softmax'))

    print("✅ Model initialized")
    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("✅ Model compiled")
    return model


def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=16,
        patience=2,
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
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

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
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, MAE: {round(accuracy, 2)}")

    return metrics
