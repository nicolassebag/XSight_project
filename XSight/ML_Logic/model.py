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


def compile_model(model: Model, num_labels: int, loss:str ='binary_crossentropy') -> Model:
    """
    Compile the Neural Network
    Loss : 'binary_crossentropy' or 'binary_focal_crossentropy'
    - Use 'binary_crossentropy' for multi-label classification as multiple labels can be true for each sample
    - Alternatively use 'binary_focal_crossentropy' to offset imbalanced classes : penalizes more strongly errors on rare classes.
    """
    learning_rate_schedule = ExponentialDecay(initial_learning_rate=0.001,
                                              decay_steps=100,
                                              decay_rate=0.9)

    optimizer = Adam(learning_rate=learning_rate_schedule)

    metrics = [AUC(name='pr_auc', multi_label=True, num_labels=num_labels, curve='PR'),
               'accuracy',
               'precision',
               'recall']

    if loss == 'binary_focal_crossentropy':
        loss = BinaryFocalCrossentropy(apply_class_balancing=True,
                                       gamma=2.0, # peut être augmenté (jusqu'à 5 par exemple pour pénalisation plus forte)
                                       label_smoothing=0.0,
                                       reduction='sum_over_batch_size',
                                    )

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)




    print("✅ Model compiled")
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
