# Data manipulation
import numpy as np
import pandas as pd
from typing import Tuple

# KERAS
from keras import Sequential, Input, layers
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
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    pass

def train(model,X,y):
    batch_size = 16
    epochs = 10
    es = EarlyStopping(patience=2)
    validation_split = 0.1

    model.fit(X,
          y,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[es],
          validation_split=validation_split
          )
    return model.get_metrics_result()
