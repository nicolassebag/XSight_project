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

from Model.model import initialize_model, compile_model, train_model, evaluate_model
from Preprocess.preprocess import load_data, drop_unnecessary_columns, encode_labels, preprocess_basic, preprocess_one_target, preprocess_6cat, resize_all_images
from XSight.ML_Logic.data import fetch_images_to_memory
