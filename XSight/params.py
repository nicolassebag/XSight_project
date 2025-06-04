import os
import numpy as np


############### XSIGHT-PROJECT CONSTANTS #################

DATA_SIZE= 100
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.5

PATIENT_ID_COL = 'patient ID'
LABEL_COLUMN = 'Finding Labels'

PATHO_COLUMNS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
    'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening',
    'Pneumonia', 'Pneumothorax'
]

SELECTED_COLUMNS = [
    'Image Index', 'Patient Age', 'No Finding', 'Patient Sex M',
    'View Position PA', 'patient ID', 'maladie'
]

GROUPS = {
    "cardio_pleurale": ['Cardiomegaly', 'Edema', 'Effusion', 'Pleural_Thickening'],
    "pulmonaire_diffuse": ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumonia'],
    "pulmonaire_chronique": ['Emphysema', 'Fibrosis'],
    "tumeur": ['Mass', 'Nodule'],
    "autres": ['Hernia', 'Pneumothorax']
}


#############################################################
################ TO BE UPDATED ##############################
#############################################################

# ##################  VARIABLES  ##################
# DATA_SIZE = os.environ.get("DATA_SIZE")
# GCP_PROJECT = os.environ.get("GCP_PROJECT")
# GCP_REGION = os.environ.get("GCP_REGION")
# BUCKET_NAME = os.environ.get("BUCKET_NAME")

# ##################  CONSTANTS  #####################
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
# LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")

# COLUMN_NAMES_RAW = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

# DTYPES_RAW = {
#     "fare_amount": "float32",
#     "pickup_datetime": "datetime64[ns, UTC]",
#     "pickup_longitude": "float32",
#     "pickup_latitude": "float32",
#     "dropoff_longitude": "float32",
#     "dropoff_latitude": "float32",
#     "passenger_count": "int16"
# }

# DTYPES_PROCESSED = np.float32



# ################## VALIDATIONS #################

# env_valid_options = dict(
#     DATA_SIZE=["1k", "200k", "all"],
#     MODEL_TARGET=["local", "gcs", "mlflow"],
# )

# def validate_env_value(env, valid_options):
#     env_value = os.environ[env]
#     if env_value not in valid_options:
#         raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


# for env, valid_options in env_valid_options.items():
#     validate_env_value(env, valid_options)
