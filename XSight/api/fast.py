import pandas as pd
import numpy as np
import joblib
from typing import Literal, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf

from XSight.params import PATHO_COLUMNS

from XSight.ML_Logic.registery import load_model_from_gcp
from XSight.ML_Logic.preprocess import preprocess_basic, preprocess_one_target, preprocess_6cat, resize_all_images, stratified_chunk_split
from PIL import Image
import io

app = FastAPI()
# app.state.model=load_model_from_gcp("model_registry/test_model_20250609_103057/model.keras")
app.state.model=load_model_from_gcp("model_registry/test_model_20250609_154800/model.keras")

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2

@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  patient_age: int = Form(...),
                  patient_sex: Literal["M", "F"] = Form(...),
                  view_position: Literal["PA", "AP"] = Form(...),
                  pixel_spacing_x: float = Form(...),
                  pixel_spacing_y: float = Form(...),
                  final_width: int = Form(512),
                  final_height: int = Form(512)
                  ):
    """
    Predicts pathology of patient given X-ray image and metadata :
    - Age (int)
    - Sex (M or F)
    - View point (PA or AP)
    """

    try:
        ### ---- Scaling metadata ---- ###

        final_size = (final_width, final_height)

        metadata = dict({"Patient Sex": patient_sex,
                        "Patient Age": patient_age,
                        "View Position": view_position})

        ### ---- Preprocessing file ---- ###
        metadata_df = pd.DataFrame([metadata])

        # Encodage des variables catégorielles
        metadata_df["Patient Sex M"] = metadata_df["Patient Sex"].map({"M": 1, "F": 0})
        metadata_df["View Position PA"] = metadata_df["View Position"].map({"PA": 1, "AP": 0})
        metadata_df.drop(columns=["Patient Sex", "View Position"], inplace=True)

        # Scaling de l'âge
        scaler = joblib.load("XSight/ML_Logic/scaler.joblib")
        metadata_df["Patient Age"] = scaler.transform(metadata_df[["Patient Age"]])

        # Conversion en numpy + batch
        X_tab = metadata_df.values.astype("float32")  # (1, 3)

        ### ---- Prétraitement image ---- ###
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Redimension à taille physique homogène
        target_phys_size = (256, 256)
        new_width = int(target_phys_size[0] / pixel_spacing_x)
        new_height = int(target_phys_size[1] / pixel_spacing_y)

        image = image.resize((new_width, new_height))
        image = image.resize(final_size)

        # Tensor, normalisation, batch
        X_img = np.array(image).astype("float32") / 255.0  # (H, W, 3)
        # X_img = np.expand_dims(X_img, axis=-1)  # (H, W, 1)
        X_img = np.expand_dims(X_img, axis=0)   # (1, H, W, 1)

        ### ---- Prédiction ---- ###
        y_pred = app.state.model.predict([X_img, X_tab])  # (n_classes,)

        prediction = pd.DataFrame(y_pred, columns=PATHO_COLUMNS)
        prediction = prediction.loc[:,(prediction !=0).any(axis=0)]
        prediction = prediction.iloc[0].to_dict()

        prediction = pd.DataFrame(y_pred, columns=PATHO_COLUMNS)
        prediction = prediction.loc[:,(prediction !=0).any(axis=0)]
        prediction_dict = prediction.iloc[0].to_dict()

        # IMPORTANT : Conversion en types Python natifs
        prediction_serializable = {k: float(v) for k, v in prediction_dict.items()}
        return {"pathologies": prediction_serializable}

    except Exception as e:
        import traceback
        error_detail = f"Erreur : {str(e)}\nTraceback : {traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

# Define a root \`/\` endpoint
@app.get("/")
def root():
    return {'greeting': 'hello'}



########### OLD ###########
# async def predict(file: UploadFile = File(...),
#                   patient_age: int = Form(...),
#                   patient_sex: Literal["M", "F"] = Form(...),
#                   view_position: Literal["PA", "AP"] = Form(...),
#                   pixel_spacing_x: float = Form(...),
#                   pixel_spacing_y: float = Form(...),
#                   final_size: Tuple[int, int] = (64, 64)
#                   ):
#     """
#     Predicts pathology of patient given X-ray image and metadata :
#     - Age (int)
#     - Sex (M or F)
#     - View point (PA or AP)
#     """

#     ### ---- Scaling metadata ---- ###

#     metadata = dict({"Patient Sex": patient_sex,
#                      "Patient Age": patient_age,
#                      "View Point": view_position})

#     metadata_df = pd.DataFrame([metadata])
#     scaler = joblib.load("XSight/ML_Logic/scaler.joblib")
#     X_tab = scaler.transform(metadata_df)


#     ### ---- Preprocessing file ---- ###

#     if file.content_type != "image/png":
#         raise HTTPException(status_code=400, detail="Image must be a PNG")

#     # Lecture de l'image en mémoire
#     image_bytes = await file.read()
#     try:
#         image = Image.open(io.BytesIO(image_bytes)).convert("L")  # "L" = mode 8-bit grayscale
#     except Exception as e:
#         raise HTTPException(status_code=400, detail="Impossible de lire l'image PNG")


#     # Taille physique cible (en pixels)
#     pixel_spacing_x = pixel_spacing_x
#     pixel_spacing_y = pixel_spacing_y

#     target_phys_size=(256, 256)

#     new_width = int(target_phys_size[0] / pixel_spacing_x)
#     new_height = int(target_phys_size[1] / pixel_spacing_y)

#     # Redimensionnement à taille physique homogène
#     image = tf.image.resize(image, (new_height, new_width))

#     # Resize final pour le modèle
#     image = tf.image.resize(image, final_size)

#     # Normalisation des pixels
#     X_img = image / 255.0

#     #Predicting with the model

#     y_pred = app.state.model.predict([X_img,X_tab])

#     # return y_pred
#     return {'pathologies':pd.DataFrame(y_pred)}
