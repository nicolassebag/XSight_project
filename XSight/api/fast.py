import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from XSight.ML_Logic.registery import load_model
from XSight.ML_Logic.preprocess import encode_metada, preprocess_basic, preprocess_one_target, preprocess_6cat, resize_all_images, stratified_chunk_split


app = FastAPI()
app.state.model=load_model()

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
async def predict(image: UploadFile = File(...),
                  patient_age: int = Form(...),
                  patient_sex: str["M", "F"] = Form(...),
                  view_position: str["PA", "AP"] = Form(...),
                  pixel_spacing_x: float = Form(...),
                  pixel_spacing_y: float = Form(...),
                  ):
    """
    Predicts pathology of patient given X-ray image and metadata :
    - Age (int)
    - Sex (M or F)
    - View point (PA or AP)
    """

    metadata = dict({"Patient Sex": patient_sex,
                     "Patient Age": patient_age,
                     "View Point": view_position})
    metadata_df = pd.DataFrame([metadata])

    X_tab = encode_metada(metadata_df)

    X_img = None ##### A COMPLETER

    y_pred = app.state.model.predict([X_img,X_tab])

    # return y_pred
    return {'pathologies':pd.DataFrame(y_pred)}

# Define a root \`/\` endpoint
@app.get("/")
def root():
    return {'greeting': 'hello'}
