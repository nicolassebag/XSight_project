import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms
import os
from google.cloud import storage  # Optionnel : pour télécharger le modèle depuis GCS


import pandas as pd
import numpy as np
import joblib
from typing import Literal, Tuple, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
import tensorflow as tf



from XSight.params import PATHO_COLUMNS
from XSight.Visualisation.gradcam import *
from XSight.ML_Logic.registery import load_pt_model_from_gcp
from XSight.ML_Logic.preprocess import preprocess_basic, preprocess_one_target, preprocess_6cat, resize_all_images, stratified_chunk_split
from PIL import Image
import io

app = FastAPI()

X_PRED={} # stockage des inputs


app.state.model=load_pt_model_from_gcp('model_registry/final_yolo_model/best_nico_balanced.pt')

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Prétraite l'image pour le modèle."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def predict(image: Image.Image) -> Dict[str, float]:
    """Prédit les probabilités des classes sur l'image."""
    img_np = np.array(image.resize((224, 224)))
    results = app.state.model.predict(source=img_np, imgsz=224, conf=0.25)
    preds = results[0].probs.data.cpu().numpy()
    class_names = app.state.model.names
    return {name: float(prob) for name, prob in zip(class_names.values(), preds)}

def generate_gradcam(image: Image.Image) -> Tuple[np.ndarray, Dict[str, float]]:
    """Génère la heatmap Grad-CAM et retourne l'image superposée + les prédictions."""
    img_np = np.array(image.resize((224, 224)))
    input_tensor = preprocess_image(image)
    input_tensor.requires_grad_(True)

    activations = {}
    gradients = {}
    def hook_activations(module, input, output):
        activations['value'] = output.detach()
    def hook_gradients(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    sequential_model = app.state.model.model.model
    last_conv = sequential_model[9]
    handle_activations = last_conv.register_forward_hook(hook_activations)
    handle_gradients = last_conv.register_backward_hook(hook_gradients)

    app.state.model.model.eval()
    output = app.state.model.model(input_tensor)
    logits = output[0]
    pred_class = logits.argmax(dim=1).item()

    app.state.model.model.zero_grad()
    logits[0, pred_class].backward()

    activation = activations['value'].squeeze(0)
    gradient = gradients['value'].squeeze(0)
    weights = gradient.mean(dim=(1, 2), keepdim=True)
    heatmap = (weights * activation).sum(dim=0)
    heatmap = torch.relu(heatmap)
    heatmap = heatmap / heatmap.max()
    heatmap = heatmap.detach().cpu().numpy()

    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    results = app.state.model.predict(source=img_np, imgsz=224, conf=0.25)
    preds = results[0].probs.data.cpu().numpy()
    class_names = app.state.model.names
    predictions = {name: float(prob) for name, prob in zip(class_names.values(), preds)}

    handle_activations.remove()
    handle_gradients.remove()
    return superimposed_img, predictions

def encode_image(img: np.ndarray) -> bytes:
    """Encode une image numpy en bytes (JPEG)."""
    ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buffer.tobytes()



@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        superimposed_img, predictions = generate_gradcam(image)
        image_bytes = encode_image(superimposed_img)
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/jpeg",
            headers={"X-Predictions": str(predictions)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        predictions = predict(image)
        return JSONResponse(content=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Define a root \`/\` endpoint
@app.get("/")
def root():
    return {'greeting': 'hello'}
