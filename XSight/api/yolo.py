import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms
import os
from google.cloud import storage  # Optionnel : pour télécharger le modèle depuis GCS

import base64
import pandas as pd
import numpy as np
import joblib
from typing import Literal, Tuple, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse, HTMLResponse
import tensorflow as tf



# from XSight.params import PATHO_COLUMNS
# from XSight.Visualisation.gradcam import *
from XSight.ML_Logic.registery import load_pt_model_from_gcp
# from XSight.ML_Logic.preprocess import preprocess_basic, preprocess_one_target, preprocess_6cat, resize_all_images, stratified_chunk_split
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



# --- Configuration et fonctions (à adapter selon ton code) ---
def preprocess_image(image: Image.Image) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def predict(model, img_np):
    results = model.predict(source=img_np, imgsz=224, conf=0.25)
    preds = results[0].probs.data.cpu().numpy()
    class_names = model.names
    return {name: float(prob) for name, prob in zip(class_names.values(), preds)}

def generate_gradcam(model, image: Image.Image):
    img_np = np.array(image.resize((224, 224)))
    input_tensor = preprocess_image(image)
    input_tensor.requires_grad_(True)

    activations = {}
    gradients = {}
    def hook_activations(module, input, output):
        activations['value'] = output.detach()
    def hook_gradients(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    sequential_model = model.model.model
    last_conv = sequential_model[9]
    handle_activations = last_conv.register_forward_hook(hook_activations)
    handle_gradients = last_conv.register_backward_hook(hook_gradients)

    model.model.eval()
    output = model.model(input_tensor)
    logits = output[0]
    pred_class = logits.argmax(dim=1).item()

    model.model.zero_grad()
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

    results = model.predict(source=img_np, imgsz=224, conf=0.25)
    preds = results[0].probs.data.cpu().numpy()
    class_names = model.names
    predictions = {name: float(prob) for name, prob in zip(class_names.values(), preds)}

    handle_activations.remove()
    handle_gradients.remove()
    return superimposed_img, predictions

def plot_probs(predictions: dict) -> bytes:
    sorted_names = sorted(predictions.keys(), key=lambda x: -predictions[x])
    sorted_probs = [predictions[name] for name in sorted_names]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(sorted_names, sorted_probs, color='orange', height=0.7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Probabilité prédite')
    ax.set_title('Probabilités des classes (ordre décroissant)')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    for bar, prob in zip(bars, sorted_probs):
        width = bar.get_width()
        ax.text(width + 0.02,
                bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%',
                va='center',
                ha='left',
                fontsize=10,
                color='black')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def encode_image(img: np.ndarray) -> bytes:
    ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buffer.tobytes()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(image.resize((224, 224)))

        # Générer la heatmap et les prédictions
        superimposed_img, predictions = generate_gradcam(app.state.model, image)

        # Générer le graphique des probabilités
        prob_plot = plot_probs(predictions)

        # Encoder les images en base64 pour les inclure dans une page HTML ou un JSON
        heatmap_bytes = encode_image(superimposed_img)
        heatmap_base64 = base64.b64encode(heatmap_bytes).decode('utf-8')
        prob_plot_base64 = base64.b64encode(prob_plot.read()).decode('utf-8')

        # Retourner une page HTML avec les deux images (ou un JSON)
        html_content = f"""
        <html>
            <body>
                <h2>Image avec heatmap</h2>
                <img src="data:image/jpeg;base64,{heatmap_base64}" />
                <h2>Probabilités par pathologie</h2>
                <img src="data:image/png;base64,{prob_plot_base64}" />
            </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define a root \`/\` endpoint
@app.get("/")
def root():
    return {'greeting': 'hello'}
