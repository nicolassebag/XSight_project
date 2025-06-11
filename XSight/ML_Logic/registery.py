import os
import json
import pandas as pd
from datetime import datetime
from tensorflow import keras
from colorama import Fore, Style
from google.cloud import storage
import tempfile
from XSight.params import *
from ultralytics import YOLO
import tempfile

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

model_bucket_name = MODEL_BUCKET_NAME
model_prefix = MODEL_PREFIX

os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)

# ────────────────────────────────────────────────
# LOCAL FILE HANDLING
# ────────────────────────────────────────────────

def get_run_dir(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOCAL_REGISTRY_PATH, f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(Fore.GREEN + f"▶️  Run directory created: {run_dir}" + Style.RESET_ALL)
    return run_dir

def save_model(model: keras.Model, run_dir: str) -> str:
    model_path = os.path.join(run_dir, "model.keras")
    model.save(model_path)
    print(Fore.GREEN + f"✅ Model saved to {model_path}" + Style.RESET_ALL)
    return model_path

def save_weights(model: keras.Model, run_dir: str) -> str:
    weights_path = os.path.join(run_dir, "final_weights.weights.h5")
    model.save_weights(weights_path)
    print(Fore.GREEN + f"🏁 Weights saved to {weights_path}" + Style.RESET_ALL)
    return weights_path

def save_metrics_csv(history: dict, run_dir: str) -> str:
    metrics_df = pd.DataFrame(history)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index_label="epoch")
    print(Fore.GREEN + f"📈 Metrics saved to {metrics_path}" + Style.RESET_ALL)
    return metrics_path

def save_results_json(params: dict, metrics: dict, run_dir: str) -> None:
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(Fore.GREEN + "✅ Params and metrics JSON saved." + Style.RESET_ALL)


# ────────────────────────────────────────────────
# METRICS AUTO-EXTRACTION
# ────────────────────────────────────────────────

def extract_final_metrics(history: dict) -> dict:
    """
    Extracts final epoch metrics from model.fit() history.
    """
    final_metrics = {}
    for key, values in history.items():
        if isinstance(values, list) and len(values) > 0:
            final_metrics[key] = values[-1]
    return final_metrics

# ────────────────────────────────────────────────
# GCS UPLOAD
# ────────────────────────────────────────────────

def upload_directory_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, LOCAL_REGISTRY_PATH)
            blob_path = os.path.join(gcs_prefix, relative_path)

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(Fore.CYAN + f"☁️ Uploaded to gs://{bucket_name}/{blob_path}" + Style.RESET_ALL)

# ────────────────────────────────────────────────
# FINALIZATION
# ────────────────────────────────────────────────

def finalize_and_upload(model, history, params, model_name="model"):
    # Create a fresh run directory internally
    run_dir = get_run_dir(model_name)

    # Save model and the params and metrics
    save_model(model, run_dir)
    save_weights(model, run_dir)
    save_metrics_csv(history, run_dir)

    # Always extract metrics from history automatically
    final_metrics = extract_final_metrics(history)
    save_results_json(params, final_metrics, run_dir)

    # Upload everything to GCS
    upload_directory_to_gcs(run_dir, MODEL_BUCKET_NAME, MODEL_PREFIX)

    print(Fore.MAGENTA + "✅ All artifacts saved and uploaded securely." + Style.RESET_ALL)

# ────────────────────────────────────────────────
# LOAD MODEL FROM GCP
# ────────────────────────────────────────────────

def load_model_from_gcp(gcp_path: str) -> keras.Model:
    """
    Download model.keras from GCP and load it with keras.

        gcp_path: Full GCS path to model.keras (e.g. "model_registry/your_model/model.keras").

    Returns:
        keras.Model
    """
    import tempfile
    bucket_name = MODEL_BUCKET_NAME
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcp_path)

    if not blob.exists():
        print(Fore.RED + f"❌ Model file not found in GCS at: gs://{bucket_name}/{gcp_path}" + Style.RESET_ALL)
        return None

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
        blob.download_to_filename(tmp_file.name)
        print(Fore.CYAN + f"⬇️  Downloaded model from GCS to: {tmp_file.name}" + Style.RESET_ALL)
        model = keras.models.load_model(tmp_file.name)
        print(Fore.GREEN + f"✅ Model loaded successfully from GCS!" + Style.RESET_ALL)

    return model

def load_pt_model_from_gcp(gcp_path: str) -> YOLO:
    """
    Download model.pt from GCP and load it with ultralytics.YOLO.

        gcp_path: Full GCS path to model.pt (e.g. "model_registry/your_model/best_nico_balanced.pt").

    Returns:
        ultralytics.YOLO
    """
    bucket_name = MODEL_BUCKET_NAME
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcp_path)

    if not blob.exists():
        print(f"❌ Model file not found in GCS at: gs://{bucket_name}/{gcp_path}")
        return None

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        blob.download_to_filename(tmp_file.name)
        print(f"⬇️  Downloaded model from GCS to: {tmp_file.name}")
        model = YOLO(tmp_file.name)
        print(f"✅ Model loaded successfully from GCS!")

    return model

# load_model_from_gcp("model_registry/test_model_20250609_154800/model.keras")

################ USAGE DE REGISTERY ###########################
#from registry import finalize_and_upload

#finalize_and_upload(
#    model=model,
#    history=history.history,
#    params={"lr": 0.001, "batch_size": 64},
#    model_name="my_cnn_model"
#)
