import os
import json
import pandas as pd
from datetime import datetime
from tensorflow import keras
from colorama import Fore, Style

LOCAL_REGISTRY_PATH = "model_registry"
os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)

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

def load_model_from_run(run_dir: str) -> keras.Model:
    model_path = os.path.join(run_dir, "model.keras")
    if not os.path.exists(model_path):
        print(Fore.RED + f"❌ Model file not found at {model_path}" + Style.RESET_ALL)
        return None
    model = keras.models.load_model(model_path)
    print(Fore.GREEN + f"✅ Model loaded from {model_path}" + Style.RESET_ALL)
    return model



################ USAGE DE REGISTERY ###########################
#run_dir = get_run_dir("the_model_name")

#save_model(model, run_dir)
#save_weights(model, run_dir)
#save_metrics_csv(history.history, run_dir)
#save_results_json(params, metrics, run_dir)
