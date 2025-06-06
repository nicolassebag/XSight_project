import os
import json
import yaml
import pandas as pd
from datetime import datetime
from tensorflow import keras
from colorama import Fore, Style

LOCAL_REGISTRY_PATH = "model_registry"
os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)


def get_run_dir(model_name: str) -> str:
    """
    Generate a unique run directory based on timestamp and model name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOCAL_REGISTRY_PATH, f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(Fore.GREEN + f"▶️  Run directory: {run_dir}" + Style.RESET_ALL)
    return run_dir


def save_model(model: keras.Model, run_dir: str) -> str:
    """
    Save model in Keras format to the given run directory.
    """
    model_path = os.path.join(run_dir, "model.keras")
    model.save(model_path)
    print(Fore.GREEN + f"✅ Model saved to {model_path}" + Style.RESET_ALL)
    return model_path


def save_weights(model: keras.Model, run_dir: str) -> str:
    """
    Save final model weights to a .h5 file in run_dir.
    """
    weights_path = os.path.join(run_dir, "final_weights.weights.h5")
    model.save_weights(weights_path)
    print(Fore.GREEN + f"🏁 Final weights saved to: {weights_path}" + Style.RESET_ALL)
    return weights_path


def save_config(config: dict, run_dir: str) -> None:
    """
    Save a training config as config.yaml in the run_dir.
    """
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(Fore.BLUE + f"🔖 Config saved to {config_path}" + Style.RESET_ALL)


def save_metrics_csv(history: dict, run_dir: str) -> str:
    """
    Save training history (metrics) to metrics.csv in run_dir.
    """
    metrics_df = pd.DataFrame(history)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index_label="epoch")
    print(Fore.GREEN + f"📈 Metrics saved to {metrics_path}" + Style.RESET_ALL)
    return metrics_path


def save_results_json(params: dict, metrics: dict, run_dir: str) -> None:
    """
    Save parameters and metrics as JSON files in run_dir.
    """
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(Fore.GREEN + "✅ Params and metrics JSON saved." + Style.RESET_ALL)


def load_model_from_run(run_dir: str) -> keras.Model:
    """
    Load a model from a specific run directory.
    """
    model_path = os.path.join(run_dir, "model.keras")
    if not os.path.exists(model_path):
        print(Fore.RED + f"❌ Model file not found in {run_dir}" + Style.RESET_ALL)
        return None

    print(Fore.BLUE + f"Loading model from {model_path}..." + Style.RESET_ALL)
    model = keras.models.load_model(model_path)
    print(Fore.GREEN + f"✅ Model loaded from {run_dir}" + Style.RESET_ALL)
    return model



################ USAGE DE REGISTERY ###########################
#run_dir = get_run_dir(model_name="model_name")

#save_model(model, run_dir)
#save_weights(model, run_dir)

#save_config(config, run_dir)
#save_metrics_csv(history.history, run_dir)
#save_results_json(params, metrics, run_dir)
