import os
import json
import glob
from tensorflow import keras
from colorama import Fore, Style

LOCAL_REGISTRY_PATH = "model_registry"
os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)

def save_model(model: keras.Model, model_name: str = "latest") -> str:
    """
    Save the model with a custom name.
    """
    model_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.keras")
    model.save(model_path)

    print(Fore.GREEN + f"✅ Model saved to {model_path}" + Style.RESET_ALL)
    return model_path

def save_results(params: dict, metrics: dict, model_name: str = "latest") -> None:
    """
    Save training parameters and evaluation metrics, including model name.
    """
    params_dir = os.path.join(LOCAL_REGISTRY_PATH, "params")
    metrics_dir = os.path.join(LOCAL_REGISTRY_PATH, "metrics")
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    params["model_name"] = model_name

    with open(os.path.join(params_dir, f"{model_name}.json"), "w") as f:
        json.dump(params, f, indent=2)

    with open(os.path.join(metrics_dir, f"{model_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(Fore.GREEN + f"✅ Params and metrics saved for model '{model_name}'" + Style.RESET_ALL)


def load_model(model_name: str = "latest") -> keras.Model:
    """
    Load a model by name.
    """
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{model_name}.keras")

    if not os.path.exists(model_path):
        print(Fore.RED + f"❌ No model found with name '{model_name}'" + Style.RESET_ALL)
        return None

    print(Fore.BLUE + f"Loading model '{model_name}' from {model_path}..." + Style.RESET_ALL)
    model = keras.models.load_model(model_path)

    print(Fore.GREEN + f"✅ Model '{model_name}' loaded successfully" + Style.RESET_ALL)
    return model


################ USAGE DE REGISTERY ###########################
#params = {
#    les params de ton model:
#    "epochs": 2
#    ...
#   }
#model_name='le_nom_de_ton_model'
#
#save_model(model, model_name=model_name)
#
#save_results(params, metrics, model_name=model_name)
#
#loaded_model = load_model(model_name=model_name)
