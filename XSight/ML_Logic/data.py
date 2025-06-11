import os
from pathlib import Path
from typing import Optional
from google.cloud import storage
import tensorflow as tf
import pandas as pd
from XSight.params import *
from colorama import Fore, Style

IMAGE_LIST = pd.read_csv(IMAGE_LIST_PATH)['Image Index'].tolist()

def fetch_png_images(
    image_list: list = IMAGE_LIST,
    local_dir: str = LOCAL_DIR,
    gcp_project: str = GCP_PROJECT,
    bucket_name: str = BUCKET_NAME,
    prefix: str = PREFIX,
) -> dict:
    """
    Retrieve PNG images from local folder if available, otherwise from GCS.
    Does NOT store downloaded images locally.
    Returns: {image_name: tf.io.decode_image(img_bytes) object}.
    Default values are taken from environment variables if not provided.
    """
    local_path = Path(local_dir)
    image_dict = {}

    client = storage.Client(project=gcp_project)
    bucket = client.bucket(bucket_name)

    for image_name in image_list:
        file_path = local_path / image_name

        if file_path.is_file():
            print(Fore.BLUE + f"Load {image_name} from local cache..." + Style.RESET_ALL)
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            print(f"✅ {image_name} loaded from local cache")
        else:
            print(Fore.BLUE + f"Download {image_name} from GCS..." + Style.RESET_ALL)
            blob = bucket.blob(f"{prefix}{image_name}")
            try:
                img_bytes = blob.download_as_bytes()
                print(f"✅ {image_name} downloaded from GCS")
            except Exception as e:
                print(Fore.RED + f"❌ Failed to download {image_name} from GCS: {e}" + Style.RESET_ALL)
                image_dict[image_name] = None
                continue

        image_dict[image_name] = tf.io.decode_image(img_bytes,channels=1)

    print(f"✅ All images loaded, total: {len(image_dict)}")
    return image_dict

def upload_image_folder_to_gcs(
    local_folder: str,
    bucket_name: str = BUCKET_NAME,
    gcp_project: str = GCP_PROJECT,
    prefix: str = None,
) -> None:
    """
    Upload all files from a local folder (e.g., images_64x64) to a GCS bucket under the appropriate prefix.
    The prefix is determined by the folder name (e.g., images_preprocessed_64/ for images_64x64).
    Files are uploaded to GCS with their relative paths preserved.
    """
    # Infer compression level from folder name
    folder_name = os.path.basename(os.path.normpath(local_folder))
    # Example: images_64x64 -> 64
    try:
            compression_level = [int(s) for s in folder_name.split("_") if s.isdigit()][0]
    except Exception:
        print(Fore.RED + f"❌ Could not extract compression level from folder name: {folder_name}" + Style.RESET_ALL)
        return

    gcs_prefix = prefix if prefix is not None else f"images_preprocessed_{compression_level}/"

    # Gather all PNG files in the folder (non-recursive, or use rglob for recursive)
    local_path = Path(local_folder)
    file_paths = list(local_path.glob("*.png"))

    if not file_paths:
        print(Fore.YELLOW + f"No PNG files found in {local_folder}." + Style.RESET_ALL)
        return

    print(Fore.BLUE + f"Found {len(file_paths)} PNG files in {local_folder}. Beginning upload..." + Style.RESET_ALL)

    storage_client = storage.Client(project=gcp_project)
    bucket = storage_client.bucket(bucket_name)

    for file_path in file_paths:
        blob_name = f"{gcs_prefix}{file_path.name}"
        blob = bucket.blob(blob_name)
        try:
            blob.upload_from_filename(str(file_path))
            print(Fore.GREEN + f"✅ Uploaded {file_path.name} to gs://{bucket_name}/{blob_name}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"❌ Failed to upload {file_path.name}: {e}" + Style.RESET_ALL)

    print(Fore.BLUE + f"✅ All uploads complete for {local_folder}." + Style.RESET_ALL)
