import os
from pathlib import Path
from typing import Optional
from google.cloud import storage
import tensorflow as tf
import pandas as pd
from params import *
from colorama import Fore, Style


image_list = pd.read_csv("data/Data_Entry_2017_v2020.csv")['Image Index'].tolist()

def fetch_png_images(
    image_list: list = image_list,
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

        image_dict[image_name] = tf.io.decode_image(img_bytes)

    print(f"✅ All images loaded, total: {len(image_dict)}")
    return image_dict

def store_preprocessed_images(
    df: pd.DataFrame,
    dataframe_name: str,
    gcp_project: str = GCP_PROJECT,
    bucket_name: str = BUCKET_NAME
) -> None:
    """
    Save preprocessed dataframe to GCS in the appropriate prefix based on compression level.
    Data is NOT stored locally.
    """
    # Extract compression level from dataframe_name (expects format: ..._<compression>)
    try:
        compression_level = int(dataframe_name.split('_')[-1])
    except Exception:
        print(Fore.RED + f"❌ Could not extract compression level from dataframe name: {dataframe_name}" + Style.RESET_ALL)
        return

    prefix = f"images_preprocessed_{compression_level}/"
    filename = f"{dataframe_name}.parquet"
    gcs_path = prefix + filename

    # Convert DataFrame to parquet in memory
    import io
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    # Upload to GCS
    client = storage.Client(project=gcp_project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_file(buffer, content_type='application/octet-stream')
    print(Fore.GREEN + f"✅ DataFrame {dataframe_name} saved to gs://{bucket_name}/{gcs_path}" + Style.RESET_ALL)
