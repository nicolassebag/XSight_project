import os
from pathlib import Path
from typing import Optional
from google.cloud import storage
import tensorflow as tf
import pandas as pd
from params import *
from colorama import Fore, Style
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from split_data import df_train
from typing import Tuple, Dict, Any, Optional


def get_stratified_sample(
    df: Optional[pd.DataFrame] = df_train,
    sample_size: Optional[int] = DATA_SIZE,
    label_columns: Optional[list] = PATHO_COLUMNS,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:

    # Ensure we have the required columns
    missing_cols = [col for col in label_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

    # Create a single split to get the sample
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=sample_size/len(df),
        random_state=random_state
    )

    # Get the indices for the sample
    for _, sample_idx in msss.split(df, df[label_columns]):
        return df.iloc[sample_idx].copy()

    return df


def fetch_png_images(
    image_df: pd.DataFrame = get_stratified_sample(),
    local_dir: str = '{LOCAL_DIR}',
    gcp_project: str = '{GCP_PROJECT_WAGON}',
    bucket_name: str = '{BUCKET_NAME}',
    prefix: str = '{PREFIX}',
    image_column: str = "Image Index"
) -> dict:
    
    """
    Retrieve PNG images from local folder if available, otherwise from GCS.
    Does NOT store downloaded images locally.
    Returns: {image_name: tf.io.decode_image(img_bytes) object}
    """

    local_path = Path(local_dir)
    image_dict = {}

    client = storage.Client(project=gcp_project)
    bucket = client.bucket(bucket_name)

    for image_name in image_df[image_column]:
        file_path = local_path / image_name

        if file_path.is_file():
            print(Fore.BLUE + f"Load {image_name} from local cache..." + Style.RESET_ALL)
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            print(f"✅ {image_name} loaded from local cache")
        else:
            print(Fore.BLUE + f"Download {image_name} from GCS..." + Style.RESET_ALL)
            blob = bucket.blob(f"{prefix}{image_name}")
            img_bytes = blob.download_as_bytes()
            print(f"✅ {image_name} downloaded from GCS")

        image_dict[image_name] = tf.io.decode_image(img_bytes)

    print(f"✅ All images loaded, total: {len(image_dict)}")
    return image_dict
