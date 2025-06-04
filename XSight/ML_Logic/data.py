

from google.cloud import storage
import tensorflow as tf


def fetch_images_to_memory(
    gcp_project: str,
    bucket_name: str,
    prefix: str,
    image_list: list
) -> dict:
    """
    Fetch images from GCS into memory as BytesIO objects
    Returns: {image_name: tf.io.decode_image(img_bytes) object}
    """
    client = storage.Client(project=gcp_project)
    bucket = client.bucket(bucket_name)

    image_dict = {}
    for image_name in image_list:
        blob = bucket.blob(f"{prefix}{image_name}")
        img_bytes = blob.download_as_bytes()
        image_dict[image_name] = tf.io.decode_image(img_bytes,channels=1)

    return image_dict

# Usage example
images = fetch_images_to_memory(
    gcp_project="xsight-project-le-wagon",
    bucket_name="cxr8_images_bucket",
    prefix="all_images/",
    image_list=["00000001_000.png", "00000002_000.png"]
)
