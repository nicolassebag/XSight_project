import pandas as pd
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """Load a csv file into a Dataframe."""
    return pd.read_csv(filepath)

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not needed for model training."""
    columns_to_drop = [
        'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 'Follow-up #', 'Patient ID'
    ]
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

def encode_labels(df: pd.DataFrame, label_column: str = 'Finding Labels') -> pd.DataFrame:
    """OneHotEncode the labels and drop the original column."""
    if label_column in df.columns:
        dummies = df[label_column].str.get_dummies(sep='|')
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[label_column])
    return df

                ############################################################

def preprocess_1(filepath: str) -> pd.DataFrame:
    """Complete preprocessing the pipeline."""
    df = load_data(filepath)
    df = drop_unnecessary_columns(df)
    df = encode_labels(df)
    return df
