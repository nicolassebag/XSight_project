import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from typing import Tuple, List

def split_data(
    df: pd.DataFrame,
    label_columns: List[str],
    id_column: str = 'patient ID',
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a two-stage Split to get train/val/test sets.
    """
    X = df[[id_column]].values
    y = df[label_columns].values

    # First split (train vs temp)
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, temp_idx = next(msss1.split(X, y))

    train_ids = df[id_column].iloc[train_idx].values
    temp_ids  = df[id_column].iloc[temp_idx].values

    df_temp = df[df[id_column].isin(temp_ids)].reset_index(drop=True)
    X_temp = df_temp[[id_column]].values
    y_temp = df_temp[label_columns].values

    # Second split (val vs test)
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    val_idx, test_idx = next(msss2.split(X_temp, y_temp))

    val_ids  = df_temp[id_column].iloc[val_idx].values
    test_ids = df_temp[id_column].iloc[test_idx].values

    # Final splits
    df_train = df[df[id_column].isin(train_ids)].reset_index(drop=True)
    df_val   = df[df[id_column].isin(val_ids)].reset_index(drop=True)
    df_test  = df[df[id_column].isin(test_ids)].reset_index(drop=True)

    return df_train, df_val, df_test
