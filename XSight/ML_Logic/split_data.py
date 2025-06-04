import pandas as pd
import numpy as np
from typing import Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from XSight.params import PATHO_COLUMNS, PATIENT_ID_COL, TEST_SIZE, VAL_SIZE, RANDOM_STATE

def split_data(
    data_encoded: pd.DataFrame,
    patient_id_col: str = 'patient ID',
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a 2-stage stratified split based on patient-level pathology labels.

    Returns:
        Tuple of (df_train, df_val, df_test)
    """

    # Define the label columns internally
    label_columns = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
        'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening',
        'Pneumonia', 'Pneumothorax'
    ]

    # Step 1: Patient-level pathology aggregation
    df_patho = data_encoded.groupby(patient_id_col)[label_columns].max().reset_index()

    X_patho = df_patho[[patient_id_col]].values
    y_patho = df_patho[label_columns].values

    # Step 2: First split (train vs temp)
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, temp_idx = next(msss1.split(X_patho, y_patho))

    train_ids = df_patho.iloc[train_idx][patient_id_col].values
    temp_ids  = df_patho.iloc[temp_idx][patient_id_col].values

    # Step 3: Second split (val vs test)
    df_temp = df_patho[df_patho[patient_id_col].isin(temp_ids)].reset_index(drop=True)
    X_temp = df_temp[[patient_id_col]].values
    y_temp = df_temp[label_columns].values

    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    val_idx, test_idx = next(msss2.split(X_temp, y_temp))

    val_ids  = df_temp.iloc[val_idx][patient_id_col].values
    test_ids = df_temp.iloc[test_idx][patient_id_col].values

    # Step 4: Final dataset splits
    df_train = data_encoded[data_encoded[patient_id_col].isin(train_ids)].reset_index(drop=True)
    df_val   = data_encoded[data_encoded[patient_id_col].isin(val_ids)].reset_index(drop=True)
    df_test  = data_encoded[data_encoded[patient_id_col].isin(test_ids)].reset_index(drop=True)

    return df_train, df_val, df_test
