import pandas as pd
import numpy as np


def train_test_split_by_unit(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset by unit_id to avoid data leakage.

    Parameters:
        df: DataFrame with 'unit_id'
        test_size: proportion of units for test
        random_state: reproducibility

    Returns:
        train_df, test_df
    """

    # Lista de motores únicos
    unit_ids = df["unit_id"].unique()

    # Embaralhar
    np.random.seed(random_state)
    np.random.shuffle(unit_ids)

    # Definir corte
    split_idx = int(len(unit_ids) * (1 - test_size))

    train_units = unit_ids[:split_idx]
    test_units = unit_ids[split_idx:]

    # Filtrar dados
    train_df = df[df["unit_id"].isin(train_units)]
    test_df = df[df["unit_id"].isin(test_units)]

    return train_df, test_df

def split_features_target(df: pd.DataFrame):
    """
    Split features and target.
    """

    X = df.drop(columns=["RUL"])
    y = df["RUL"]

    return X, y