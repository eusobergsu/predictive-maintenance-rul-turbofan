import pandas as pd
from pathlib import Path

COLUMN_NAMES = (
    ["unit_id", "cycles"]
    + [f"op_settings_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_single_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load a single turbofan dataset file and assing proper column names.
    """
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df.columns = COLUMN_NAMES
    return df

def load_all_dataset(data_dir: Path) -> dict:
    """
    Load all FD datasets into a dictionary

    Returns:
         dict: { 'FD001': df, FD002: df, ...}
    """
    datasets = {}

    for i in range(1, 5):
        file_name = f"train_FD00{i}.txt"
        file_path = data_dir / file_name

        df = load_single_dataset(file_path)
        datasets[f"FD00{i}"] = df

    return datasets

def validate_dataset(df: pd.DataFrame) -> None:
    """"
    Basic validation checks
    """
    assert not df.empty, "Dataset is empty"
    assert "unit_id" in df.columns, "Missing unit_id"
    assert "cycles" in df.columns, "Missing cycles"



