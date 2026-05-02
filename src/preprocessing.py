import pandas as pd


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for each unit.

    RUL = max_cycle_per_unit - current_cycle
    """

    max_cycles = df.groupby("unit_id")["cycles"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycles"]

    df = df.merge(max_cycles, on="unit_id", how="left")

    df["RUL"] = df["max_cycles"] - df["cycles"]

    df = df.drop(columns=["max_cycles"])
    
    return df

def apply_rul_cap(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """
    Apply upper cap to RUL values.

    This reduce nois in early life cycles.
    """
    df["RUL"] = df["RUL"].clip(upper=cap)

    return df

def preprocess_pipeline(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Compute RUL
    - Apply cap
    """

    df = compute_rul(df)
    df = apply_rul_cap(df, cap)

    return df

