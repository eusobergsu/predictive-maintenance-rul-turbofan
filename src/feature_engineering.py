import pandas as pd

def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure time ordering for each unit.
    """
    return df.sort_values(by=["unit_id", "cycles"])

def add_rolling_features(df, window=5):
    sensor_cols = [col for col in df.columns if "sensor_" in col]

    rolling_features = {}

    for col in sensor_cols:
        rolling_features[f"{col}_mean"] = (
            df.groupby("unit_id")[col]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        rolling_features[f"{col}_std"] = (
            df.groupby("unit_id")[col]
            .rolling(window)
            .std()
            .reset_index(level=0, drop=True)
        )

    rolling_df = pd.DataFrame(rolling_features)

    df = pd.concat([df, rolling_df], axis=1)

    return df

def add_diff_features(df):
    sensor_cols = [col for col in df.columns if "sensor_" in col]

    diff_features = {}

    for col in sensor_cols:
        diff_features[f"{col}_diff"] = df.groupby("unit_id")[col].diff()

    diff_df = pd.DataFrame(diff_features)

    df = pd.concat([df, diff_df], axis=1)

    return df

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.bfill()

def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline
    """

    df = sort_by_time(df)
    df = add_rolling_features(df)
    df = add_diff_features(df)
    df = handle_missing(df)

    return df

