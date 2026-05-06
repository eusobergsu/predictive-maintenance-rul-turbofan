from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from xgboost import XGBRegressor

def build_model_pipeline():
    """
    Build a pipeline with scaling and SGBoost model
    """

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ))
    ])

    return pipeline

def train_model(pipeline, X_train, y_train):
    """
    Train model pipeline
    """
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model pipeline
    """

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
    }