"""Model evaluation metrics and visualization helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> dict:
    """Compute regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "model": model_name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Create comparison DataFrame sorted by R2 score."""
    df = pd.DataFrame(results)
    return df.sort_values("R2", ascending=False).reset_index(drop=True)
