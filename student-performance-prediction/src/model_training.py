"""Model training for student performance prediction."""

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline


def get_models() -> dict:
    """Return dictionary of regression models to compare."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }


def build_pipeline(preprocessor, model) -> Pipeline:
    """Combine preprocessor and model into a single pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
