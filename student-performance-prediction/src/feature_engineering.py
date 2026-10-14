"""Feature engineering for student performance prediction."""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw academic indicators."""
    featured = df.copy()

    # Average of internal assessments
    featured["internal_avg"] = (
        featured["internal_assessment_1"] + featured["internal_assessment_2"]
    ) / 2

    # Weighted academic progress score (assignments + internals + midterm)
    featured["academic_progress"] = (
        0.3 * featured["assignment_avg"]
        + 0.35 * featured["internal_avg"]
        + 0.35 * featured["midterm_score"]
    )

    # Engagement score combining attendance and study hours
    featured["engagement_score"] = (
        0.6 * featured["attendance_pct"] + 0.4 * (featured["study_hours_weekly"] / 30 * 100)
    )

    # Risk flag: low attendance AND low assignment average
    featured["at_risk"] = (
        (featured["attendance_pct"] < 70) & (featured["assignment_avg"] < 60)
    ).astype(int)

    # Parental education ordinal encoding
    education_map = {"High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
    featured["parental_education_level"] = featured["parental_education"].map(education_map)

    return featured


def get_feature_columns() -> list[str]:
    """Return feature column names used for model training."""
    return [
        "attendance_pct",
        "assignment_avg",
        "internal_assessment_1",
        "internal_assessment_2",
        "midterm_score",
        "study_hours_weekly",
        "extracurricular_hours",
        "previous_gpa",
        "internal_avg",
        "academic_progress",
        "engagement_score",
        "at_risk",
        "parental_education_level",
    ]


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Build sklearn preprocessor with scaling for numeric features."""
    return ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), feature_columns),
        ],
        remainder="drop",
    )
