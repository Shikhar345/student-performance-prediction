"""Data cleaning utilities for student performance dataset."""

import pandas as pd


def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw student data: remove duplicates, handle missing values and outliers."""
    cleaned = df.copy()

    # Remove duplicate student records
    cleaned = cleaned.drop_duplicates(subset=["student_id"], keep="first")

    # Validate score ranges (0-100)
    score_cols = [
        "attendance_pct",
        "assignment_avg",
        "internal_assessment_1",
        "internal_assessment_2",
        "midterm_score",
        "final_score",
    ]
    for col in score_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].clip(0, 100)

    # Cap GPA to valid range
    if "previous_gpa" in cleaned.columns:
        cleaned["previous_gpa"] = cleaned["previous_gpa"].clip(1.0, 4.0)

    # Handle outliers in midterm using IQR method
    if "midterm_score" in cleaned.columns:
        q1 = cleaned["midterm_score"].quantile(0.25)
        q3 = cleaned["midterm_score"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned["midterm_score"] = cleaned["midterm_score"].clip(lower, upper)

    # Impute missing numeric values with median
    numeric_cols = cleaned.select_dtypes(include="number").columns
    for col in numeric_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # Impute missing categorical values with mode
    cat_cols = cleaned.select_dtypes(include="object").columns
    for col in cat_cols:
        if col not in {"student_id", "name"} and cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    if "name" in cleaned.columns and cleaned["name"].isna().any():
        cleaned["name"] = cleaned["name"].fillna("Unknown Student")

    for col in score_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].round(1)
    if "study_hours_weekly" in cleaned.columns:
        cleaned["study_hours_weekly"] = cleaned["study_hours_weekly"].round(1)
    if "extracurricular_hours" in cleaned.columns:
        cleaned["extracurricular_hours"] = cleaned["extracurricular_hours"].round(1)
    if "previous_gpa" in cleaned.columns:
        cleaned["previous_gpa"] = cleaned["previous_gpa"].round(2)

    return cleaned.reset_index(drop=True)
