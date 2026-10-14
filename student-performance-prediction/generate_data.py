"""Generate synthetic student performance dataset with intentional noise and missing values."""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N_STUDENTS = 500

FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Ananya", "Kabir", "Isha", "Vihaan", "Meera",
    "Arjun", "Sneha", "Dev", "Kavya", "Aditya", "Nisha", "Rahul", "Pooja",
    "Karan", "Divya", "Vikram", "Shreya", "Amit", "Neha", "Sanjay", "Ritu",
    "Manish", "Tanvi", "Nikhil", "Aisha", "Harsh", "Simran", "Yash", "Palak",
    "Deepak", "Anjali", "Suresh", "Kriti", "Varun", "Sakshi", "Gaurav", "Nidhi",
]

LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Gupta", "Reddy", "Iyer", "Mehta",
    "Verma", "Nair", "Joshi", "Chopra", "Malhotra", "Rao", "Desai", "Bhat",
    "Kapoor", "Saxena", "Pillai", "Khanna", "Agarwal", "Mishra", "Das", "Shah",
    "Banerjee", "Kulkarni", "Menon", "Tiwari", "Chauhan", "Pandey",
]


def _generate_names(n: int) -> list[str]:
    rng = np.random.default_rng(42)
    first = rng.choice(FIRST_NAMES, n)
    last = rng.choice(LAST_NAMES, n)
    names = [f"{f} {l}" for f, l in zip(first, last)]

    # Ensure unique names where possible
    seen = set()
    for i, name in enumerate(names):
        if name in seen:
            names[i] = f"{name} {i + 1}"
        seen.add(names[i])
    return names


def generate_student_data(n: int = N_STUDENTS) -> pd.DataFrame:
    attendance = np.clip(np.random.normal(82, 12, n), 40, 100)
    assignment_avg = np.clip(np.random.normal(72, 15, n), 30, 100)
    internal_1 = np.clip(np.random.normal(68, 18, n), 20, 100)
    internal_2 = np.clip(np.random.normal(70, 16, n), 25, 100)
    midterm = np.clip(np.random.normal(65, 20, n), 15, 100)
    study_hours = np.clip(np.random.normal(12, 5, n), 2, 30)
    extracurricular = np.clip(np.random.exponential(3, n), 0, 15)
    previous_gpa = np.clip(np.random.normal(3.0, 0.6, n), 1.5, 4.0)

    parental_education = np.random.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        n,
        p=[0.35, 0.35, 0.22, 0.08],
    )

    # Final score is a weighted combination of features plus noise
    final_score = (
        0.25 * attendance
        + 0.20 * assignment_avg
        + 0.15 * internal_1
        + 0.15 * internal_2
        + 0.15 * midterm
        + 0.05 * study_hours * 3
        + 0.05 * previous_gpa * 20
        + np.random.normal(0, 5, n)
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
            "name": _generate_names(n),
            "attendance_pct": attendance.round(1),
            "assignment_avg": assignment_avg.round(1),
            "internal_assessment_1": internal_1.round(1),
            "internal_assessment_2": internal_2.round(1),
            "midterm_score": midterm.round(1),
            "study_hours_weekly": study_hours.round(1),
            "extracurricular_hours": extracurricular.round(1),
            "previous_gpa": previous_gpa.round(2),
            "parental_education": parental_education,
            "final_score": final_score.round(1),
        }
    )

    # Introduce missing values (~3% random)
    for col in ["name", "attendance_pct", "assignment_avg", "internal_assessment_1", "study_hours_weekly"]:
        mask = np.random.random(n) < 0.03
        df.loc[mask, col] = np.nan

    # Introduce a few duplicate rows
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)

    # Introduce outliers in midterm scores
    outlier_idx = np.random.choice(n, size=5, replace=False)
    df.loc[outlier_idx, "midterm_score"] = [5, 98, 3, 99, 7]

    return df


if __name__ == "__main__":
    from src.data_cleaning import clean_student_data

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = generate_student_data()
    raw_path = output_dir / "student_performance.csv"
    raw.to_csv(raw_path, index=False)

    clean = clean_student_data(raw)
    clean_path = output_dir / "student_performance_clean.csv"
    clean.to_csv(clean_path, index=False)

    sample_path = output_dir / "sample_students.csv"
    clean.head(20).to_csv(sample_path, index=False)

    print(f"Raw dataset:   {len(raw)} records -> {raw_path}")
    print(f"Clean dataset: {len(clean)} records -> {clean_path}")
    print(f"Sample set:    20 records -> {sample_path}")
