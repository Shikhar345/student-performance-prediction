"""Streamlit app for Student Performance Prediction."""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import clean_student_data
from src.evaluation import compare_models, evaluate_model
from src.feature_engineering import (
    build_preprocessor,
    engineer_features,
    get_feature_columns,
)
from src.model_training import build_pipeline, get_models
from src.visualization import (
    build_prediction_comparison_table,
    get_correlation_insights,
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_model_comparison,
)


def show_figure(fig):
    """Display matplotlib figure in Streamlit and close it."""
    st.pyplot(fig)
    plt.close(fig)


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        from generate_data import generate_student_data

        data_path.parent.mkdir(parents=True, exist_ok=True)
        generate_student_data().to_csv(data_path, index=False)
    return pd.read_csv(data_path)


@st.cache_resource
def load_model_and_insights():
    data_path = PROJECT_ROOT / "data" / "student_performance.csv"
    raw_df = load_data(data_path)
    cleaned_df = clean_student_data(raw_df)
    featured_df = engineer_features(cleaned_df)
    feature_cols = get_feature_columns()

    X = featured_df[feature_cols]
    y = featured_df["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(feature_cols)
    results = []
    best_pipeline = None
    best_name = "Linear Regression"
    best_r2 = -float("inf")

    for name, model in get_models().items():
        pipeline = build_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)
        y_pred_model = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred_model, name)
        results.append(metrics)
        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_pipeline = pipeline
            best_name = name

    y_pred = best_pipeline.predict(X_test)
    best_metrics = evaluate_model(y_test, y_pred, best_name)
    comparison_df = compare_models(results)
    comparison_table = build_prediction_comparison_table(
        featured_df, X_test.index, y_test, y_pred
    )
    correlation_insights = get_correlation_insights(featured_df)

    return (
        best_pipeline,
        feature_cols,
        featured_df,
        y_test,
        y_pred,
        best_metrics,
        comparison_df,
        comparison_table,
        correlation_insights,
    )


def build_input_row(
    name: str,
    attendance_pct: float,
    assignment_avg: float,
    internal_1: float,
    internal_2: float,
    midterm_score: float,
    study_hours_weekly: float,
    extracurricular_hours: float,
    previous_gpa: float,
    parental_education: str,
) -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "student_id": "FORM001",
                "name": name or "Unknown Student",
                "attendance_pct": attendance_pct,
                "assignment_avg": assignment_avg,
                "internal_assessment_1": internal_1,
                "internal_assessment_2": internal_2,
                "midterm_score": midterm_score,
                "study_hours_weekly": study_hours_weekly,
                "extracurricular_hours": extracurricular_hours,
                "previous_gpa": previous_gpa,
                "parental_education": parental_education,
            }
        ]
    )
    return engineer_features(raw)


def score_label(score: float) -> tuple[str, str]:
    if score >= 90:
        return "Excellent", "🟢"
    if score >= 75:
        return "Good", "🔵"
    if score >= 60:
        return "Average", "🟡"
    if score >= 40:
        return "Below Average", "🟠"
    return "At Risk", "🔴"


def render_predict_section(pipeline, feature_cols):
    st.header("1. Predict Student Performance")

    with st.form("prediction_form"):
        student_name = st.text_input("Student Name", placeholder="Enter full name")

        col1, col2 = st.columns(2)

        with col1:
            attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0, 0.5)
            assignment_avg = st.slider("Assignment Average", 0.0, 100.0, 72.0, 0.5)
            internal_1 = st.slider("Internal Assessment 1", 0.0, 100.0, 68.0, 0.5)
            internal_2 = st.slider("Internal Assessment 2", 0.0, 100.0, 70.0, 0.5)
            midterm = st.slider("Midterm Score", 0.0, 100.0, 65.0, 0.5)

        with col2:
            study_hours = st.slider("Study Hours (per week)", 0.0, 30.0, 12.0, 0.5)
            extracurricular = st.slider("Extracurricular Hours (per week)", 0.0, 15.0, 3.0, 0.5)
            previous_gpa = st.slider("Previous GPA", 1.0, 4.0, 3.0, 0.01)
            parental_education = st.selectbox(
                "Parental Education",
                ["High School", "Bachelor", "Master", "PhD"],
            )

        submitted = st.form_submit_button("Predict Performance", type="primary", use_container_width=True)

    if submitted:
        featured = build_input_row(
            student_name,
            attendance,
            assignment_avg,
            internal_1,
            internal_2,
            midterm,
            study_hours,
            extracurricular,
            previous_gpa,
            parental_education,
        )

        prediction = float(pipeline.predict(featured[feature_cols])[0])
        prediction = max(0.0, min(100.0, prediction))
        label, icon = score_label(prediction)
        display_name = student_name.strip() or "Unknown Student"

        st.success(f"Prediction complete for **{display_name}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Final Score", f"{prediction:.1f}")
        m2.metric("Performance Level", f"{icon} {label}")
        m3.metric("At-Risk Flag", "Yes" if featured["at_risk"].iloc[0] else "No")
        m4.metric("Engagement Score", f"{featured['engagement_score'].iloc[0]:.1f}")

        st.progress(prediction / 100)


def render_model_section(comparison_df, metrics):
    st.header("2. Model Evaluation")
    st.caption("Comparison of regression models on the test set.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model R²", f"{metrics['R2']:.3f}")
    c2.metric("Best Model RMSE", f"{metrics['RMSE']:.2f}")
    c3.metric("Best Model MAE", f"{metrics['MAE']:.2f}")

    show_figure(plot_model_comparison(comparison_df))
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def render_actual_vs_predicted_section(y_test, y_pred, metrics, comparison_table):
    st.header("3. Actual vs Predicted Scores")
    st.caption("Each point represents one student from the 20% test set.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{metrics['R2']:.3f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    c3.metric("MAE", f"{metrics['MAE']:.2f}")
    c4.metric("Correlation", f"{pd.Series(y_test).corr(pd.Series(y_pred)):.3f}")

    show_figure(plot_actual_vs_predicted(y_test, y_pred))

    st.subheader("Student-wise Actual vs Predicted")
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)


def render_correlation_section(featured_df, correlation_insights):
    st.header("4. Correlation Insights")
    st.caption("How academic indicators relate to the final score.")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        show_figure(plot_correlation_heatmap(featured_df))

    with col_right:
        st.markdown("**Top correlations with final score**")
        for insight in correlation_insights["insight"]:
            st.markdown(f"- {insight}")

        top_pos = correlation_insights[correlation_insights["correlation"] > 0].iloc[0]
        top_feature = top_pos["feature"].replace("_", " ").title()
        st.info(
            f"Higher **{top_feature}** values tend to align with better final scores "
            f"(r = {top_pos['correlation']:.2f}). Monitor attendance, assignments, "
            "and internal assessments as early warning signals."
        )

        st.markdown("**Correlation table**")
        st.dataframe(
            correlation_insights[["feature", "correlation"]],
            use_container_width=True,
            hide_index=True,
        )


def main():
    st.set_page_config(
        page_title="Student Performance Predictor",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 Student Performance Prediction System")
    st.caption("Predict final academic score · analyze model performance and correlations")

    (
        pipeline,
        feature_cols,
        featured_df,
        y_test,
        y_pred,
        metrics,
        comparison_df,
        comparison_table,
        correlation_insights,
    ) = load_model_and_insights()

    with st.sidebar:
        st.header("Navigation")
        st.markdown(
            """
            1. Predict Performance  
            2. Model Evaluation  
            3. Actual vs Predicted  
            4. Correlation Insights
            """
        )
        st.divider()
        st.subheader("Score Guide")
        st.markdown(
            """
            | Range | Level |
            |-------|-------|
            | 90–100 | Excellent |
            | 75–89 | Good |
            | 60–74 | Average |
            | 40–59 | Below Average |
            | 0–39 | At Risk |
            """
        )

    render_predict_section(pipeline, feature_cols)
    st.divider()

    render_model_section(comparison_df, metrics)
    st.divider()

    render_actual_vs_predicted_section(y_test, y_pred, metrics, comparison_table)
    st.divider()

    render_correlation_section(featured_df, correlation_insights)


if __name__ == "__main__":
    main()
