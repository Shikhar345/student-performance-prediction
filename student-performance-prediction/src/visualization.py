"""Visualization helpers: actual vs predicted plots and correlation insights."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

__all__ = [
    "plot_model_comparison",
    "plot_actual_vs_predicted",
    "plot_correlation_heatmap",
    "get_correlation_insights",
    "build_prediction_comparison_table",
]

CORRELATION_FEATURES = [
    "attendance_pct",
    "assignment_avg",
    "internal_assessment_1",
    "internal_assessment_2",
    "midterm_score",
    "study_hours_weekly",
    "previous_gpa",
    "internal_avg",
    "academic_progress",
    "engagement_score",
    "final_score",
]

FLOW_STEPS = [
    ("Raw Dataset\n(CSV Input)", "#DBEAFE"),
    ("Data Cleaning\nDuplicates · Missing · Outliers", "#BFDBFE"),
    ("Feature Engineering\nDerived Indicators", "#93C5FD"),
    ("Train/Test Split\n80% / 20%", "#60A5FA"),
    ("Model Training\nLinear Regression", "#3B82F6"),
    ("Prediction\nFinal Score", "#2563EB"),
]


def plot_dataflow_diagram() -> plt.Figure:
    """Draw prediction pipeline dataflow using matplotlib."""
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    box_w, box_h = 1.9, 1.2
    y = 2.0
    x_start = 0.4
    gap = 0.35

    for i, (label, color) in enumerate(FLOW_STEPS):
        x = x_start + i * (box_w + gap)
        box = FancyBboxPatch(
            (x, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#1E3A8A",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y, label, ha="center", va="center", fontsize=9, color="#0F172A")

        if i > 0:
            prev_x = x_start + (i - 1) * (box_w + gap) + box_w
            arrow = FancyArrowPatch(
                (prev_x + 0.05, y),
                (x - 0.05, y),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.5,
                color="#334155",
            )
            ax.add_patch(arrow)

    eval_x = x_start + 4 * (box_w + gap) + box_w / 2
    eval_box = FancyBboxPatch(
        (eval_x - 0.85, 0.35),
        1.7,
        0.85,
        boxstyle="round,pad=0.06,rounding_size=0.06",
        linewidth=1.2,
        edgecolor="#1E3A8A",
        facecolor="#FEF3C7",
    )
    ax.add_patch(eval_box)
    ax.text(eval_x, 0.78, "Evaluation\nMAE · RMSE · R²", ha="center", va="center", fontsize=8)

    train_x = x_start + 4 * (box_w + gap) + box_w / 2
    eval_arrow = FancyArrowPatch(
        (train_x, y - box_h / 2 - 0.05),
        (eval_x, 1.25),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.2,
        color="#64748B",
        linestyle="dashed",
    )
    ax.add_patch(eval_arrow)

    infer_start_x = x_start + 2 * (box_w + gap) + box_w / 2
    infer_end_x = x_start + 5 * (box_w + gap) + box_w / 2
    ax.annotate(
        "",
        xy=(infer_end_x, 3.35),
        xytext=(infer_start_x, 3.35),
        arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.2, linestyle="dashed"),
    )
    ax.text(
        (infer_start_x + infer_end_x) / 2,
        3.55,
        "inference path",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#64748B",
    )

    ax.set_title("Student Performance Prediction — Dataflow", fontsize=13, pad=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """Bar charts comparing regression models by R² and RMSE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.barplot(
        data=comparison_df,
        x="model",
        y="R2",
        hue="model",
        ax=axes[0],
        palette="viridis",
        legend=False,
    )
    axes[0].set_title("Model Comparison (R² — Higher is Better)")
    axes[0].set_ylabel("R² Score")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=comparison_df,
        x="model",
        y="RMSE",
        hue="model",
        ax=axes[1],
        palette="magma",
        legend=False,
    )
    axes[1].set_title("Model Comparison (RMSE — Lower is Better)")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Final Score",
) -> plt.Figure:
    """Scatter plot of actual vs predicted scores with ideal line."""
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.65, edgecolors="white", linewidth=0.6, c="#2563EB")
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Final Score")
    ax.set_ylabel("Predicted Final Score")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}\nR²: {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of feature correlations with final score."""
    available = [c for c in CORRELATION_FEATURES if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    return fig


def get_correlation_insights(df: pd.DataFrame, target: str = "final_score", top_n: int = 5) -> pd.DataFrame:
    """Return top features correlated with the target and narrative insights."""
    numeric_df = df.select_dtypes(include="number")
    if target not in numeric_df.columns:
        return pd.DataFrame()

    corr_series = numeric_df.corr()[target].drop(target).sort_values(key=np.abs, ascending=False)
    top = corr_series.head(top_n).reset_index()
    top.columns = ["feature", "correlation"]

    def insight(row) -> str:
        strength = "strong" if abs(row["correlation"]) >= 0.5 else "moderate" if abs(row["correlation"]) >= 0.3 else "weak"
        direction = "positively" if row["correlation"] > 0 else "negatively"
        label = row["feature"].replace("_", " ").title()
        return f"**{label}** is {strength} {direction} correlated with final score (r = {row['correlation']:.2f})."

    top["insight"] = top.apply(insight, axis=1)
    return top


def build_prediction_comparison_table(
    featured_df: pd.DataFrame,
    test_indices: pd.Index,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Build table with name, actual, predicted, and error for test set."""
    meta_cols = [c for c in ["student_id", "name"] if c in featured_df.columns]
    table = featured_df.loc[test_indices, meta_cols].copy()
    table["actual_score"] = y_true.values.round(1)
    table["predicted_score"] = np.round(y_pred, 1)
    table["error"] = (table["predicted_score"] - table["actual_score"]).round(1)
    return table.sort_values("error", key=np.abs).reset_index(drop=True)
