"""Student Performance Prediction System - Main Pipeline."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from src.data_cleaning import clean_student_data
from src.evaluation import compare_models, evaluate_model
from src.feature_engineering import (
    build_preprocessor,
    engineer_features,
    get_feature_columns,
)
from src.model_training import build_pipeline, get_models
from src.visualization import (
    get_correlation_insights,
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_dataflow_diagram,
)


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        from generate_data import generate_student_data

        data_path.parent.mkdir(parents=True, exist_ok=True)
        generate_student_data().to_csv(data_path, index=False)
        print(f"Generated dataset at {data_path}")

    return pd.read_csv(data_path)


def plot_results(comparison_df: pd.DataFrame, output_dir: Path):
    """Save model comparison and prediction plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=comparison_df, x="model", y="R2", hue="model", ax=axes[0], palette="viridis", legend=False)
    axes[0].set_title("Model Comparison (R² Score)")
    axes[0].set_ylabel("R² Score")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(data=comparison_df, x="model", y="RMSE", hue="model", ax=axes[1], palette="magma", legend=False)
    axes[1].set_title("Model Comparison (RMSE)")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()
    print(f"Saved plot -> {output_dir / 'model_comparison.png'}")


def save_insight_plots(
    featured_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred,
    output_dir: Path,
    model_name: str,
):
    """Save dataflow diagram, actual vs predicted, and correlation plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    diagram = plot_dataflow_diagram()
    diagram.savefig(output_dir / "prediction_dataflow.png", dpi=150, bbox_inches="tight")
    plt.close(diagram)
    print(f"Saved diagram -> {output_dir / 'prediction_dataflow.png'}")

    fig = plot_actual_vs_predicted(y_test, y_pred, title=f"Actual vs Predicted ({model_name})")
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {output_dir / 'actual_vs_predicted.png'}")

    fig = plot_correlation_heatmap(featured_df)
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {output_dir / 'correlation_heatmap.png'}")

    insights = get_correlation_insights(featured_df)
    insights.to_csv(output_dir / "correlation_insights.csv", index=False)
    print(f"Saved insights -> {output_dir / 'correlation_insights.csv'}")



def main():
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "student_performance.csv"
    output_dir = project_root / "outputs"

    print("=" * 60)
    print("Student Performance Prediction System")
    print("=" * 60)

    # 1. Load Data
    print("\n[1] Loading data...")
    raw_df = load_data(data_path)
    print(f"    Raw records: {len(raw_df)}")

    # 2. Data Cleaning
    print("\n[2] Cleaning data...")
    cleaned_df = clean_student_data(raw_df)
    print(f"    Cleaned records: {len(cleaned_df)}")
    print(f"    Missing values remaining: {cleaned_df.isna().sum().sum()}")

    # 3. Feature Engineering
    print("\n[3] Engineering features...")
    featured_df = engineer_features(cleaned_df)
    feature_cols = get_feature_columns()
    print(f"    Features created: {len(feature_cols)}")

    X = featured_df[feature_cols]
    y = featured_df["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"    Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 4. Model Training & Evaluation
    print("\n[4] Training and evaluating models...")
    preprocessor = build_preprocessor(feature_cols)
    results = []
    best_model = None
    best_r2 = -float("inf")

    for name, model in get_models().items():
        pipeline = build_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, name)
        results.append(metrics)
        print(f"    {name}: R²={metrics['R2']}, RMSE={metrics['RMSE']}, MAE={metrics['MAE']}")

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model = (name, pipeline)

    comparison = compare_models(results)
    print("\n[5] Model Ranking:")
    print(comparison.to_string(index=False))

    # 5. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model_results.csv", index=False)
    plot_results(comparison, output_dir)

    if best_model:
        name, pipeline = best_model
        y_pred_best = pipeline.predict(X_test)
        save_insight_plots(featured_df, y_test, y_pred_best, output_dir, name)

    # 6. Sample prediction
    if best_model:
        name, pipeline = best_model
        sample = X_test.iloc[:3]
        predictions = pipeline.predict(sample)
        print(f"\n[6] Sample predictions using {name}:")
        for i, pred in enumerate(predictions):
            actual = y_test.iloc[i]
            print(f"    Student {i + 1}: Predicted={pred:.1f}, Actual={actual:.1f}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
