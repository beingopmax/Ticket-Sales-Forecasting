"""
evaluate.py

Responsibility:
    Standalone evaluation of already-trained model pipelines. This
    module contains NO training, splitting, or preprocessing-fitting
    logic of its own -- it only:
        1. Loads saved pipelines from models/*.pkl
           (preprocessing + model already fitted, bundled together).
        2. Loads the held-out test set from
           data/processed/test_set.csv (an artifact saved by
           train.py at split time -- NOT re-derived here, so this
           module never risks evaluating a model on rows it was
           trained on).
        3. Generates predictions and scores them: MAE, RMSE, R2.
        4. Writes reports/model_comparison.csv.
        5. Generates comparison visualizations.

    This can be re-run any time, independently of train.py, as long
    as models/*.pkl and data/processed/test_set.csv exist on disk.
    It imports only shared column-name constants from preprocessing.py
    (the single source of truth for feature/target names) -- it does
    NOT import or call anything from train.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import ALL_MODEL_FEATURES, TARGET_COLUMN

MODELS_DIR = Path("models")
TEST_SET_PATH = Path("data/processed/test_set.csv")
REPORTS_DIR = Path("reports")
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"
COMPARISON_CHART_PATH = REPORTS_DIR / "model_comparison_chart.png"
PRED_VS_ACTUAL_PATH = REPORTS_DIR / "best_model_predicted_vs_actual.png"

# Files in models/ to evaluate. best_model_pipeline.pkl is deliberately
# excluded here -- it's a copy of whichever of these three already won,
# so scoring it too would just duplicate one of these rows.
MODEL_FILES = {
    "Linear Regression": "linear_regression_pipeline.pkl",
    "Random Forest": "random_forest_pipeline.pkl",
    "XGBoost": "xgboost_pipeline.pkl",
}


def load_test_set() -> tuple:
    """Load the held-out test set artifact saved by train.py."""
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(
            f"{TEST_SET_PATH} not found. Run train.py first -- it saves "
            f"the held-out test set as an artifact so evaluate.py never "
            f"has to re-derive (and risk mismatching) the train/test split."
        )
    df = pd.read_csv(TEST_SET_PATH)
    X_test = df[ALL_MODEL_FEATURES]
    y_test = df[TARGET_COLUMN]
    return X_test, y_test


def load_pipelines() -> dict:
    """Load all saved model pipelines from models/."""
    pipelines = {}
    for model_name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {model_name}.")
            continue
        pipelines[model_name] = joblib.load(path)
    if not pipelines:
        raise FileNotFoundError(
            f"No model pipelines found in {MODELS_DIR}. Run train.py first."
        )
    return pipelines


def score_pipeline(pipeline, X_test, y_test) -> dict:
    """Generate predictions and compute MAE, RMSE, R2."""
    preds = pipeline.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
        "predictions": preds,
    }


def evaluate_all(pipelines: dict, X_test, y_test) -> tuple:
    """
    Score every loaded pipeline. Returns (comparison_df, predictions_dict)
    where predictions_dict maps model_name -> np.array of predictions
    (kept around for the predicted-vs-actual plot, not written to CSV).
    """
    rows = []
    predictions = {}
    for model_name, pipeline in pipelines.items():
        result = score_pipeline(pipeline, X_test, y_test)
        predictions[model_name] = result.pop("predictions")
        rows.append({"model": model_name, **result})

    comparison_df = pd.DataFrame(rows).rename(
        columns={"mae": "test_mae", "rmse": "test_rmse", "r2": "test_r2"}
    )
    comparison_df = comparison_df.sort_values("test_rmse").reset_index(drop=True)
    return comparison_df, predictions


def plot_metric_comparison(comparison_df: pd.DataFrame, output_path: Path):
    """Bar chart comparing MAE, RMSE, and R2 across all evaluated models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("test_mae", "MAE (lower is better)"),
               ("test_rmse", "RMSE (lower is better)"),
               ("test_r2", "R\u00b2 (higher is better)")]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, (col, title) in zip(axes, metrics):
        ax.bar(comparison_df["model"], comparison_df[col], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(comparison_df[col]):
            ax.text(i, v, f"{v:.2f}" if col != "test_r2" else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(y_test, best_model_name: str, predictions: np.ndarray,
                              output_path: Path):
    """Scatter plot of predicted vs actual tickets_sold for the best model."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, predictions, alpha=0.15, s=8, color="#4C72B0")

    max_val = max(y_test.max(), predictions.max())
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", label="Perfect prediction")

    ax.set_xlabel("Actual tickets_sold")
    ax.set_ylabel("Predicted tickets_sold")
    ax.set_title(f"Predicted vs Actual -- {best_model_name} (test set)")
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def merge_with_existing_report(fresh_df: pd.DataFrame) -> pd.DataFrame:
    """
    If reports/model_comparison.csv already exists (e.g. from a prior
    train.py run) and contains cv_* / best_params columns, preserve
    those alongside the freshly computed test metrics -- this enriches
    the report with training provenance when available, while
    evaluate.py still works fully standalone (test-metrics-only) when
    it isn't, e.g. if only models/*.pkl and the test set were copied
    to a fresh environment without train.py ever having run there.
    """
    if not COMPARISON_CSV_PATH.exists():
        return fresh_df

    try:
        existing_df = pd.read_csv(COMPARISON_CSV_PATH)
    except Exception:
        return fresh_df

    provenance_cols = [c for c in existing_df.columns
                        if c.startswith("cv_") or c == "best_params"]
    if not provenance_cols or "model" not in existing_df.columns:
        return fresh_df

    merged = fresh_df.merge(existing_df[["model"] + provenance_cols], on="model", how="left")

    ordered_cols = (["model"] +
                    [c for c in provenance_cols if c.startswith("cv_")] +
                    ["test_mae", "test_rmse", "test_r2"] +
                    (["best_params"] if "best_params" in provenance_cols else []))
    return merged[[c for c in ordered_cols if c in merged.columns]]


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_set()
    print(f"Loaded held-out test set: {X_test.shape[0]} rows (from {TEST_SET_PATH})")

    pipelines = load_pipelines()
    print(f"Loaded {len(pipelines)} pipeline(s): {list(pipelines.keys())}\n")

    comparison_df, predictions = evaluate_all(pipelines, X_test, y_test)
    comparison_df = merge_with_existing_report(comparison_df)

    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"Saved: {COMPARISON_CSV_PATH}")
    print(comparison_df.to_string(index=False))

    plot_metric_comparison(comparison_df, COMPARISON_CHART_PATH)
    print(f"Saved: {COMPARISON_CHART_PATH}")

    best_model_name = comparison_df.iloc[0]["model"]
    plot_predicted_vs_actual(
        y_test, best_model_name, predictions[best_model_name], PRED_VS_ACTUAL_PATH
    )
    print(f"Saved: {PRED_VS_ACTUAL_PATH}")

    best_row = comparison_df.iloc[0]
    print(f"\nBest model on this test set: {best_row['model']} "
          f"(RMSE={best_row['test_rmse']:.2f}, R2={best_row['test_r2']:.4f})")


if __name__ == "__main__":
    main()
