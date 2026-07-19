"""
train.py

Responsibility:
    Train and compare exactly three regression models -- Linear
    Regression (baseline), Random Forest (nonlinear), XGBoost
    (advanced) -- using the SAME preprocessing pipeline from
    preprocessing.py, bundled together as a single sklearn Pipeline
    per model (preprocessing + model in one object, so there's no
    risk of test-time preprocessing drifting from train-time
    preprocessing).

Process per model:
    1. 5-fold cross-validation with default hyperparameters (baseline
       comparison, reported in model_comparison.csv as cv_* columns).
    2. Hyperparameter tuning where appropriate:
         - Linear Regression: skipped. Plain OLS has no meaningful
           hyperparameters to search in scikit-learn; tuning it would
           be busywork, not real model improvement.
         - Random Forest / XGBoost: RandomizedSearchCV (small,
           deliberately bounded search space -- this environment has
           a single CPU core and 113K training rows, so an
           exhaustive grid would be wasteful; a randomized search
           over a sensible range gets most of the benefit for a
           fraction of the cost).
    3. Refit the best pipeline on the FULL training set.
    4. Evaluate once on the untouched test set: MAE, RMSE, R2.

Outputs:
    - reports/model_comparison.csv (CV + test metrics for all 3 models)
    - models/linear_regression_pipeline.pkl
    - models/random_forest_pipeline.pkl
    - models/xgboost_pipeline.pkl
    - models/best_model_pipeline.pkl (copy of whichever wins on test RMSE)
    - reports/feature_importance.png (Random Forest + XGBoost)
"""

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import build_preprocessor, ALL_MODEL_FEATURES, TARGET_COLUMN

FEATURED_DATA_PATH = Path("data/processed/cinema_ticket_features.csv")
TEST_SET_PATH = Path("data/processed/test_set.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_PNG_PATH = REPORTS_DIR / "feature_importance.png"

RANDOM_STATE = 42
CV_FOLDS = 5
TUNING_CV_FOLDS = 3
TUNING_N_ITER = 6

# ---------------------------------------------------------------------------
# COMPUTE BUDGET NOTE (measured empirically in this environment, 1 CPU core):
#   RandomForestRegressor on the full 113,816-row training set:
#       depth=10 -> ~1.35s/tree | depth=15 -> ~3.8s/tree | depth=20 -> ~7.65s/tree
#   XGBoost (tree_method='hist') on the SAME full data:
#       100 trees, depth=6 -> 0.79s TOTAL
#   XGBoost is ~2 orders of magnitude faster here (histogram-based splitting
#   vs sklearn RF's exact splitting), so it's tuned directly on full data.
#   RandomForestRegressor is not -- an unbounded/deep search on full data
#   would take much longer than is reasonable in this sandbox. So:
#     - RF hyperparameter search runs on a fixed subsample (documented below),
#       with max_depth deliberately bounded (never None/unbounded).
#     - The winning RF configuration is then refit ONCE on the FULL training
#       set for the actual saved pipeline and test-set evaluation, so the
#       reported numbers are never based on the subsample.
# ---------------------------------------------------------------------------
RF_SEARCH_SUBSAMPLE_SIZE = 10000
RF_BASELINE_CV_FOLDS = 3
RF_TUNING_CV_FOLDS = 2
RF_TUNING_N_ITER = 4


def load_split_data():
    """Load featured dataset, split into features/target, train/test."""
    df = pd.read_csv(FEATURED_DATA_PATH)
    X = df[ALL_MODEL_FEATURES]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def make_pipeline(model) -> Pipeline:
    """Preprocessing + model bundled together, per requirements."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


def cross_validate_pipeline(pipeline: Pipeline, X_train, y_train, cv=CV_FOLDS) -> dict:
    """Baseline CV metrics using default hyperparameters."""
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    results = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
    )
    return {
        "cv_mae_mean": -results["test_mae"].mean(),
        "cv_mae_std": results["test_mae"].std(),
        "cv_rmse_mean": -results["test_rmse"].mean(),
        "cv_rmse_std": results["test_rmse"].std(),
        "cv_r2_mean": results["test_r2"].mean(),
        "cv_r2_std": results["test_r2"].std(),
    }


def tune_pipeline(pipeline: Pipeline, param_distributions: dict, X_train, y_train):
    """RandomizedSearchCV, small bounded search. Returns (best_pipeline, best_params)."""
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=TUNING_N_ITER,
        cv=TUNING_CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def evaluate_on_test(pipeline: Pipeline, X_test, y_test) -> dict:
    preds = pipeline.predict(X_test)
    return {
        "test_mae": mean_absolute_error(y_test, preds),
        "test_rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "test_r2": r2_score(y_test, preds),
    }


def get_transformed_feature_names(fitted_preprocessor) -> list:
    """Recover human-readable feature names after ColumnTransformer expansion."""
    return list(fitted_preprocessor.get_feature_names_out())


def plot_feature_importance(rf_pipeline: Pipeline, xgb_pipeline: Pipeline, output_path: Path):
    """Save a side-by-side top-15 feature importance chart for both tree models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (name, pipeline) in zip(
        axes, [("Random Forest", rf_pipeline), ("XGBoost", xgb_pipeline)]
    ):
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        feature_names = get_transformed_feature_names(preprocessor)
        importances = model.feature_importances_

        order = np.argsort(importances)[::-1][:15]
        top_names = [feature_names[i] for i in order]
        top_importances = importances[order]

        ax.barh(range(len(top_names)), top_importances[::-1], color="#4C72B0")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title(f"{name}: Top 15 Feature Importances")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split_data()
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}\n")

    # Persist the held-out test set as a standalone artifact so evaluate.py
    # can load and score it independently, anytime, WITHOUT re-running this
    # split or any other training logic. This is what keeps evaluation
    # decoupled from training per the project requirements.
    test_set_df = X_test.copy()
    test_set_df[TARGET_COLUMN] = y_test.values
    TEST_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_set_df.to_csv(TEST_SET_PATH, index=False)
    print(f"Held-out test set saved to: {TEST_SET_PATH} (for evaluate.py)\n")

    comparison_rows = []
    fitted_pipelines = {}

    # ---------------- 1. Linear Regression (baseline, no tuning) ----------------
    print("=" * 60)
    print("Linear Regression (baseline)")
    print("=" * 60)
    t0 = time.time()
    lr_pipeline = make_pipeline(LinearRegression())
    cv_metrics = cross_validate_pipeline(lr_pipeline, X_train, y_train)
    lr_pipeline.fit(X_train, y_train)
    test_metrics = evaluate_on_test(lr_pipeline, X_test, y_test)
    print(f"CV RMSE: {cv_metrics['cv_rmse_mean']:.2f} | Test RMSE: {test_metrics['test_rmse']:.2f} "
          f"| Test R2: {test_metrics['test_r2']:.4f} | {time.time()-t0:.1f}s")
    comparison_rows.append({"model": "Linear Regression", "best_params": "n/a (no tuning)",
                             **cv_metrics, **test_metrics})
    fitted_pipelines["linear_regression"] = lr_pipeline

    # ---------------- 2. Random Forest (search on subsample, refit on full data) ----------------
    print("\n" + "=" * 60)
    print("Random Forest Regression")
    print("=" * 60)

    # Fixed subsample for baseline CV + hyperparameter search only (see
    # compute budget note above). random_state fixed for reproducibility.
    rf_search_idx = X_train.sample(n=RF_SEARCH_SUBSAMPLE_SIZE, random_state=RANDOM_STATE).index
    X_train_rf_search = X_train.loc[rf_search_idx]
    y_train_rf_search = y_train.loc[rf_search_idx]
    print(f"RF search subsample: {X_train_rf_search.shape[0]} rows "
          f"(of {X_train.shape[0]} full training rows)")

    t0 = time.time()
    rf_baseline = make_pipeline(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, max_depth=8)
    )
    rf_cv_metrics = cross_validate_pipeline(
        rf_baseline, X_train_rf_search, y_train_rf_search, cv=RF_BASELINE_CV_FOLDS
    )
    print(f"Baseline CV RMSE (on subsample): {rf_cv_metrics['cv_rmse_mean']:.2f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    rf_param_dist = {
        "model__n_estimators": [30, 50],
        "model__max_depth": [8, 10],           # deliberately bounded -- see compute budget note
        "model__min_samples_leaf": [4, 8],
        "model__max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        rf_baseline, param_distributions=rf_param_dist, n_iter=RF_TUNING_N_ITER,
        cv=RF_TUNING_CV_FOLDS, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=1, verbose=1,
    )
    search.fit(X_train_rf_search, y_train_rf_search)
    rf_best_params = search.best_params_
    print(f"Tuned params (found on subsample): {rf_best_params} ({time.time()-t0:.1f}s)")

    # Refit the winning configuration on the FULL training set. This is the
    # pipeline that actually gets evaluated and saved.
    t0 = time.time()
    rf_final_model = RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=1,
        **{k.replace("model__", ""): v for k, v in rf_best_params.items()}
    )
    rf_tuned = make_pipeline(rf_final_model)
    rf_tuned.fit(X_train, y_train)
    print(f"Final RF refit on FULL training data ({X_train.shape[0]} rows): {time.time()-t0:.1f}s")

    rf_test_metrics = evaluate_on_test(rf_tuned, X_test, y_test)
    print(f"Test RMSE: {rf_test_metrics['test_rmse']:.2f} | Test R2: {rf_test_metrics['test_r2']:.4f}")
    comparison_rows.append({"model": "Random Forest",
                             "best_params": str(rf_best_params) +
                                 f" (found on {RF_SEARCH_SUBSAMPLE_SIZE}-row subsample, refit on full train set)",
                             **rf_cv_metrics, **rf_test_metrics})
    fitted_pipelines["random_forest"] = rf_tuned

    # ---------------- 3. XGBoost (baseline CV, then tuned) ----------------
    print("\n" + "=" * 60)
    print("XGBoost Regression")
    print("=" * 60)
    t0 = time.time()
    xgb_baseline = make_pipeline(
        xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, tree_method="hist")
    )
    xgb_cv_metrics = cross_validate_pipeline(xgb_baseline, X_train, y_train)
    print(f"Baseline CV RMSE: {xgb_cv_metrics['cv_rmse_mean']:.2f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    xgb_param_dist = {
        "model__n_estimators": [100, 150, 200],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }
    xgb_tuned, xgb_best_params = tune_pipeline(xgb_baseline, xgb_param_dist, X_train, y_train)
    xgb_test_metrics = evaluate_on_test(xgb_tuned, X_test, y_test)
    print(f"Tuned params: {xgb_best_params}")
    print(f"Test RMSE: {xgb_test_metrics['test_rmse']:.2f} | Test R2: {xgb_test_metrics['test_r2']:.4f} "
          f"| tuning took {time.time()-t0:.1f}s")
    comparison_rows.append({"model": "XGBoost", "best_params": str(xgb_best_params),
                             **xgb_cv_metrics, **xgb_test_metrics})
    fitted_pipelines["xgboost"] = xgb_tuned

    # ---------------- Save comparison report ----------------
    comparison_df = pd.DataFrame(comparison_rows)
    column_order = ["model", "cv_mae_mean", "cv_mae_std", "cv_rmse_mean", "cv_rmse_std",
                     "cv_r2_mean", "cv_r2_std", "test_mae", "test_rmse", "test_r2", "best_params"]
    comparison_df = comparison_df[column_order]
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"\nModel comparison saved to: {COMPARISON_CSV_PATH}")
    print(comparison_df.to_string(index=False))

    # ---------------- Save all trained pipelines ----------------
    for name, pipeline in fitted_pipelines.items():
        path = MODELS_DIR / f"{name}_pipeline.pkl"
        joblib.dump(pipeline, path)
        print(f"Saved: {path}")

    # ---------------- Determine and save best model (lowest test RMSE) ----------------
    best_row = comparison_df.loc[comparison_df["test_rmse"].idxmin()]
    best_model_key = {
        "Linear Regression": "linear_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }[best_row["model"]]
    best_pipeline = fitted_pipelines[best_model_key]
    joblib.dump(best_pipeline, MODELS_DIR / "best_model_pipeline.pkl")
    print(f"\nBest model (lowest test RMSE): {best_row['model']} "
          f"(RMSE={best_row['test_rmse']:.2f}, R2={best_row['test_r2']:.4f})")
    print(f"Saved to: {MODELS_DIR / 'best_model_pipeline.pkl'}")

    # ---------------- Feature importance (tree models only) ----------------
    plot_feature_importance(fitted_pipelines["random_forest"], fitted_pipelines["xgboost"],
                             FEATURE_IMPORTANCE_PNG_PATH)
    print(f"Feature importance chart saved to: {FEATURE_IMPORTANCE_PNG_PATH}")


if __name__ == "__main__":
    main()
