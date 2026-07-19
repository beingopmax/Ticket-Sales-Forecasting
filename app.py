"""
app.py

Streamlit dashboard for the Ticket Sales Forecasting project.

IMPORTANT: This app does NOT train or fit anything. It only:
    - Loads the already-trained best model pipeline
      (models/best_model_pipeline.pkl), which bundles the fitted
      preprocessing (ColumnTransformer: StandardScaler + OneHotEncoder
      + the custom FrequencyEncoder) together with the fitted XGBoost
      model, as a single sklearn Pipeline object.
    - Loads pre-generated reports (CSV/JSON/PNG) produced by
      src/train.py and src/evaluate.py.
    - Runs pipeline.predict() on user input for live inference.

Because preprocessing is baked into the saved pipeline object, calling
pipeline.predict(raw_dataframe) automatically applies the exact same
scaling / one-hot encoding / frequency encoding used during training --
there is no separate preprocessing step to reimplement here, and no
risk of train/inference skew.

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Reuse the single source of truth for feature/target column names and
# ordering, rather than re-typing (and risking a typo'd mismatch with)
# the column list the model was actually trained on.
from src.preprocessing import ALL_MODEL_FEATURES, TARGET_COLUMN

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, where `streamlit run app.py`
# is expected to be invoked from)
# ---------------------------------------------------------------------------
BEST_MODEL_PATH = Path("models/best_model_pipeline.pkl")
COMPARISON_CSV_PATH = Path("reports/model_comparison.csv")
CLEANING_SUMMARY_JSON_PATH = Path("reports/cleaning_summary.json")
FEATURED_DATA_PATH = Path("data/processed/cinema_ticket_features.csv")

FEATURE_IMPORTANCE_IMG = Path("reports/feature_importance.png")
MODEL_COMPARISON_IMG = Path("reports/model_comparison_chart.png")
PRED_VS_ACTUAL_IMG = Path("reports/best_model_predicted_vs_actual.png")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Cached loaders -- each of these reads a saved artifact from disk once
# per session and reuses it, rather than re-reading on every interaction.
# None of these fit/train anything; they only load what train.py and
# evaluate.py already produced.
# ---------------------------------------------------------------------------

@st.cache_resource
def load_best_model():
    """Load the saved, already-fitted best model pipeline (inference only)."""
    if not BEST_MODEL_PATH.exists():
        return None
    return joblib.load(BEST_MODEL_PATH)


@st.cache_data
def load_comparison_report() -> pd.DataFrame:
    """Load the model comparison metrics produced by evaluate.py."""
    if not COMPARISON_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_CSV_PATH)


@st.cache_data
def load_cleaning_summary() -> dict:
    """Load the structured (JSON) data cleaning report from data_cleaning.py."""
    if not CLEANING_SUMMARY_JSON_PATH.exists():
        return {}
    with open(CLEANING_SUMMARY_JSON_PATH) as f:
        return json.load(f)


@st.cache_data
def load_dropdown_options() -> tuple:
    """
    Load only the columns needed to populate the film_code / cinema_code
    dropdowns with real, known values (so predictions are always made on
    categories the model has actually seen).
    """
    if not FEATURED_DATA_PATH.exists():
        return [], []
    df = pd.read_csv(FEATURED_DATA_PATH, usecols=["film_code", "cinema_code"])
    film_codes = sorted(df["film_code"].unique().tolist())
    cinema_codes = sorted(df["cinema_code"].unique().tolist())
    return film_codes, cinema_codes


# ---------------------------------------------------------------------------
# Tab 1: Project Overview
# ---------------------------------------------------------------------------

def render_overview_tab():
    st.header("Ticket Sales Forecasting & Demand Analysis")
    st.markdown(
        "An end-to-end regression pipeline that predicts **tickets sold** for a "
        "cinema screening, using only information available *before* the show "
        "happens (schedule, pricing, venue, and calendar features) -- with "
        "leakage features like `total_sales` and `occu_perc` explicitly excluded "
        "from the model."
    )

    cleaning = load_cleaning_summary()
    comparison_df = load_comparison_report()

    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of records (raw)", f"{cleaning.get('rows_before_cleaning', 'N/A'):,}"
                   if cleaning else "N/A")
    with col2:
        st.metric("Number of features (raw)", len(cleaning.get("columns", [])) or "N/A")

    st.subheader("Data Cleaning Summary")
    if cleaning:
        c1, c2, c3 = st.columns(3)
        c1.metric("Duplicates removed", cleaning.get("duplicates_removed", "N/A"))
        c2.metric("Invalid rows removed", cleaning.get("invalid_rows_removed_total", "N/A"))
        missing_handled = (
            cleaning.get("missing_capacity_imputed_per_cinema_median", 0)
            + cleaning.get("missing_capacity_dropped_no_reference", 0)
        )
        c3.metric("Missing values handled", missing_handled)
        st.caption(
            f"Rows before cleaning: {cleaning.get('rows_before_cleaning'):,} -> "
            f"Rows after cleaning: {cleaning.get('rows_after_cleaning'):,} "
            f"({cleaning.get('total_rows_removed')} removed, "
            f"{cleaning.get('total_rows_removed') / cleaning.get('rows_before_cleaning') * 100:.2f}%)."
        )
    else:
        st.warning("Cleaning summary not found. Run src/data_cleaning.py first.")

    st.subheader("Final Feature List (used by the model)")
    st.write(", ".join(f"`{f}`" for f in ALL_MODEL_FEATURES))
    st.caption(
        "Excludes leakage features (`total_sales`, `ticket_use`, `occu_perc`) which "
        "are mathematically derived from the target and were confirmed as such "
        "during the leakage analysis step."
    )

    st.subheader("Best Performing Model")
    if not comparison_df.empty:
        best_row = comparison_df.sort_values("test_rmse").iloc[0]
        st.success(f"**{best_row['model']}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{best_row['test_mae']:.2f}")
        m2.metric("RMSE", f"{best_row['test_rmse']:.2f}")
        m3.metric("R\u00b2 Score", f"{best_row['test_r2']:.3f}")
    else:
        st.warning("Model comparison report not found. Run src/train.py and src/evaluate.py first.")


# ---------------------------------------------------------------------------
# Tab 2: Data Analysis
# ---------------------------------------------------------------------------

def render_analysis_tab():
    st.header("Data Analysis & Model Insights")

    st.subheader("Feature Importance")
    if FEATURE_IMPORTANCE_IMG.exists():
        st.image(str(FEATURE_IMPORTANCE_IMG), width='stretch')
        st.caption(
            "Shows which inputs drive each tree-based model's predictions most. "
            "`show_time` and `capacity` dominate in both Random Forest and XGBoost, "
            "and the frequency-encoded `cinema_code` also carries real signal -- "
            "confirming that encoding choice added value rather than just noise."
        )
    else:
        st.info("Feature importance chart not found. Run src/train.py first.")

    st.subheader("Model Comparison")
    if MODEL_COMPARISON_IMG.exists():
        st.image(str(MODEL_COMPARISON_IMG), width='stretch')
        st.caption(
            "Side-by-side MAE, RMSE, and R\u00b2 across all three models on the same "
            "held-out test set. XGBoost has the lowest error and highest R\u00b2, which "
            "is why it was selected as the final deployed model."
        )
    else:
        st.info("Model comparison chart not found. Run src/evaluate.py first.")

    st.subheader("Predicted vs Actual (Best Model)")
    if PRED_VS_ACTUAL_IMG.exists():
        st.image(str(PRED_VS_ACTUAL_IMG), width='stretch')
        st.caption(
            "Each point is one test-set screening: x-axis is the true tickets sold, "
            "y-axis is the model's prediction. Points on the red dashed line are "
            "perfect predictions. Predictions stay tight for smaller/typical "
            "screenings and spread out more for the rare very-high-attendance "
            "shows, where there's simply less training data to learn from."
        )
    else:
        st.info("Predicted vs actual chart not found. Run src/evaluate.py first.")


# ---------------------------------------------------------------------------
# Tab 3: Ticket Sales Prediction
# ---------------------------------------------------------------------------

def build_input_dataframe(film_code, cinema_code, ticket_price, capacity,
                           tickets_out, show_time, month, quarter, day,
                           day_of_week) -> pd.DataFrame:
    """
    Assemble a single-row DataFrame with EXACTLY the columns (and only
    those columns) the saved pipeline expects, in a form matching what
    ALL_MODEL_FEATURES defines. is_weekend is derived here, not asked
    of the user.
    """
    is_weekend = 1 if day_of_week in (5, 6) else 0  # 5=Saturday, 6=Sunday

    row = {
        "ticket_price": ticket_price,
        "capacity": capacity,
        "tickets_out": tickets_out,
        "show_time": show_time,
        "month": month,
        "quarter": quarter,
        "day": day,
        "is_weekend": is_weekend,
        "film_code": film_code,
        "day_of_week": day_of_week,
        "cinema_code": cinema_code,
    }
    # Reindex to ALL_MODEL_FEATURES to guarantee exact column order/set
    # matches what the pipeline was trained on.
    return pd.DataFrame([row])[ALL_MODEL_FEATURES]


def render_prediction_tab():
    st.header("Ticket Sales Prediction")
    st.markdown("Fill in the screening details below to get a predicted ticket count.")

    model = load_best_model()
    film_codes, cinema_codes = load_dropdown_options()

    if model is None:
        st.error("No trained model found at models/best_model_pipeline.pkl. Run src/train.py first.")
        return

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            film_code = st.selectbox(
                "Film Code",
                options=film_codes if film_codes else [0],
                help="Only film codes seen during training are offered here.",
            )
            cinema_code = st.selectbox(
                "Cinema Code",
                options=cinema_codes if cinema_codes else [0],
                help="Only cinema codes seen during training are offered here.",
            )
            ticket_price = st.number_input(
                "Ticket Price", min_value=1.0, max_value=1_000_000.0,
                value=80000.0, step=1000.0,
            )
            capacity = st.number_input(
                "Capacity", min_value=1.0, max_value=5000.0, value=300.0, step=1.0,
            )
            tickets_out = st.number_input(
                "Tickets Out", min_value=0, max_value=5000, value=0, step=1,
                help="Complimentary / non-sale tickets issued for the screening.",
            )

        with col2:
            show_time = st.number_input(
                "Show Time", min_value=1, max_value=60, value=4, step=1,
                help="Show slot index for the day, as used in the source data.",
            )
            month = st.selectbox("Month", options=list(range(1, 13)), index=4)
            quarter = st.selectbox("Quarter", options=[1, 2, 3, 4], index=1)
            day = st.number_input("Day (day of month)", min_value=1, max_value=31, value=15, step=1)
            day_of_week_name = st.selectbox("Day of Week", options=DAY_NAMES, index=1)
            day_of_week = DAY_NAMES.index(day_of_week_name)  # Monday=0 ... Sunday=6

            # is_weekend is derived automatically, not user-entered --
            # shown here as read-only feedback so the user can see it.
            is_weekend_display = "Yes" if day_of_week in (5, 6) else "No"
            st.text_input("is_weekend (auto-calculated)", value=is_weekend_display, disabled=True)

        submitted = st.form_submit_button("Predict", width='stretch')

    if submitted:
        # Basic input validation beyond the widgets' own min/max bounds --
        # catches anything that would produce a nonsensical prediction.
        errors = []
        if ticket_price <= 0:
            errors.append("Ticket price must be greater than 0.")
        if capacity <= 0:
            errors.append("Capacity must be greater than 0.")
        if tickets_out < 0:
            errors.append("Tickets out cannot be negative.")

        if errors:
            for e in errors:
                st.error(e)
            return

        try:
            input_df = build_input_dataframe(
                film_code, cinema_code, ticket_price, capacity, tickets_out,
                show_time, month, quarter, day, day_of_week,
            )
            # Preprocessing + model both live inside this one pipeline object,
            # fitted during training -- this call does inference only.
            prediction = model.predict(input_df)[0]
            prediction = max(0, round(float(prediction)))

            st.markdown("### Result")
            st.metric(label="Predicted Tickets Sold", value=f"{prediction:,}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ---------------------------------------------------------------------------
# Main app layout
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Ticket Sales Forecasting",
        layout="wide",
    )

    st.title("\U0001F3AC Ticket Sales Forecasting & Demand Analysis")

    tab1, tab2, tab3 = st.tabs(["Project Overview", "Data Analysis", "Ticket Sales Prediction"])

    with tab1:
        render_overview_tab()
    with tab2:
        render_analysis_tab()
    with tab3:
        render_prediction_tab()


if __name__ == "__main__":
    main()
