"""
feature_engineering.py

Responsibility:
    Take the cleaned dataset (output of data_cleaning.py) and add
    engineered features that are available BEFORE ticket sales
    happen, per the approved feature list.

Features created:
    1. day_of_week (0=Monday ... 6=Sunday)
       Rationale: demand is not evenly spread across the week.
       Checked on this dataset before deciding to include it:
           Mon=76.9, Tue=275.1, Wed=114.7, Thu=148.3,
           Fri=171.4, Sat=103.2, Sun=81.5   (mean tickets_sold)
       This is a real, non-trivial pattern (not noise), so it's kept
       as a feature. Note it is NOT the generic "weekends sell more"
       pattern you'd expect a priori -- Tuesday is actually the
       highest here, which is exactly the kind of thing a model can
       pick up on but a human wouldn't hand-code, which is the
       argument for including day_of_week explicitly rather than
       relying on month/quarter/day alone.

    2. is_weekend (1 if Saturday/Sunday, else 0)
       Rationale: a coarser, lower-variance version of the same
       signal, useful for tree models to split on directly instead of
       needing multiple splits across 7 day_of_week categories.
       Checked: weekend mean = 92.5 vs weekday mean = 159.2 on this
       dataset -- confirms it captures a real (if inverted-from-usual)
       split, kept for the same reason as day_of_week.

Features considered and DELIBERATELY NOT created (documented so this
decision isn't re-litigated later without reason):

    - price_category (binning ticket_price):
        Checked mean tickets_sold across price quintiles:
        132, 97, 146, 134, 201 -- not monotonic, not a clean step
        pattern. Pearson correlation of raw ticket_price with
        tickets_sold is only 0.10. Binning a weak, non-monotonic
        relationship doesn't add signal a tree model doesn't already
        get from the raw numeric column, so it was skipped to avoid
        unnecessary complexity.

    - show_time_category (binning show_time):
        show_time is actually the STRONGEST engineered candidate
        (Pearson corr = 0.52, Spearman = 0.46 with tickets_sold), but
        binning it isn't warranted: values above ~35 occur only 1-8
        times each in 142K rows (sparse tail), so a categorical
        bucket there would just re-create the numeric ordering with
        less precision and less reliable per-bucket statistics.
        Kept as-is, as a numeric feature.

Both decisions favor "keep it simple, let Random Forest / XGBoost
handle nonlinearity natively" over hand-engineering bins that don't
show clear evidence of adding value on this dataset.
"""

import pandas as pd
from pathlib import Path


CLEANED_DATA_PATH = Path("data/processed/cinema_ticket_cleaned.csv")
FEATURED_DATA_PATH = Path("data/processed/cinema_ticket_features.csv")


def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """0=Monday ... 6=Sunday, derived from the `date` column."""
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


def add_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """1 if Saturday/Sunday, else 0. Depends on day_of_week existing."""
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def engineer_features(cleaned_path: Path = CLEANED_DATA_PATH,
                       output_path: Path = FEATURED_DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned dataset, add engineered features, save and return
    the result.
    """
    df = pd.read_csv(cleaned_path, parse_dates=["date"])

    df = add_day_of_week(df)
    df = add_is_weekend(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def summarize_new_features(df: pd.DataFrame) -> None:
    """Print a quick sanity-check summary of the engineered features."""
    print("=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print("\nMean tickets_sold by day_of_week (0=Mon..6=Sun):")
    print(df.groupby("day_of_week")["tickets_sold"].mean().round(1))
    print("\nMean tickets_sold by is_weekend:")
    print(df.groupby("is_weekend")["tickets_sold"].mean().round(1))
    print(f"\nRows: {len(df)}")
    print(f"Columns now: {list(df.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    featured_df = engineer_features()
    print(f"Featured dataset saved to: {FEATURED_DATA_PATH}\n")
    summarize_new_features(featured_df)
