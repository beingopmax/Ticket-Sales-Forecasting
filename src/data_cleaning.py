"""
data_cleaning.py

Responsibility:
    Load the raw cinema ticket dataset and produce a cleaned, validated
    dataset ready for feature engineering.

Cleaning decisions (approved by project owner):
    1. Drop fully duplicate rows.
    2. Drop rows with physically impossible values:
         - capacity <= 0
         - ticket_use < 0
    3. Do NOT drop rows where occu_perc > 100%. These are kept for
       EDA / business analysis. occu_perc is excluded from modeling
       entirely (see leakage analysis) so its internal inconsistency
       does not affect the model.
    4. Missing `capacity` (125 rows) is handled as follows:
         - IMPORTANT DATA QUALITY NOTE: `capacity` is not a stable,
           fixed property of a cinema screen in this dataset -- for
           242/244 cinemas it takes on many different values for the
           same cinema_code (up to 130% relative std). It appears to
           be back-calculated as tickets_sold / (occu_perc / 100),
           which amplifies rounding noise when occu_perc is small.
           This is documented here rather than silently ignored.
         - For rows whose cinema_code has OTHER valid capacity
           readings elsewhere in the dataset: impute with that
           cinema's MEDIAN capacity (robust to the rounding-driven
           outliers described above).
         - For rows whose cinema_code has NO capacity reading
           anywhere in the dataset (cinema_code 637, 543 -> 32 rows):
           no reasonable reconstruction is possible, so these rows
           are dropped. This is logged explicitly in the cleaning
           report, not silently discarded.
    5. A full before/after cleaning summary is generated and saved
       to reports/cleaning_summary.txt (and returned as a dict for
       programmatic use in later notebooks/scripts).

Leakage note:
    total_sales, ticket_use, and occu_perc are NOT dropped in this
    module -- they are kept in the cleaned dataset for EDA purposes.
    They are excluded later, explicitly, in preprocessing.py / at the
    start of feature_engineering.py, right before modeling. Keeping
    them here means notebooks/01_EDA.ipynb can still use them for
    business analysis (e.g. occupancy trends per cinema).
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/cinema_ticket.csv")
PROCESSED_DATA_PATH = Path("data/processed/cinema_ticket_cleaned.csv")
REPORT_PATH = Path("reports/cleaning_summary.txt")
REPORT_JSON_PATH = Path("reports/cleaning_summary.json")


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and parse the date column."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def drop_duplicates(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Drop fully duplicate rows. Records count in report."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    report["duplicates_removed"] = before - after
    return df


def drop_invalid_rows(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Drop physically impossible rows:
        - capacity <= 0
        - ticket_use < 0
    occu_perc > 100 rows are intentionally KEPT (see module docstring).
    """
    before = len(df)

    invalid_capacity_mask = df["capacity"] <= 0
    invalid_ticket_use_mask = df["ticket_use"] < 0

    n_invalid_capacity = int(invalid_capacity_mask.sum())
    n_invalid_ticket_use = int(invalid_ticket_use_mask.sum())
    # Rows can overlap between the two conditions, so track union count too.
    combined_mask = invalid_capacity_mask | invalid_ticket_use_mask
    n_combined = int(combined_mask.sum())

    df = df[~combined_mask].copy()

    report["invalid_capacity_le_0_removed"] = n_invalid_capacity
    report["invalid_ticket_use_lt_0_removed"] = n_invalid_ticket_use
    report["invalid_rows_removed_total"] = n_combined
    report["rows_after_invalid_removal"] = len(df)
    report["occu_perc_over_100_kept"] = int((df["occu_perc"] > 100).sum())

    assert len(df) == before - n_combined
    return df


def handle_missing_capacity(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """
    Impute or drop rows with missing `capacity`.

    Strategy:
        - Per cinema_code median imputation where other readings exist
          for that cinema (robust to rounding-driven variance).
        - Drop rows where the cinema_code has NO capacity reading
          anywhere in the dataset (no reasonable reconstruction
          possible).
    """
    n_missing_before = int(df["capacity"].isnull().sum())

    # Per-cinema median capacity, computed only from non-null rows.
    cinema_median_capacity = (
        df.dropna(subset=["capacity"])
        .groupby("cinema_code")["capacity"]
        .median()
    )

    missing_mask = df["capacity"].isnull()
    missing_cinema_codes = df.loc[missing_mask, "cinema_code"]

    # Rows we CAN impute: cinema_code has a median available.
    imputable_mask = missing_mask & df["cinema_code"].isin(cinema_median_capacity.index)
    # Rows we CANNOT impute: cinema_code has zero capacity data anywhere.
    unrecoverable_mask = missing_mask & ~df["cinema_code"].isin(cinema_median_capacity.index)

    n_imputed = int(imputable_mask.sum())
    n_unrecoverable = int(unrecoverable_mask.sum())
    unrecoverable_cinema_codes = sorted(df.loc[unrecoverable_mask, "cinema_code"].unique().tolist())

    # Apply imputation.
    df.loc[imputable_mask, "capacity"] = df.loc[imputable_mask, "cinema_code"].map(
        cinema_median_capacity
    )

    # Drop unrecoverable rows.
    df = df[~unrecoverable_mask].copy()

    report["missing_capacity_before"] = n_missing_before
    report["missing_capacity_imputed_per_cinema_median"] = n_imputed
    report["missing_capacity_dropped_no_reference"] = n_unrecoverable
    report["missing_capacity_dropped_cinema_codes"] = unrecoverable_cinema_codes
    report["missing_capacity_after"] = int(df["capacity"].isnull().sum())

    return df


def add_derived_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanity-check: confirm month/quarter/day columns already in the
    raw data agree with the parsed `date` column. Does NOT create
    day_of_week / is_weekend here -- that belongs to
    feature_engineering.py per the project architecture.
    """
    mismatches = (
        (df["month"] != df["date"].dt.month)
        | (df["day"] != df["date"].dt.day)
    ).sum()
    if mismatches > 0:
        print(f"WARNING: {mismatches} rows have month/day inconsistent with date column.")
    return df


def clean_data(raw_path: Path = RAW_DATA_PATH,
               processed_path: Path = PROCESSED_DATA_PATH,
               report_path: Path = REPORT_PATH) -> tuple[pd.DataFrame, dict]:
    """
    Run the full cleaning pipeline and return (cleaned_df, report).
    Also writes the cleaned CSV and a human-readable report to disk.
    """
    report = {}

    df = load_data(raw_path)
    report["rows_before_cleaning"] = len(df)
    report["columns"] = list(df.columns)

    df = drop_duplicates(df, report)
    df = drop_invalid_rows(df, report)
    df = handle_missing_capacity(df, report)
    df = add_derived_date_parts(df)

    report["rows_after_cleaning"] = len(df)
    report["total_rows_removed"] = report["rows_before_cleaning"] - report["rows_after_cleaning"]

    # Final validation: no remaining nulls in columns we care about.
    remaining_nulls = df.isnull().sum()
    report["remaining_nulls"] = {
        col: int(n) for col, n in remaining_nulls.items() if n > 0
    }

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report, report_path)
    _write_report_json(report, REPORT_JSON_PATH)

    return df, report


def _write_report_json(report: dict, path: Path) -> None:
    """
    Structured, machine-readable version of the same cleaning report --
    used by app.py (Tab 1) so the dashboard never has to parse the
    human-readable .txt report with string logic.
    """
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def _write_report(report: dict, path: Path) -> None:
    lines = []
    lines.append("=" * 60)
    lines.append("DATA CLEANING SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Rows before cleaning:              {report['rows_before_cleaning']}")
    lines.append(f"Duplicate rows removed:             {report['duplicates_removed']}")
    lines.append("")
    lines.append("Invalid (physically impossible) rows removed:")
    lines.append(f"  capacity <= 0:                     {report['invalid_capacity_le_0_removed']}")
    lines.append(f"  ticket_use < 0:                    {report['invalid_ticket_use_lt_0_removed']}")
    lines.append(f"  total removed (union, dedup'd):    {report['invalid_rows_removed_total']}")
    lines.append(f"  occu_perc > 100% rows KEPT:        {report['occu_perc_over_100_kept']} (excluded from modeling only, not dropped)")
    lines.append("")
    lines.append("Missing capacity handling:")
    lines.append(f"  Missing before:                    {report['missing_capacity_before']}")
    lines.append(f"  Imputed via per-cinema median:     {report['missing_capacity_imputed_per_cinema_median']}")
    lines.append(f"  Dropped (no reference available):  {report['missing_capacity_dropped_no_reference']}")
    lines.append(f"    cinema_codes with no reference:  {report['missing_capacity_dropped_cinema_codes']}")
    lines.append(f"  Missing after:                     {report['missing_capacity_after']}")
    lines.append("")
    lines.append(f"Rows after cleaning:                {report['rows_after_cleaning']}")
    lines.append(f"Total rows removed:                 {report['total_rows_removed']} "
                 f"({report['total_rows_removed']/report['rows_before_cleaning']*100:.2f}%)")
    lines.append("")
    if report["remaining_nulls"]:
        lines.append(f"WARNING - remaining nulls: {report['remaining_nulls']}")
    else:
        lines.append("Remaining nulls in cleaned dataset: none")
    lines.append("=" * 60)

    with open(path, "w") as f:
        f.write("\n".join(lines))


# Columns that will actually be fed to the model (per approved feature
# list). Excludes leakage columns (total_sales, ticket_use, occu_perc)
# and day_of_week/is_weekend, which don't exist yet at this stage --
# those are created in feature_engineering.py and validated there.
MODEL_FEATURE_COLUMNS = [
    "film_code", "cinema_code", "ticket_price", "capacity",
    "tickets_out", "show_time", "month", "quarter", "day",
]
TARGET_COLUMN = "tickets_sold"


def validate_cleaned_data(df: pd.DataFrame) -> dict:
    """
    Final validation gate for the cleaned dataset. Raises AssertionError
    on any failure so a broken dataset can never silently flow into
    feature_engineering.py / modeling.

    Checks:
        1. No invalid capacity values (capacity must be > 0).
        2. No negative ticket-related values (tickets_sold, tickets_out,
           ticket_use must all be >= 0).
        3. No duplicate rows.
        4. Model feature columns + target contain no missing values.
           (occu_perc is intentionally EXCLUDED from this check -- it's
           excluded from modeling entirely, so its nulls are left
           unchanged by design, not a defect.)

    Returns a dict summary (also useful to print/log) and raises if
    any check fails.
    """
    results = {}

    # 1. Invalid capacity
    invalid_capacity = int((df["capacity"] <= 0).sum())
    results["invalid_capacity_count"] = invalid_capacity
    assert invalid_capacity == 0, (
        f"Validation failed: {invalid_capacity} rows have capacity <= 0"
    )

    # 2. Negative ticket-related values
    negative_counts = {
        "tickets_sold": int((df["tickets_sold"] < 0).sum()),
        "tickets_out": int((df["tickets_out"] < 0).sum()),
        "ticket_use": int((df["ticket_use"] < 0).sum()),
    }
    results["negative_ticket_value_counts"] = negative_counts
    for col, n in negative_counts.items():
        assert n == 0, f"Validation failed: {n} rows have negative {col}"

    # 3. Duplicates
    n_duplicates = int(df.duplicated().sum())
    results["duplicate_rows"] = n_duplicates
    assert n_duplicates == 0, f"Validation failed: {n_duplicates} duplicate rows found"

    # 4. Missing values in model features + target (occu_perc excluded)
    check_columns = MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_in_features = {
        col: int(df[col].isnull().sum())
        for col in check_columns
        if df[col].isnull().sum() > 0
    }
    results["missing_in_model_features"] = missing_in_features
    assert not missing_in_features, (
        f"Validation failed: missing values found in model feature columns: {missing_in_features}"
    )

    # Informational only, not asserted on: occu_perc nulls are expected
    # and intentionally left as-is.
    results["occu_perc_nulls_kept_by_design"] = int(df["occu_perc"].isnull().sum())

    results["status"] = "PASSED"
    return results


def _print_validation_results(results: dict) -> None:
    print("=" * 60)
    print("FINAL VALIDATION CHECK")
    print("=" * 60)
    print(f"Invalid capacity (<=0) rows:         {results['invalid_capacity_count']}")
    print(f"Negative ticket-value rows:           {results['negative_ticket_value_counts']}")
    print(f"Duplicate rows:                       {results['duplicate_rows']}")
    print(f"Missing values in model features:     {results['missing_in_model_features'] or 'none'}")
    print(f"occu_perc nulls (kept by design):     {results['occu_perc_nulls_kept_by_design']}")
    print(f"Status: {results['status']}")
    print("=" * 60)


if __name__ == "__main__":
    cleaned_df, report = clean_data()
    print(f"Cleaned dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Cleaning report saved to: {REPORT_PATH}")
    print()
    with open(REPORT_PATH) as f:
        print(f.read())
    print()

    validation_results = validate_cleaned_data(cleaned_df)
    _print_validation_results(validation_results)
