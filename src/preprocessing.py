"""
preprocessing.py

Responsibility:
    Build a single, reusable sklearn preprocessing Pipeline
    (ColumnTransformer) that can be plugged in front of Linear
    Regression, Random Forest, or XGBoost identically in train.py.

Encoding decisions (see full comparison in project discussion):

    cinema_code (246 raw categories, 243 after cleaning) ->
        FREQUENCY ENCODING (custom transformer below).
        Chosen over one-hot (too high-dimensional: ~243 extra sparse
        columns) and over native categorical handling (only XGBoost
        supports that -- Linear Regression and Random Forest don't,
        which would force two divergent preprocessing paths instead
        of one reusable pipeline). Frequency encoding never touches
        the target variable, so there is no target-leakage risk --
        unlike target encoding, it needs no cross-validation trick to
        be safe. It only needs to be fit on the training fold, which
        is the normal behavior of Pipeline.fit(X_train, y_train).

    film_code (48 categories) -> ONE-HOT ENCODING.
        Low cardinality, no ordinal meaning, cheap to expand.

    day_of_week (7 categories) -> ONE-HOT ENCODING.
        Low cardinality, non-ordinal (Tuesday being the peak day in
        this data is a good example of why treating it as ordinal
        1-7 would be misleading).

    Numerical features (ticket_price, capacity, tickets_out,
    show_time, month, quarter, day, is_weekend) -> StandardScaler.
        Scaling doesn't hurt Random Forest / XGBoost and is required
        for Linear Regression, so applying it uniformly keeps one
        single shared pipeline across all three models instead of
        model-specific preprocessing branches.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


FEATURED_DATA_PATH = Path("data/processed/cinema_ticket_features.csv")

NUMERICAL_FEATURES = [
    "ticket_price", "capacity", "tickets_out", "show_time",
    "month", "quarter", "day", "is_weekend",
]
ONEHOT_FEATURES = ["film_code", "day_of_week"]
FREQUENCY_FEATURES = ["cinema_code"]
TARGET_COLUMN = "tickets_sold"

ALL_MODEL_FEATURES = NUMERICAL_FEATURES + ONEHOT_FEATURES + FREQUENCY_FEATURES


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes a single categorical column as its normalized frequency
    (proportion of rows) observed at fit time.

    Leakage safety:
        - Never uses the target (y) -- fit(X, y=None) ignores y
          entirely. There is nothing for it to leak from the target,
          which is the key advantage over target encoding.
        - As long as it's fit only on the training fold (the default
          behavior when this sits inside a Pipeline / ColumnTransformer
          and you call pipeline.fit(X_train, y_train) in train.py),
          test-set and future/inference-time category frequencies
          never influence the encoding.

    Unseen categories at transform time (a cinema_code never seen in
    training -- e.g. a brand-new cinema entered in the Streamlit app)
    map to 0.0: an explicit "unknown" signal rather than a silently
    misleading average.
    """

    def __init__(self):
        self.freq_map_ = {}

    def fit(self, X, y=None):
        series = self._to_series(X)
        self.freq_map_ = series.value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        series = self._to_series(X)
        encoded = series.map(self.freq_map_).fillna(0.0)
        return encoded.to_numpy().reshape(-1, 1)

    @staticmethod
    def _to_series(X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        return pd.Series(np.ravel(X))

    def get_feature_names_out(self, input_features=None):
        col = input_features[0] if input_features is not None else "cinema_code"
        return np.array([f"{col}_freq"])


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer. This is the single object reused for
    Linear Regression, Random Forest, and XGBoost in train.py -- each
    model just gets `Pipeline([('preprocessor', build_preprocessor()),
    ('model', <model>)])`.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("film_onehot", OneHotEncoder(handle_unknown="ignore"), ["film_code"]),
            ("dow_onehot", OneHotEncoder(handle_unknown="ignore"), ["day_of_week"]),
            ("cinema_freq", FrequencyEncoder(), ["cinema_code"]),
        ],
        remainder="drop",
    )


def build_preprocessing_pipeline() -> Pipeline:
    """Wraps the ColumnTransformer in a Pipeline for consistency/reuse."""
    return Pipeline(steps=[("preprocessor", build_preprocessor())])


if __name__ == "__main__":
    df = pd.read_csv(FEATURED_DATA_PATH)
    X = df[ALL_MODEL_FEATURES]
    y = df[TARGET_COLUMN]

    # Quick smoke test: fit/transform on a train-like split, then
    # transform a held-out slice to prove unseen-category handling
    # works (simulates train.py's real train/test split in Step 5).
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_train, y_train)

    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    print("=" * 60)
    print("PREPROCESSING PIPELINE SMOKE TEST")
    print("=" * 60)
    print(f"Input features:            {len(ALL_MODEL_FEATURES)} -> {ALL_MODEL_FEATURES}")
    print(f"X_train shape (raw):       {X_train.shape}")
    print(f"X_train shape (transformed): {X_train_transformed.shape}")
    print(f"X_test shape (transformed):  {X_test_transformed.shape}")

    # Confirm unseen-category handling: any cinema_code in test but not
    # in train should be encoded as frequency 0.0, not error out.
    train_cinemas = set(X_train["cinema_code"].unique())
    test_only_cinemas = set(X_test["cinema_code"].unique()) - train_cinemas
    print(f"\ncinema_codes in test but not in train: {len(test_only_cinemas)}")
    if test_only_cinemas:
        sample_code = next(iter(test_only_cinemas))
        freq_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cinema_freq"]
        encoded_val = freq_encoder.transform(pd.DataFrame({"cinema_code": [sample_code]}))
        print(f"Example unseen cinema_code={sample_code} encoded as: {encoded_val.flatten()[0]} (expected 0.0)")

    print("\nNo errors -- pipeline is fit-safe and transform-safe on unseen categories.")
    print("=" * 60)
