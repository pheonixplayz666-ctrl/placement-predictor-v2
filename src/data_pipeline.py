"""
data_pipeline.py

Leakage-safe data pipeline for the Placement Predictor v2 project.

Key design decisions (and why):
1. Classification (placement_status) and regression (salary_package_lpa) are
   ENTIRELY SEPARATE datasets. The regression target only exists for placed
   students, so training regression on the full dataset (with imputed/zeroed
   salary for unplaced students) silently leaks information and produces a
   model that looks good on paper but is nonsense in production.
2. Train/test splits are done ONCE per task and reused everywhere downstream.
   No re-splitting after feature engineering — that's how leakage sneaks back in.
3. All fitted transformers (encoders, scalers) are fit on TRAIN ONLY, then
   applied to test. Never fit on the full dataset.
4. Splits are saved to disk so training and evaluation scripts never
   accidentally re-shuffle data differently.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import joblib

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "student_placement_synthetic.csv"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2

CATEGORICAL_COLS = ["branch", "college_tier"]
CLASSIFICATION_TARGET = "placement_status"
REGRESSION_TARGET = "salary_package_lpa"


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    return df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list):
    """
    Fit an OrdinalEncoder on TRAIN ONLY, apply to both. Tree-based models
    (LightGBM/CatBoost) handle ordinal-encoded categoricals fine, and this
    avoids the column-explosion of one-hot encoding on branch (7 categories)
    + college_tier (3 categories) while keeping things simple and reproducible.
    """
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[cols] = encoder.fit_transform(train_df[cols])
    test_df[cols] = encoder.transform(test_df[cols])
    return train_df, test_df, encoder


def build_classification_splits(df: pd.DataFrame):
    """
    Full dataset is valid for classification -- every row has a
    placement_status label. Stratify on the target to preserve the
    68.5% / 31.5% class balance in both splits.
    """
    feature_cols = [c for c in df.columns if c not in [CLASSIFICATION_TARGET, REGRESSION_TARGET]]
    X = df[feature_cols].copy()
    y = df[CLASSIFICATION_TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_test, encoder = encode_categoricals(X_train, X_test, CATEGORICAL_COLS)

    return X_train, X_test, y_train, y_test, encoder


def build_regression_splits(df: pd.DataFrame):
    """
    CRITICAL: only rows where placement_status == 1 are valid for regression.
    Unplaced students have NaN salary by definition, not "missing data" to
    impute -- imputing them would fabricate a target that doesn't exist.
    """
    placed_df = df[df[CLASSIFICATION_TARGET] == 1].copy()
    assert placed_df[REGRESSION_TARGET].isnull().sum() == 0, \
        "Unexpected nulls in salary for placed students -- investigate before proceeding."

    # placement_status is now constant (all 1) for this subset, so it carries
    # zero information -- drop it to avoid a degenerate feature.
    feature_cols = [c for c in placed_df.columns if c not in [CLASSIFICATION_TARGET, REGRESSION_TARGET]]
    X = placed_df[feature_cols].copy()
    y = placed_df[REGRESSION_TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train, X_test, encoder = encode_categoricals(X_train, X_test, CATEGORICAL_COLS)

    return X_train, X_test, y_train, y_test, encoder


def run_pipeline():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Placement rate: {df[CLASSIFICATION_TARGET].mean():.1%}")

    # --- Classification splits ---
    Xc_train, Xc_test, yc_train, yc_test, clf_encoder = build_classification_splits(df)
    joblib.dump((Xc_train, Xc_test, yc_train, yc_test), PROCESSED_DIR / "classification_splits.pkl")
    joblib.dump(clf_encoder, MODELS_DIR / "classification_encoder.pkl")
    print(f"Classification: train={len(Xc_train):,}, test={len(Xc_test):,}")

    # --- Regression splits (placed subset only) ---
    Xr_train, Xr_test, yr_train, yr_test, reg_encoder = build_regression_splits(df)
    joblib.dump((Xr_train, Xr_test, yr_train, yr_test), PROCESSED_DIR / "regression_splits.pkl")
    joblib.dump(reg_encoder, MODELS_DIR / "regression_encoder.pkl")
    print(f"Regression (placed only): train={len(Xr_train):,}, test={len(Xr_test):,}")

    print("\nPipeline complete. Processed splits saved to data/processed/")


if __name__ == "__main__":
    run_pipeline()
