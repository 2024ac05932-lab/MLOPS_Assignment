"""
Shared preprocessing logic for the Heart Disease pipeline.

Pulled out of the notebook / train.py into its own module so it can be
imported and unit-tested directly, without needing network access to
ucimlrepo and without duplicating the logic in three different places.
"""
import pandas as pd

REQUIRED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def clean_heart_disease_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning steps used in the notebook:
      - binarize the target ("num" > 0 -> 1, else 0)
      - fill missing "ca" / "thal" with their mode

    Raises a clear, actionable error if the input doesn't have the
    columns this pipeline depends on, instead of failing deep inside
    pandas with a confusing KeyError.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input dataframe is missing required column(s): {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    df = df.copy()
    df["num"] = (df["num"] > 0).astype(int)
    df["ca"] = df["ca"].fillna(df["ca"].mode()[0])
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0])
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "num"):
    """Split a cleaned dataframe into X (features) and y (target)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
