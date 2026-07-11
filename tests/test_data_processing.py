import pandas as pd
import pytest

from src.preprocessing import clean_heart_disease_df, split_features_target, REQUIRED_COLUMNS


def test_clean_fills_missing_ca_and_thal(raw_heart_disease_sample):
    """The two rows with NaN in ca/thal should be filled with the mode, not dropped."""
    assert raw_heart_disease_sample["ca"].isna().sum() == 1
    assert raw_heart_disease_sample["thal"].isna().sum() == 1

    cleaned = clean_heart_disease_df(raw_heart_disease_sample)

    assert cleaned["ca"].isna().sum() == 0
    assert cleaned["thal"].isna().sum() == 0
    assert len(cleaned) == len(raw_heart_disease_sample)


def test_clean_binarizes_target(raw_heart_disease_sample):
    """"num" (0-4 severity) must become strictly binary 0/1."""
    cleaned = clean_heart_disease_df(raw_heart_disease_sample)
    assert set(cleaned["num"].unique()).issubset({0, 1})
    assert cleaned.loc[1, "num"] == 1  # raw num was 2
    assert cleaned.loc[4, "num"] == 0  # raw num was 0


def test_clean_raises_clear_error_on_missing_columns():
    """If the pipeline gets a dataframe with the wrong schema, it should fail
    loudly with a clear message -- not deep inside pandas with a KeyError."""
    bad_df = pd.DataFrame({"age": [50, 60], "sex": [1, 0]})
    with pytest.raises(ValueError) as exc_info:
        clean_heart_disease_df(bad_df)
    assert "missing required column" in str(exc_info.value).lower()


def test_split_features_target_shapes(raw_heart_disease_sample):
    cleaned = clean_heart_disease_df(raw_heart_disease_sample)
    X, y = split_features_target(cleaned)
    assert "num" not in X.columns
    assert list(X.columns) == [c for c in REQUIRED_COLUMNS if c != "num"]
    assert len(X) == len(y) == len(cleaned)
