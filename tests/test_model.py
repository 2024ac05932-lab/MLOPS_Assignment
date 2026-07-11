import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import clean_heart_disease_df, split_features_target


def _build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=50, random_state=42)),
    ])


def test_pipeline_fits_and_predicts_on_real_schema(raw_heart_disease_sample):
    df = clean_heart_disease_df(raw_heart_disease_sample)
    X, y = split_features_target(df)

    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_pipeline_predict_proba_confidence_in_valid_range(raw_heart_disease_sample):
    df = clean_heart_disease_df(raw_heart_disease_sample)
    X, y = split_features_target(df)

    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    confidences = pipeline.predict_proba(X).max(axis=1)
    assert (confidences >= 0.5).all()
    assert (confidences <= 1.0).all()


def test_pipeline_raises_clearly_on_wrong_number_of_features(raw_heart_disease_sample):
    """If the serving API ever gets the wrong number of features, the pipeline
    should fail with a clear, catchable error -- not silently return garbage."""
    df = clean_heart_disease_df(raw_heart_disease_sample)
    X, y = split_features_target(df)

    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    malformed_input = X.iloc[:, :5]
    with pytest.raises(ValueError):
        pipeline.predict(malformed_input)


def test_model_hyperparameters_are_applied():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    assert model.n_estimators == 100
