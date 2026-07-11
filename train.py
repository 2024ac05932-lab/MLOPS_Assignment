"""
Training entry point for the Heart Disease pipeline.

Fetches the UCI Heart Disease dataset (id=45), cleans it using the shared
src/preprocessing module, trains a StandardScaler + RandomForestClassifier
pipeline, evaluates it, logs to a file, and saves the fitted pipeline to
artifacts/heart_disease_model.joblib.

Run with:
    python train.py
"""
import logging
import os

import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.preprocessing import clean_heart_disease_df, split_features_target

os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def train_model():
    dataset = fetch_ucirepo(id=45)
    df = dataset.data.features.join(dataset.data.targets)
    df = clean_heart_disease_df(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logging.info(f"Model Accuracy: {acc}")

    joblib.dump(pipeline, "artifacts/heart_disease_model.joblib")

    print("Training Complete")
    print("Accuracy:", acc)
    return pipeline, acc


if __name__ == "__main__":
    train_model()
