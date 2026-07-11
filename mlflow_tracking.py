"""
Task 3: Experiment Tracking Using MLflow

Trains Logistic Regression and Random Forest, logging params, metrics,
and artifacts (confusion matrix + ROC curve plots) for each run.

Run with:
    mlflow ui   (in one terminal, to view results after)
    python mlflow_tracking.py

View the MLflow UI at http://localhost:5000 after running `mlflow ui`
from the same directory (it reads the local ./mlruns folder).
"""
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import pandas as pd

from src.preprocessing import clean_heart_disease_df, split_features_target


def load_data():
    heart_disease = fetch_ucirepo(id=45)
    df = pd.concat([heart_disease.data.features, heart_disease.data.targets], axis=1)
    df = clean_heart_disease_df(df)
    return split_features_target(df)


def track_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    mlflow.end_run()
    with mlflow.start_run(run_name="Logistic_Regression"):
        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("max_iter", 1000)

        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        y_pred_lr = lr_model.predict(X_test_scaled)
        y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_lr))
        mlflow.log_metric("precision", precision_score(y_test, y_pred_lr))
        mlflow.log_metric("recall", recall_score(y_test, y_pred_lr))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_lr))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob_lr))

        mlflow.sklearn.log_model(lr_model, "logistic_regression_model")

        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, ax=ax)
        plt.title("Logistic Regression Confusion Matrix")
        plt.savefig("lr_confusion_matrix.png")
        mlflow.log_artifact("lr_confusion_matrix.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        RocCurveDisplay.from_predictions(y_test, y_prob_lr, ax=ax)
        plt.title("Logistic Regression ROC Curve")
        plt.savefig("lr_roc_curve.png")
        mlflow.log_artifact("lr_roc_curve.png")
        plt.close()

        print("Logistic Regression Logged Successfully")
    return lr_model


def track_random_forest(X_train, X_test, y_train, y_test):
    mlflow.end_run()
    with mlflow.start_run(run_name="Random_Forest"):
        mlflow.log_param("model", "Random Forest")
        mlflow.log_param("n_estimators", 200)

        rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_rf))
        mlflow.log_metric("precision", precision_score(y_test, y_pred_rf))
        mlflow.log_metric("recall", recall_score(y_test, y_pred_rf))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_rf))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob_rf))

        mlflow.sklearn.log_model(rf_model, "random_forest_model")

        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, ax=ax)
        plt.title("Random Forest Confusion Matrix")
        plt.savefig("rf_confusion_matrix.png")
        mlflow.log_artifact("rf_confusion_matrix.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        RocCurveDisplay.from_predictions(y_test, y_prob_rf, ax=ax)
        plt.title("Random Forest ROC Curve")
        plt.savefig("rf_roc_curve.png")
        mlflow.log_artifact("rf_roc_curve.png")
        plt.close()

        print("Random Forest Logged Successfully")
    return rf_model


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    track_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test)
    track_random_forest(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
