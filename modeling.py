"""
Task 2: Feature Engineering & Model Development

Trains and evaluates Logistic Regression and Random Forest classifiers on
the UCI Heart Disease dataset, with 5-fold CV and GridSearchCV tuning.

Run with:
    python modeling.py
"""
from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
)

from src.preprocessing import clean_heart_disease_df, split_features_target


def load_data():
    heart_disease = fetch_ucirepo(id=45)
    df = pd.concat([heart_disease.data.features, heart_disease.data.targets], axis=1)
    df = clean_heart_disease_df(df)
    return split_features_target(df)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Model 1: Logistic Regression ---
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    print("LOGISTIC REGRESSION")
    print("Accuracy :", accuracy_score(y_test, y_pred_lr))
    print("Precision:", precision_score(y_test, y_pred_lr))
    print("Recall   :", recall_score(y_test, y_pred_lr))
    print("F1 Score :", f1_score(y_test, y_pred_lr))
    print("ROC-AUC  :", roc_auc_score(y_test, y_prob_lr))
    print(classification_report(y_test, y_pred_lr))

    # --- Model 2: Random Forest ---
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    print("RANDOM FOREST")
    print("Accuracy :", accuracy_score(y_test, y_pred_rf))
    print("Precision:", precision_score(y_test, y_pred_rf))
    print("Recall   :", recall_score(y_test, y_pred_rf))
    print("F1 Score :", f1_score(y_test, y_pred_rf))
    print("ROC-AUC  :", roc_auc_score(y_test, y_prob_rf))
    print(classification_report(y_test, y_pred_rf))

    # --- 5-fold CV ---
    cv_lr = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    cv_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy")
    print("Logistic Regression CV Accuracy:", cv_lr.mean())
    print("Random Forest CV Accuracy:", cv_rf.mean())

    # --- Hyperparameter tuning ---
    grid_lr = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [0.01, 0.1, 1, 10, 100]},
        cv=5, scoring="roc_auc",
    )
    grid_lr.fit(X_train_scaled, y_train)
    print("LR Best Parameters:", grid_lr.best_params_)
    print("LR Best ROC-AUC:", grid_lr.best_score_)

    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 200, 300], "max_depth": [5, 10, 15, None], "min_samples_split": [2, 5]},
        cv=5, scoring="roc_auc",
    )
    grid_rf.fit(X_train, y_train)
    print("RF Best Parameters:", grid_rf.best_params_)
    print("RF Best ROC-AUC:", grid_rf.best_score_)

    # --- ROC curve comparison ---
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression AUC={roc_auc_score(y_test, y_prob_lr):.3f}")
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest AUC={roc_auc_score(y_test, y_prob_rf):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eda_plots/roc_curve_comparison.png")
    plt.close()

    return {
        "lr_model": lr_model, "rf_model": rf_model,
        "X_test": X_test, "X_test_scaled": X_test_scaled, "y_test": y_test,
        "y_pred_lr": y_pred_lr, "y_prob_lr": y_prob_lr,
        "y_pred_rf": y_pred_rf, "y_prob_rf": y_prob_rf,
    }


if __name__ == "__main__":
    main()
