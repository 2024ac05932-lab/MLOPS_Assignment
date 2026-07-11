"""
Task 1: Data Acquisition & Exploratory Data Analysis (EDA) - Heart Disease UCI Dataset

Run with:
    python eda.py

Produces plots in eda_plots/:
    class_balance.png, histograms.png, correlation_heatmap.png, boxplots_by_target.png
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo


def run_eda():
    os.makedirs("eda_plots", exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100

    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)

    print(df.head())
    print(df.shape)

    # Binary target used only for EDA labeling (original 'num' is 0-4 severity)
    df["target_binary"] = (df["num"] > 0).astype(int)

    # 1. Class balance
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(
        x="target_binary", hue="target_binary", data=df,
        palette=["#4C72B0", "#DD8452"], legend=False,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Disease (0)", "Disease (1)"])
    plt.title("Class Balance: Heart Disease Presence", fontsize=13, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=10,
        )
    plt.tight_layout()
    plt.savefig("eda_plots/class_balance.png")
    plt.close()

    # 2. Histograms of key numeric features
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#4C72B0")
        axes[i].set_title(f"Distribution of {col}", fontsize=11)
    fig.delaxes(axes[-1])
    plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_plots/histograms.png")
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(12, 9))
    corr = df.drop(columns=["target_binary"]).corr(numeric_only=True)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_plots/correlation_heatmap.png")
    plt.close()

    # 4. Key features by diagnosis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(["age", "chol", "thalach"]):
        sns.boxplot(
            x="target_binary", y=col, hue="target_binary", data=df, ax=axes[i],
            palette=["#4C72B0", "#DD8452"], legend=False,
        )
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["No Disease", "Disease"])
        axes[i].set_title(f"{col} by Diagnosis", fontsize=11)
    plt.tight_layout()
    plt.savefig("eda_plots/boxplots_by_target.png")
    plt.close()

    print(
        "EDA plots saved to eda_plots/: class_balance.png, histograms.png, "
        "correlation_heatmap.png, boxplots_by_target.png"
    )


if __name__ == "__main__":
    run_eda()
