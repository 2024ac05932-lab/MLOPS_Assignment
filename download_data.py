"""
Downloads the UCI Heart Disease dataset (id=45) and saves it locally as
a CSV so the pipeline doesn't need network access to UCI every run.

Run with:
    python download_data.py

Produces:
    data/heart_disease_raw.csv
"""
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download():
    os.makedirs("data", exist_ok=True)
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)
    out_path = os.path.join("data", "heart_disease_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    download()
