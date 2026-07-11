import pandas as pd
import pytest


@pytest.fixture
def raw_heart_disease_sample():
    """
    A small, real-schema sample of the UCI Heart Disease dataset
    (rows taken from the actual dataset, as printed by df.head() in the
    notebook). Two rows have NaN in "ca"/"thal" to exercise the
    missing-value handling, the same way the real dataset does.

    Kept intentionally small and network-free so unit tests are fast
    and don't depend on ucimlrepo/internet access in CI.
    """
    data = [
        # age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,  thal, num
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0.0, 6.0, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3.0, 3.0, 2],
        [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, None, 7.0, 1],  # NaN ca
        [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0.0, None, 0],  # NaN thal
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0.0, 3.0, 0],
        [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0.0, 3.0, 0],
        [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, 2.0, 3.0, 3],
        [57, 0, 4, 120, 354, 0, 0, 163, 1, 0.6, 1, 0.0, 3.0, 0],
        [63, 1, 4, 130, 254, 0, 2, 147, 0, 1.4, 2, 1.0, 7.0, 2],
        [53, 1, 4, 140, 203, 1, 2, 155, 1, 3.1, 3, 0.0, 7.0, 1],
    ]
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    return pd.DataFrame(data, columns=columns)
