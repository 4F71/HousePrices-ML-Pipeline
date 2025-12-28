"""
Generate Kaggle submission file (RidgeCV model, log-scaled target support)
Author: Onur Tilki
"""

import pandas as pd
import numpy as np
from joblib import load
from src.features import feature_engineer

# --- Paths ---
MODEL_PATH = "models/houseprice.joblib"
TEST_PATH = "data/test.csv"
SUB_PATH = "submission.csv"

def main():
    print(" Generating submission file...")

    model = load(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")

    test = pd.read_csv(TEST_PATH)
    ids = test["Id"].copy()
    print(f"Test data loaded: {test.shape}")

    test_fe = feature_engineer(test.copy())
    print("Feature engineering applied.")

    if len(test_fe) != len(ids):
        diff = len(ids) - len(test_fe)
        print(f" {diff} rows dropped during feature engineering → restoring alignment...")
        test_fe = test_fe.reindex(range(len(ids)), fill_value=0)
        test_fe["Id"] = ids

    expected_cols = getattr(model, "feature_names_in_", test_fe.columns)
    for col in expected_cols:
        if col not in test_fe.columns:
            test_fe[col] = 0
    test_fe = test_fe[expected_cols]

    preds = model.predict(test_fe)
    print(f" Raw predictions complete ({len(preds)} values).")

    if np.mean(preds) < 20:  
        print(" Detected log-transformed target, applying expm1()...")
        preds = np.expm1(preds)


    if len(preds) != len(ids):
        print(f" Fixing prediction length mismatch ({len(preds)} vs {len(ids)})...")
        min_len = min(len(preds), len(ids))
        preds = preds[:min_len]
        ids = ids[:min_len]

    submission = pd.DataFrame({
        "Id": ids,
        "SalePrice": preds
    })

    submission.to_csv(SUB_PATH, index=False)
    print(f" Submission file saved → {SUB_PATH}")
    print(submission.head())
    print(f" Final rows: {len(submission)}")

if __name__ == "__main__":
    main()
