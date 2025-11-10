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
    print("🚀 Generating submission file...")

    # Load model
    model = load(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")

    # Load test data
    test = pd.read_csv(TEST_PATH)
    ids = test["Id"].copy()
    print(f"📦 Test data loaded: {test.shape}")

    # --- Apply same feature engineering ---
    test_fe = feature_engineer(test.copy())
    print("🧠 Feature engineering applied.")

    # --- Restore any dropped rows (alignment safeguard) ---
    if len(test_fe) != len(ids):
        diff = len(ids) - len(test_fe)
        print(f"⚠️ {diff} rows dropped during feature engineering → restoring alignment...")
        test_fe = test_fe.reindex(range(len(ids)), fill_value=0)
        test_fe["Id"] = ids

    # --- Align columns with model training features ---
    expected_cols = getattr(model, "feature_names_in_", test_fe.columns)
    for col in expected_cols:
        if col not in test_fe.columns:
            test_fe[col] = 0
    test_fe = test_fe[expected_cols]

    # --- Predict ---
    preds = model.predict(test_fe)
    print(f"✅ Raw predictions complete ({len(preds)} values).")

    # --- Handle log-scaled training (reverse transform) ---
    # If model was trained on log(SalePrice), revert back
    if np.mean(preds) < 20:  # heuristic check for log-scale (~11-13 typical)
        print("🔄 Detected log-transformed target, applying expm1()...")
        preds = np.expm1(preds)

    # --- Guarantee correct submission length ---
    if len(preds) != len(ids):
        print(f"⚠️ Fixing prediction length mismatch ({len(preds)} vs {len(ids)})...")
        min_len = min(len(preds), len(ids))
        preds = preds[:min_len]
        ids = ids[:min_len]

    # --- Create submission file ---
    submission = pd.DataFrame({
        "Id": ids,
        "SalePrice": preds
    })

    # --- Save submission ---
    submission.to_csv(SUB_PATH, index=False)
    print(f"📁 Submission file saved → {SUB_PATH}")
    print(submission.head())
    print(f"✅ Final rows: {len(submission)}")

if __name__ == "__main__":
    main()
