"""
Multi-model trainer (Ridge, Lasso, ElasticNet)
Auto-selects best model and generates Kaggle-ready submission.
"""

import pandas as pd
import numpy as np
from joblib import load
from src.features import feature_engineer
from src.model import get_model
from src.preprocess import build_preprocessor
from src.train import train_and_save

DATA_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODELS = ["ridge", "lasso", "elastic"]

def train_and_eval_all():
    print("🚀 Starting multi-model training...")

    # --- load & prepare ---
    train_df = pd.read_csv(DATA_PATH)
    train_df = feature_engineer(train_df)
    print(f"📦 Training data loaded: {train_df.shape}")

    # --- train all models ---
    model_scores = {}
    for model_name in MODELS:
        print(f"\n🏗️ Training model → {model_name.upper()}")
        model = get_model(model_name)

        num_cols = train_df.select_dtypes(include=["int", "float"]).columns.drop("SalePrice")
        cat_cols = train_df.select_dtypes(include=["object"]).columns
        preprocessor = build_preprocessor(num_cols, cat_cols)

        trained_model, metrics = train_and_save(train_df, preprocessor, model, "SalePrice")
        model_scores[model_name] = metrics["rmse"]
        print(f"✅ {model_name.upper()} trained | RMSE: {metrics['rmse']:.4f}")

    best_model = min(model_scores, key=model_scores.get)
    print(f"\n🏆 Best model: {best_model.upper()} (RMSE={model_scores[best_model]:.4f})")

    # --- predictions ---
    model = load("models/houseprice.joblib")
    test = pd.read_csv(TEST_PATH)
    ids = test["Id"]
    test_fe = feature_engineer(test.copy())
    print("🧠 Feature engineering applied to test set.")

    if len(test_fe) != len(ids):
        print(f"⚠️ Fixing dropped rows ({len(test_fe)} vs {len(ids)})")
        test_fe = test_fe.reindex(range(len(ids)), fill_value=0)

    expected_cols = getattr(model, "feature_names_in_", test_fe.columns)
    for c in expected_cols:
        if c not in test_fe.columns:
            test_fe[c] = 0
    test_fe = test_fe[expected_cols]

    preds = np.expm1(model.predict(test_fe))  # log dönüşümünü geri al ✅

    submission = pd.DataFrame({"Id": ids, "SalePrice": preds})
    submission.to_csv("submission_best.csv", index=False)
    print(f"📁 submission_best.csv saved | rows={len(submission)}")

if __name__ == "__main__":
    train_and_eval_all()
