"""
eval.py
--------
Modelin performans metriklerini (R², RMSE) hesaplar ve raporlar.
"""
import json
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from src.paths import REPORTS_DIR

def evaluate(y_true, y_pred, model_name="ridge"):
    """
    Gerçek ve tahmin değerlerine göre R² ve RMSE metriklerini hesaplayacak.
    Sonucu reports/metrics.json dosyasına kaydecek.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))

    metrics = {
        "model": model_name,
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4)
    }

    report_path = REPORTS_DIR / "metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Metrikler kaydedildi: {report_path}")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":  
    import joblib
    import pandas as pd
    from src.data import load_data
    from src.features import feature_engineer

    #  model  yükle
    model_path = REPORTS_DIR.parent / "models" / "houseprice.joblib"
    model = joblib.load(model_path)
    df = feature_engineer(load_data("train.csv"))

    #  değerlendirme
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]
    y_pred = model.predict(X)
    evaluate(y, y_pred, model_name="ridge")
