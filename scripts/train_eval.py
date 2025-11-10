"""
train_eval.py
--------------
HousePrices pipeline'ını uçtan uca çalıştırır:
1. Veriyi yükler ve feature engineering uygular.
2. Preprocessor + model pipelineını kurar.
3. Modeli eğitir ve kaydeder.
4. Tahmin yapar, metrikleri hesaplar ve rapora yazar.
Kullanım:
    python -m scripts.train_eval --model ridge
"""
import argparse
import joblib
from src.data import load_data
from src.features import feature_engineer
from src.preprocess import build_preprocessor
from src.model import get_model
from src.train import train_and_save
from src.eval import evaluate
from src.paths import MODELS_DIR

def main(model_kind="ridge"):
    df = load_data("train.csv")
    df = feature_engineer(df)

    numeric = df.select_dtypes(include=["int32", "int64", "float32", "float64"]).columns.tolist()
    categorical = df.select_dtypes(include=["object"]).columns.tolist()

    if "SalePrice" in numeric: numeric.remove("SalePrice")
    if "SalePrice" in categorical: categorical.remove("SalePrice")

    pre = build_preprocessor(numeric, categorical)
    model = get_model(model_kind)

    print(f"🚀 Eğitim başlıyor → {model_kind.upper()}")
    train_and_save(df, pre, model)

    model_path = MODELS_DIR / "houseprice.joblib"
    pipe = joblib.load(model_path)
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]
    y_pred = pipe.predict(X)
    evaluate(y, y_pred, model_name=model_kind)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ridge", help="Model türü: linear | ridge | lasso | elastic")
    args = parser.parse_args()
    main(args.model)
