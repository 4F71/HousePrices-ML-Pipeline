"""
train.py
--------
Veriyi train/test olarak böler, pipeline'ı eğitir ve modeli kaydeder.
"""

from sklearn.model_selection import train_test_split
import joblib #model kayıt etmek için kullanılan:
from src.paths import MODELS_DIR


def train_and_save(df, preprocessor, model, target_col="SalePrice"):
    """
    Veriyi train/test olarak böler, pipeline'ı eğitir ve modeli kaydeder.
    """

    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score,mean_squared_error
    from math import sqrt 
    import numpy as np
    from src.pipeline import build_pipeline 

    #veri bölme adımları: 

    x=df.drop(columns=[target_col]) #özellikler
    y=df[target_col]               #HEDEF DEĞİŞKEN
    x_train, x_test , y_train, y_test = train_test_split(x,y, test_size=0.2 , random_state=42)

    #pipeline oluşturma

    pipe=build_pipeline (preprocessor,model)
    pipe.fit(x_train,y_train)

    #skorlar

    y_pred = pipe.predict(x_test)
    r2= r2_score(y_test,y_pred)
    rmse =sqrt(mean_squared_error(y_test,y_pred))

    #model kayıt

    model_path=MODELS_DIR / "houseprice.joblib"
    joblib.dump(pipe, model_path)


    print(f"model kayıt edildi: {model_path}")
    print(f"r2:  {r2:.4f}| RMSE: {rmse:.4f}")


if __name__ == "__main__":  # test modu
    from src.data import load_data
    from src.features import feature_engineer
    from src.preprocess import build_preprocessor
    from src.model import get_model

    # --- veri yükleme ve hazırlık ---
    df = load_data("train.csv")
    df = feature_engineer(df)

    numeric = df.select_dtypes(include=["int32", "int64", "float32", "float64"]).columns.tolist()
    categorical = df.select_dtypes(include=["object"]).columns.tolist()

    # hedef değişkeni listelerden çıkar
    if "SalePrice" in numeric:
        numeric.remove("SalePrice")
    if "SalePrice" in categorical:
        categorical.remove("SalePrice")


    pre = build_preprocessor(numeric, categorical)
    model = get_model("ridge")

    train_and_save(df, pre, model)
