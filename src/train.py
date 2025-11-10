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
    Veriyi train/test olarak böler, pipeline'ı eğitir, log dönüşümüyle modeli kaydeder.
    Geriye (pipeline, metrics_dict) döndürür.
    """

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from math import sqrt
    import numpy as np
    import joblib
    from src.pipeline import build_pipeline
    from src.paths import MODELS_DIR

    # --- veri bölme ---
    X = df.drop(columns=[target_col])
    y = np.log1p(df[target_col])  # log dönüşümü ✅
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- pipeline oluşturma ---
    pipe = build_pipeline(preprocessor, model)
    pipe.fit(X_train, y_train)

    # --- değerlendirme ---
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    # --- model kaydet ---
    model_path = MODELS_DIR / "houseprice.joblib"
    joblib.dump(pipe, model_path)

    print(f"model kayıt edildi: {model_path}")
    print(f"r2: {r2:.4f} | RMSE: {rmse:.4f}")

    metrics = {"r2": r2, "rmse": rmse}
    return pipe, metrics
