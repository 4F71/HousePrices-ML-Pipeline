"""
visualize.py
-------------
Eğitilmiş modelin feature importance değerlerini görselleştirir.
Sonuç figures/importance.png olarak kaydedilir.
"""
import matplotlib.pyplot as plt
import numpy as np
import joblib
from src.paths import FIGURES_DIR, MODELS_DIR

def plot_feature_importance(top_n=15):
    """
    Ridge/Lasso gibi lineer modellerde katsayı büyüklüklerine göre
    en etkili feature'ları bar grafiği olarak kaydeder.
    """
    model_path = MODELS_DIR / "houseprice.joblib"
    pipe = joblib.load(model_path)

    # pipeline -> preprocessor + model
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["preprocessor"]

    # sadece linear tip modellerde coefficient bulunur
    if not hasattr(model, "coef_"):
        print("⚠️ Bu model feature importance desteği sağlamıyor.")
        return

    # Preprocessor içindeki OneHotEncoder sonrası feature adlarını çıkar
    num_features = pre.transformers_[0][2]
    cat_features = pre.transformers_[1][1]["encoder"].get_feature_names_out(pre.transformers_[1][2])
    feature_names = np.concatenate([num_features, cat_features])

    # Önem değerlerini al
    coefs = np.abs(model.coef_)
    indices = np.argsort(coefs)[-top_n:]  # en yüksek n
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), coefs[indices], color="steelblue")
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel("Coefficient Magnitude")
    plt.title(f"Top {top_n} Feature Importances ({model.__class__.__name__})")

    save_path = FIGURES_DIR / "importance.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 Feature importance grafiği kaydedildi: {save_path}")

if __name__ == "__main__":
    plot_feature_importance()
