"""
model.py
---------
Model seçimi ve yapılandırması.
LinearRegression, RidgeCV, LassoCV ve ElasticNetCV modellerini destekler.
"""

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV 
"""
LinearRegression → temel model

RidgeCV → L2 regularization (aşırı öğrenmeyi dengeler)

LassoCV → L1 regularization (gereksiz feature’ları sıfırlar)

ElasticNetCV → L1 + L2 karışımı
"""

def get_model(kind: str = "linear"):
    """
    İstenen model türüne göre sklearn regresyon modelini döndürür.
    kind: 'linear', 'ridge', 'lasso', 'elastic'
    """
    kind = kind.lower()

    if kind == "linear":
        model= LinearRegression()
    elif kind == "ridge":
        model = RidgeCV(alphas=[0.1,1.0,10.0],cv=5)     # L2 regularization + 5-fold CV
    elif kind== "lasso":
        model =LassoCV(alphas=[0.001,0.01,1.0], cv=5 , max_iter= 10000) ## L1 regularization
    elif kind == "elastic":
        model = ElasticNetCV(l1_ratio=[0.3, 0.5, 0.7, 0.9], cv=5, max_iter=10000)  # L1 + L2 dengesi
    else:
        raise ValueError(f"bilinmeyen model türü: {kind}")
    
    print(f"model oluştu: {model.__class__.__name__}")
    return model


if __name__ == "__main__":
    for m in ["linear","ridge","lasso","elastic"]:
        model = get_model(m)
        print(f"{m} -> {type(model)}")
        