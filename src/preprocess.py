"""
preprocess.py
--------------
Sayısal ve kategorik değişkenler için ön işleme pipeline'ı oluşturur.
Eksik değerleri doldurur, sayısal verileri ölçekler, kategorik verileri OneHotEncode eder.
"""
from sklearn.pipeline import Pipeline  # pipeline adım zincirleri oluşturmak için
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder #ölçekleme
from sklearn.impute import SimpleImputer #eksik veri doldurma

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Sayısal ve kategorik sütunlar için ön işleme pipeline'ı kurar.
    Numeric → SimpleImputer + StandardScaler
    Categorical → SimpleImputer + OneHotEncoder
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")), #eksik değerleri ortalama ile doldurma
        ("scaler",StandardScaler())                  #ölçekleme
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # en sık görülen değerle doldur
        ("encoder", OneHotEncoder(handle_unknown="ignore"))    # bilinmeyen kategorileri yoksay
    ]) 

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor

if __name__ == "__main__":
    import pandas as pd
    from src.data import load_data
    df = load_data("train.csv")
    numeric = df.select_dtypes(include=["int64","float64"]). columns.tolist()
    categorial = df.select_dtypes(include=[object]).columns.tolist()  
    pre = build_preprocessor(numeric,categorial)
    print("Sayısal sutün sayısı",len(numeric))
    print("kategorik sutün sayısı",len(categorial))
    print("preprocessor hazır",type(pre))
