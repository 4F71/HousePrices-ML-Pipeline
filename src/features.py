"""
features.py
-----------
-mevcut veriden anlamlı veriler üretmek için oluşturulmuştur. 
-yeni özellikler (feature) üretir outlier filtreleme ve log dönüşü uygular
"""

import pandas as pd 
import numpy as np

def feature_engineer(df:pd.DataFrame) -> pd.DataFrame: #ham veriye yeni featureler eklemek için oluşturulmuş fonksiyon
    df = df.copy() #orjinal veriyi korumak için checkpoint oluşturma
    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]  # toplam yaşam alanı # modelin korelasyon gücünü arttırmak için
    df["BathCount"] = df["FullBath"] + 0.5 * df["HalfBath"]              # toplam banyo sayısı
    df["Age"] = 2020 - df["YearBuilt"]                                   #evin yaşı

    skewed_cols= ["GrLivArea", "TotalSF", "SalePrice"] #01_eda.ipynb'deki başta görünen sağa çarpık sütunlar: 
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col]) #indexleme yöntemi sayesinde col değişkeni dinamik olur

    if 'GrLivArea' in df.columns:
        df=df[df["GrLivArea"]<np.log1p(4500)] #stabil skor ölçmek için filtre ve log uygulandı
    
    return df # dönüştürülen veriyi döndürme

if __name__ == "__main__": #name değişkeni her fonksiyonu bağımsız test etmesi içindir
    from src.data import load_data
    df = load_data("train.csv")
    df_re=feature_engineer(df)
    print("Yeni shape", df_re.shape) #shape ile satır sutun sayısını döndürürüz.
