"""
data.py
---------
'train.csv' & 'test.csv' dosyalarını 'data' klasöründen okuyacak
-proje içi merkezi veri erişimi sağlar
-data frame'yi eda.ipynb'den ayıran şey: tek sorumluluk ilkesi taşımasıdır.
-ortam bağımsızlığı için oluşturulmuştur, yol yönetimi 'src.paths' üzerinden yapılacak

"""


import pandas as pd #veri okumak ve yönetmek için pandas kütüphanesi kullanılıyor

from src.paths import DATA_DIR #diğer modüllerin doğrudan dosyaya ulaşmaması için tanımlanan dosya yolu

def load_data(filename: str) -> pd.DataFrame: 
    path= DATA_DIR/filename
    df = pd.read_csv(path)
    print(f"{filename} yüklendi -> {df.shape}")
    return df

#->" sembolü fonksiyon dönüşüm tipidir
#tapıl değiştirilmeyen liste türüdür: iki defa pd.dataframe yazılmasının sebebiyse train/test olarak ayıracak olmamız.
def load_train_test() -> tuple[pd.DataFrame,pd.DataFrame]:
    train = load_data("train.csv")
    test = load_data("test.csv")
    return train, test  #tuple sayesinde birden fazla değeri tek seferde döndürebiliz

def optimize_dtypes(df:pd.DataFrame) ->pd.DataFrame: #bellek optimizasyonu
    for col in df.select_dtypes(include="float64"): 
        df[col] = df[col].astype("float32") #eğitimi hızlandırmak, ram kullanımı azaltmak için optimizasyon
    for col in df.select_dtypes(include="int64"):
        df[col] = df[col].astype("int32") #eğitimi hızlandırmak, ram kullanımı azaltmak için optimizasyon


if __name__ == "__main__":
    load_data("train.csv")

