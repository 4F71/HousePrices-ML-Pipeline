# 🏠 HousePrices ML Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.7.2-orange?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/RMSLE(Kaggle)-0.13049-success?style=flat-square" alt="Kaggle Score"/>
  <img src="https://img.shields.io/badge/Model-ElasticNetCV-brightgreen?style=flat-square" alt="Model"/>
  <img src="https://img.shields.io/badge/Lisans-MIT-yellow?style=flat-square" alt="License"/>
</p>

<p align="center">
  <strong>Üretim seviyesinde, modüler makine öğrenimi pipeline'ı - Kaggle House Prices regresyon projesi</strong><br>
  <em>Uçtan uca iş akışı: Veri → Feature Engineering → Ön İşleme → Model → Değerlendirme → Submission</em>
</p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques" target="_blank">
    🔗 Kaggle Yarışması Sayfası
  </a> •
  <a href="#-özellikler">Özellikler</a> •
  <a href="#-pipeline-mimarisi">Mimari</a> •
  <a href="#-kullanım">Kullanım</a>
</p>

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Repo'yu klonla
git clone https://github.com/4F71/HousePrices-ML-Pipeline.git
cd HousePrices-ML-Pipeline

# 2. Ortamı hazırla
python -m venv .venv
.venv\Scripts\activate   # (Windows)
pip install -r requirements.txt

# 3. Modeli eğit
python -m scripts.train_all

# 4. Kaggle submission dosyasını düzelt
python -m scripts.fix_submission_scale
Sonuç:

Eğitilmiş model: models/houseprice.joblib

Tahmin dosyası: submission_best_fixed.csv

Kaggle Skoru: RMSLE = 0.13049

📊 Model Performansı (Yerel)
Metrik	Skor
R²	0.9235
RMSE (log)	0.0088
Kaggle RMSLE	0.13049

ElasticNetCV modeli, Ridge ve Lasso arasında denge sağlayarak %92.35 açıklama gücü elde etti.
Bu pipeline, Kaggle'da sağlam bir Level-1 baseline performansına sahiptir.

<p align="center"> <img src="figures/importance.png" alt="Feature Importance" width="700"/> </p>
📂 Veri Seti
Bu proje Kaggle House Prices: Advanced Regression Techniques yarışmasının resmi verisini kullanır.

📎 Veri Seti Linki

Yapı:

train.csv → 1460 örnek, 81 özellik

test.csv → 1459 örnek (submission için)

Verileri data/ dizinine yerleştirerek pipeline’ı doğrudan çalıştırabilirsin.

✨ Ek Özellikler (Kaggle için optimize)
Log dönüşümü (SalePrice) otomatik yönetilir (log1p / expm1)

3 model eğitimi: RidgeCV, LassoCV, ElasticNetCV

En iyi model otomatik seçilip submission_best.csv oluşturulur

Kaggle formatında (1459 satır, Id + SalePrice) CSV çıkışı

Hatalı ölçek koruması: scripts/check_scale.py

🎯 Kaggle Submission Adımları
bash
Kodu kopyala
# Submission dosyasını doğrula
python check_scale.py

# Çıktı:
# ✅ Tahminler gerçek fiyat ölçeğinde. Kaggle doğru skor verecek.

# Dosyayı Kaggle’a yükle
# (Tarayıcıdan veya CLI ile)
kaggle competitions submit \
  -c house-prices-advanced-regression-techniques \
  -f submission_best_fixed.csv \
  -m "ElasticNetCV – baseline 0.13049"
📜 Lisans
MIT Lisansı © 2025 Onur Tilki

<p align="center"> <strong>⭐ Faydalı bulduysanız yıldız vermeyi düşünün!</strong><br> Kaggle versiyonu: <em>ElasticNetCV v1.0 (baseline 0.13049)</em> </p> ```
