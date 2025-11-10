import pandas as pd
import numpy as np

sub = pd.read_csv("submission_best.csv")
sale_mean = sub["SalePrice"].mean()
sale_std = sub["SalePrice"].std()

print(f"Rows: {len(sub)}")
print(f"Mean: {sale_mean:.3f}, Std: {sale_std:.3f}")

if sale_mean < 20:
    print("⚠️ Tahminler log-ölçeğinde (Kaggle hatası sebebi bu).")
else:
    print("✅ Tahminler gerçek fiyat ölçeğinde. Kaggle doğru skor verecek.")
