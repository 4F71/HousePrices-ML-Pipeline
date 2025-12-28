"""
Fix submission scale for Kaggle (convert log predictions to real prices)
Author: Onur Tilki
"""

import pandas as pd
import numpy as np

sub_path = "submission_best.csv"
sub = pd.read_csv(sub_path)

print(f"Loaded {sub_path} → {len(sub)} rows")


sub["SalePrice"] = np.expm1(sub["SalePrice"])


mean_val = sub["SalePrice"].mean()
std_val = sub["SalePrice"].std()
print(f"New Mean: {mean_val:.2f} | Std: {std_val:.2f}")


fixed_path = "submission_best_fixed.csv"
sub.to_csv(fixed_path, index=False)
print(f" Fixed file saved: {fixed_path}")
