"""
Fix submission scale for Kaggle (convert log predictions to real prices)
Author: Onur Tilki
"""

import pandas as pd
import numpy as np

# Load your last submission
sub_path = "submission_best.csv"
sub = pd.read_csv(sub_path)

print(f"Loaded {sub_path} → {len(sub)} rows")

# Convert log-scale SalePrice back to original
sub["SalePrice"] = np.expm1(sub["SalePrice"])

# Quick sanity check
mean_val = sub["SalePrice"].mean()
std_val = sub["SalePrice"].std()
print(f"New Mean: {mean_val:.2f} | Std: {std_val:.2f}")

# Save as new Kaggle-ready file
fixed_path = "submission_best_fixed.csv"
sub.to_csv(fixed_path, index=False)
print(f"✅ Fixed file saved: {fixed_path}")
