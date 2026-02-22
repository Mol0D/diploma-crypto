import joblib
import pandas as pd
import numpy as np

# Аналіз feature importance для Combined моделі, горизонт 4h, tau=0.50
bundle = joblib.load("models/lgbm_combined_4h.joblib")

model = bundle["models"][0.5]
feature_cols = bundle["feature_cols"]

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("=== Feature Importance — Combined 4h (tau=0.50) ===")
print(importance.to_string(index=False))

print("\n=== Топ-10 ===")
print(importance.head(10).to_string(index=False))

print("\n=== FGI ознаки окремо ===")
fgi = importance[importance["feature"].str.startswith(("fg_", "fear_", "extreme_"))]
print(fgi.to_string(index=False))

print(f"\nЧастка FGI у загальній важливості: {fgi['importance'].sum() / importance['importance'].sum() * 100:.1f}%")