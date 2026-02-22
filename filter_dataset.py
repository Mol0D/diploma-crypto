import pandas as pd

# Обрізаємо dataset до кінця грудня 2025
ds = pd.read_csv("data/dataset_1h.csv")
ds["timestamp"] = pd.to_datetime(ds["timestamp"], utc=True)

print(f"До обрізання: {len(ds)} rows")
print(f"Період: {ds['timestamp'].min()} — {ds['timestamp'].max()}")

ds_filtered = ds[ds["timestamp"] < "2025-02-01"]

print(f"\nПісля обрізання: {len(ds_filtered)} rows")
print(f"Період: {ds_filtered['timestamp'].min()} — {ds_filtered['timestamp'].max()}")

ds_filtered.to_csv("data/dataset_1h_2025.csv", index=False)
print("\nСтворено: data/dataset_1h_2025.csv")