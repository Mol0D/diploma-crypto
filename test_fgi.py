import pandas as pd

fg = pd.read_csv("data/fear_greed.csv")
fg["timestamp"] = pd.to_datetime(fg["timestamp"])
fg["year_month"] = fg["timestamp"].dt.to_period("M")

print("=== Середній FGI по місяцях ===")
print(fg.groupby("year_month")["fear_greed"].mean().to_string())

print("\n=== Розподіл по зонах по роках ===")
bins = [0, 25, 45, 55, 75, 100]
labels = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
fg["zone"] = pd.cut(fg["fear_greed"], bins=bins, labels=labels)
print(fg.groupby(fg["timestamp"].dt.year)["zone"].value_counts().to_string())

print("\n=== Статистика варіації fg_delta24h ===")
fg = fg.sort_values("timestamp")
fg["fg_delta24h"] = fg["fear_greed"].diff(1)
print(fg.groupby(fg["timestamp"].dt.year)["fg_delta24h"].agg(["mean", "std", "min", "max"]))