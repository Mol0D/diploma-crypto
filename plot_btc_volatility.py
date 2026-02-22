"""
Рисунок 1.3 — Реалізована волатильність BTC (2023–2025)
Ковзне стандартне відхилення денних лог-дохідностей на вікні 24 дні,
анналізована до річного масштабу.

Встановити залежності:
    pip install yfinance pandas matplotlib

Запуск:
    python plot_btc_volatility.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# ── Параметри ──────────────────────────────────────────────────────────────────
TICKER      = "BTC-USD"
START       = "2022-12-01"   # беремо з запасом щоб перші 24 дні не були NaN
END         = "2025-03-01"
WINDOW      = 24             # вікно rolling std (днів)
OUTPUT_FILE = "figure_1_3_btc_realized_volatility.png"

# Ключові події
EVENTS = [
    ("2023-03-10", "Крах SVB",             "above"),
    ("2023-11-10", "Річниця FTX",          "above"),
    ("2024-01-11", "Схвалення ETF",        "above"),
    ("2024-04-20", "Халвінг",              "below"),
    ("2024-08-05", "Крипто-крах\nсерпень", "above"),
    ("2024-11-06", "Перемога\nТрампа",     "above"),
]

# ── Завантаження денних даних ─────────────────────────────────────────────────
print("Завантажуємо денні дані BTC з Yahoo Finance...")
raw = yf.download(TICKER, start=START, end=END,
                  auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

close = raw["Close"].dropna()
print(f"Денних точок: {len(close)}, діапазон: {close.index[0].date()} – {close.index[-1].date()}")

# ── Розрахунок реалізованої волатильності ─────────────────────────────────────
log_returns = np.log(close / close.shift(1)).dropna()

# rolling std за 24 дні × √252 → річна волатильність у %
vol = log_returns.rolling(window=WINDOW).std() * np.sqrt(252) * 100
vol = vol.dropna()

# Обрізаємо до потрібного діапазону (без запасу)
vol = vol.loc["2023-01-01":]

print(f"Точок волатильності: {len(vol)}")

# ── Побудова графіку ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# Заливка під кривою
ax.fill_between(vol.index, vol.values, alpha=0.15, color="#E74C3C")

# Основна лінія
ax.plot(vol.index, vol.values,
        color="#E74C3C", linewidth=1.0, zorder=3, label="Реалізована волатильність (rolling 24d)")

# 30-денна ковзна середня — показує зміну режимів
ma30 = vol.rolling(30).mean()
ax.plot(ma30.index, ma30.values,
        color="#2C3E50", linewidth=2.2, linestyle="--",
        label="30-денна середня", zorder=4)

# Середній рівень за весь період
mean_vol = vol.mean()
ax.axhline(mean_vol, color="#7F8C8D", linewidth=1.0,
           linestyle=":", alpha=0.8, label=f"Середня: {mean_vol:.0f}%")

# ── Анотації подій ────────────────────────────────────────────────────────────
y_max   = vol.max()
y_min   = vol.min()
y_range = y_max - y_min

for date_str, label, pos in EVENTS:
    dt = pd.Timestamp(date_str)
    if not (vol.index[0] <= dt <= vol.index[-1]):
        continue

    idx     = vol.index.searchsorted(dt)
    idx     = min(idx, len(vol) - 1)
    vol_at  = vol.iloc[idx]

    if pos == "above":
        y_text  = y_max + y_range * 0.10
        y_arrow = vol_at + y_range * 0.02
    else:
        y_text  = y_min - y_range * 0.18
        y_arrow = vol_at - y_range * 0.02

    ax.annotate(
        label,
        xy=(dt, y_arrow),
        xytext=(dt, y_text),
        fontsize=8,
        ha="center",
        va="bottom" if pos == "above" else "top",
        color="#222222",
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#cccccc", lw=0.7, alpha=0.9),
        clip_on=False,
    )
    ax.axvline(dt, color="#bbbbbb", linewidth=0.8,
               linestyle="--", alpha=0.6, zorder=1)

# ── Оформлення ────────────────────────────────────────────────────────────────
# ax.set_title(
#     "Рисунок 1.3 — Реалізована волатильність BTC\n(rolling 24-денне std лог-дохідностей, анналізована, 2023–2025)",
#     fontsize=12, fontweight="bold", pad=14
# )
ax.set_ylabel("Річна волатильність (%)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
ax.set_ylim(max(0, y_min - y_range * 0.25), y_max + y_range * 0.25)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
print(f"\nГрафік збережено: {OUTPUT_FILE}")
plt.show()