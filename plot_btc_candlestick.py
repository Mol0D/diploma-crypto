"""
Рисунок 1.2 — Динаміка ціни BTC (свічковий графік) та обсягів торгів за 2023–2025 рр.

Завантажує дані з Yahoo Finance, агрегує до тижневих свічок.

Встановити залежності:
    pip install yfinance pandas matplotlib mplfinance

Запуск:
    python plot_btc_candlestick.py
"""

import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Параметри ──────────────────────────────────────────────────────────────────
TICKER      = "BTC-USD"
START       = "2023-01-01"
END         = "2025-03-01"
OUTPUT_FILE = "figure_1_2_btc_candlestick.png"

# Ключові події: (дата, підпис, "above"/"below")
# чергуємо above/below щоб підписи не накладались
EVENTS = [
    ("2023-03-10", "Крах SVB",              "below"),
    ("2023-06-15", "ETF-заявки\nBlackRock", "above"),
    ("2024-01-11", "Схвалення\nBTC ETF",    "above"),
    ("2024-04-20", "Халвінг BTC",           "below"),
    ("2024-11-06", "Перемога\nТрампа",      "above"),
]

# ── Завантаження та підготовка даних ───────────────────────────────────────────
print("Завантажуємо дані з Yahoo Finance...")
raw = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)

# yfinance іноді повертає MultiIndex колонки
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

# Агрегуємо до тижневих свічок — читабельно на 2+ роки
weekly = raw.resample("W").agg({
    "Open":   "first",
    "High":   "max",
    "Low":    "min",
    "Close":  "last",
    "Volume": "sum",
}).dropna()

print(f"Тижневих свічок: {len(weekly)}, діапазон: {weekly.index[0].date()} – {weekly.index[-1].date()}")

# ── Стиль ─────────────────────────────────────────────────────────────────────
mc = mpf.make_marketcolors(
    up="#2ECC71", down="#E74C3C",
    edge="inherit", wick="inherit",
    volume={"up": "#2ECC71", "down": "#E74C3C"},
)
style = mpf.make_mpf_style(
    base_mpf_style="charles",
    marketcolors=mc,
    gridcolor="#e8e8e8",
    gridstyle="--",
    facecolor="white",
    figcolor="white",
    rc={"font.family": "DejaVu Sans", "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9},
)

# ── Вертикальні лінії подій ────────────────────────────────────────────────────
vline_dates = []
for date_str, _, _ in EVENTS:
    dt = pd.Timestamp(date_str)
    if weekly.index[0] <= dt <= weekly.index[-1]:
        vline_dates.append(dt)

# ── Побудова графіку ───────────────────────────────────────────────────────────
fig, axes = mpf.plot(
    weekly,
    type="candle",
    volume=True,
    style=style,
    title="",
    ylabel="Ціна BTC (USD)",
    ylabel_lower="Обсяг (BTC)",
    figsize=(14, 7),
    vlines=dict(
        vlines=vline_dates,
        linewidths=1.0,
        linestyle="--",
        colors="#aaaaaa",
        alpha=0.8,
    ),
    tight_layout=True,
    returnfig=True,
)

ax_price = axes[0]
ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# ── Текстові анотації подій ────────────────────────────────────────────────────
y_min = weekly["Low"].min()
y_max = weekly["High"].max()
y_range = y_max - y_min

for date_str, label, pos in EVENTS:
    dt = pd.Timestamp(date_str)
    if not (weekly.index[0] <= dt <= weekly.index[-1]):
        continue

    x_pos = weekly.index.searchsorted(dt)
    x_pos = min(x_pos, len(weekly) - 1)
    price_at = weekly["High"].iloc[x_pos]

    if pos == "above":
        y_text  = y_max + y_range * 0.06
        y_arrow = price_at + y_range * 0.01
    else:
        y_text  = y_min - y_range * 0.12
        y_arrow = price_at - y_range * 0.01

    ax_price.annotate(
        label,
        xy=(x_pos, y_arrow),
        xytext=(x_pos, y_text),
        fontsize=8,
        ha="center",
        va="bottom" if pos == "above" else "top",
        color="#222222",
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.7, alpha=0.9),
        clip_on=False,
    )

fig.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
print(f"\nГрафік збережено: {OUTPUT_FILE}")
plt.show()