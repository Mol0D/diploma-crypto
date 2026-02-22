"""
Рисунок 1.4 — ACF денних лог-дохідностей BTC: 2023 Q1 vs 2024 Q4

Завантажує денні дані через yfinance.

Встановити залежності:
    pip install yfinance pandas matplotlib statsmodels

Запуск:
    python plot_btc_acf.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# ── Параметри ──────────────────────────────────────────────────────────────────
TICKER      = "BTC-USD"
N_LAGS      = 30        # лагів (днів)
OUTPUT_FILE = "figure_1_4_btc_acf.png"

PERIODS = {
    "2023 Q1  (Jan–Mar 2023)": ("2023-01-01", "2023-03-31"),
    "2024 Q4  (Oct–Dec 2024)": ("2024-10-01", "2024-12-31"),
}
COLORS = {
    "2023 Q1  (Jan–Mar 2023)": "#2980B9",
    "2024 Q4  (Oct–Dec 2024)": "#E74C3C",
}

# ── Завантаження даних ─────────────────────────────────────────────────────────
print("Завантажуємо денні дані BTC з Yahoo Finance...")
raw = yf.download(TICKER, start="2022-12-01", end="2025-01-10",
                  auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

close = raw["Close"].dropna()
log_ret = np.log(close / close.shift(1)).dropna()

# ── Побудова графіку ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# fig.suptitle(
#     "Рисунок 1.4 — ACF денних лог-дохідностей BTC: 2023 Q1 vs 2024 Q4",
#     fontsize=12, fontweight="bold", y=1.02
# )

for ax, (label, (start, end)) in zip(axes, PERIODS.items()):
    color = COLORS[label]
    sub = log_ret.loc[start:end].dropna()
    print(f"  {label.strip()}: {len(sub)} спостережень")

    acf_vals, confint = acf(sub, nlags=N_LAGS, alpha=0.05, fft=True)
    lags = np.arange(len(acf_vals))
    ci_bound = 1.96 / np.sqrt(len(sub))

    # стовпці ACF (пропускаємо lag=0, він завжди 1)
    ax.bar(lags[1:], acf_vals[1:], color=color, alpha=0.75,
           width=0.6, zorder=3)

    # довірчий інтервал
    ax.axhline( ci_bound, color="#444444", linewidth=1.3,
                linestyle="--", alpha=0.8, label="95% CI")
    ax.axhline(-ci_bound, color="#444444", linewidth=1.3,
                linestyle="--", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)

    # підпис значущих лагів
    for lag in range(1, N_LAGS + 1):
        val = acf_vals[lag]
        if abs(val) > ci_bound * 1.5:   # підписуємо лише помітно значущі
            ax.annotate(
                f"lag {lag}\n{val:+.3f}",
                xy=(lag, val),
                xytext=(lag + 1.2, val + (0.018 if val > 0 else -0.022)),
                fontsize=7.5,
                color=color,
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.7),
            )

    ax.set_title(label, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Лаг (дні)", fontsize=11)
    ax.set_xlim(0, N_LAGS + 1)
    ax.set_xticks(range(0, N_LAGS + 1, 5))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")

    # статистика у кутку
    n_sig = sum(abs(acf_vals[1:]) > ci_bound)
    ax.text(0.02, 0.97,
            f"n = {len(sub)} днів\nЗначущих лагів: {n_sig}/{N_LAGS}",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#cccccc", lw=0.7, alpha=0.9))

axes[0].set_ylabel("Автокореляція", fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
print(f"\nГрафік збережено: {OUTPUT_FILE}")
plt.show()