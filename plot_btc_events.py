"""
Рисунок 1.5 — Реакція ціни BTC на ключові події (погодинні свічки)

Встановити залежності:
    pip install yfinance pandas matplotlib

Запуск:
    python plot_btc_events.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

OUTPUT_FILE = "figure_1_5_btc_events.png"

EVENTS = [
    {
        "label":       "11 січня 2024\nСхвалення\nBitcoin Spot ETF",
        "event_dt":    pd.Timestamp("2024-01-11"),
        "start":       "2024-01-08",
        "end":         "2024-01-16",
        "event_color": "#27AE60",
        "panel_title": "А) Позитивна подія — Схвалення Bitcoin Spot ETF (11 січня 2024)",
    },
    {
        "label":       "5 серпня 2024\nКрипто-крах\n(макро-паніка)",
        "event_dt":    pd.Timestamp("2024-08-05"),
        "start":       "2024-08-01",
        "end":         "2024-08-10",
        "event_color": "#C0392B",
        "panel_title": "Б) Негативна подія — Крипто-крах серпня 2024 (5 серпня 2024)",
    },
]

# ── Завантаження: спробуємо кілька способів ────────────────────────────────────
def download_hourly(start, end):
    ticker = yf.Ticker("BTC-USD")

    # Спосіб 1: history() з конкретними датами
    try:
        df = ticker.history(start=start, end=end, interval="1h", auto_adjust=True)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        if len(df) > 0:
            print(f"    Спосіб 1 (ticker.history) — {len(df)} свічок")
            return df
    except Exception as e:
        print(f"    Спосіб 1 помилка: {e}")

    # Спосіб 2: download() з конкретними датами
    try:
        raw = yf.download("BTC-USD", start=start, end=end,
                          interval="1h", auto_adjust=True,
                          progress=False, timeout=20)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open","High","Low","Close","Volume"]].dropna()
        if len(df) > 0:
            print(f"    Спосіб 2 (download) — {len(df)} свічок")
            return df
    except Exception as e:
        print(f"    Спосіб 2 помилка: {e}")

    # Спосіб 3: завантажуємо більший діапазон і фільтруємо
    try:
        raw = yf.download("BTC-USD", period="730d", interval="1h",
                          auto_adjust=True, progress=False, timeout=20)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open","High","Low","Close","Volume"]].loc[start:end].dropna()
        if len(df) > 0:
            print(f"    Спосіб 3 (period=730d + фільтр) — {len(df)} свічок")
            return df
    except Exception as e:
        print(f"    Спосіб 3 помилка: {e}")

    return pd.DataFrame()

print("Завантажуємо погодинні дані BTC...")
for ev in EVENTS:
    print(f"  {ev['label'].splitlines()[0]}:")
    ev["df"] = download_hourly(ev["start"], ev["end"])
    if len(ev["df"]) == 0:
        print(f"\n  ПОМИЛКА: не вдалось завантажити дані.")
        print(f"  Перевір версію yfinance: pip show yfinance")
        print(f"  Спробуй оновити: pip install --upgrade yfinance")
        exit(1)
    print(f"    Діапазон: {ev['df'].index[0].strftime('%Y-%m-%d %H:%M')} – "
          f"{ev['df'].index[-1].strftime('%Y-%m-%d %H:%M')}")

# ── Функція малювання панелі ───────────────────────────────────────────────────
def draw_event(ax_price, ax_vol, df, event_dt, label, panel_title, event_color):
    dates = mdates.date2num(df.index.to_pydatetime())
    diffs = np.diff(dates)
    width = float(diffs[diffs > 0].min()) * 0.72 if len(diffs) > 0 else 0.03

    y_max = float(df["High"].max())
    y_min = float(df["Low"].min())
    y_rng = y_max - y_min

    for d, row in zip(dates, df.itertuples()):
        up    = row.Close >= row.Open
        color = "#2ECC71" if up else "#E74C3C"
        bot   = min(row.Open, row.Close)
        h     = max(abs(row.Close - row.Open), y_rng * 0.001)
        ax_price.bar(d, h, bottom=bot, width=width, color=color, zorder=3)
        ax_price.plot([d, d], [row.Low, row.High],
                      color=color, linewidth=0.9, zorder=2)

    vol_colors = ["#2ECC71" if row.Close >= row.Open else "#E74C3C"
                  for row in df.itertuples()]
    ax_vol.bar(dates, df["Volume"].values, width=width,
               color=vol_colors, alpha=0.7)

    ev_num = mdates.date2num(event_dt.to_pydatetime())
    for ax in (ax_price, ax_vol):
        ax.axvline(ev_num, color=event_color, linewidth=2.2,
                   linestyle="-", alpha=0.85, zorder=5)

    x_shift = (dates[-1] - dates[0]) * 0.03
    ax_price.annotate(
        label,
        xy=(ev_num, y_max),
        xytext=(ev_num + x_shift, y_max + y_rng * 0.05),
        fontsize=8.5, ha="left", fontweight="bold", color=event_color,
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec=event_color, lw=1.1, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color=event_color, lw=0.9),
        clip_on=False,
    )

    ax_price.text((dates[0] + ev_num) / 2,
                  y_min + y_rng * 0.05, "← до події",
                  ha="center", fontsize=8, color="#95A5A6")
    ax_price.text((ev_num + dates[-1]) / 2,
                  y_min + y_rng * 0.05, "після події →",
                  ha="center", fontsize=8, color="#95A5A6")

    ax_price.set_title(panel_title, fontsize=10.5, fontweight="bold",
                       pad=8, loc="left")
    ax_price.set_ylabel("Ціна BTC (USD)", fontsize=9)
    ax_price.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_price.grid(axis="y", linestyle="--", alpha=0.4)
    ax_price.spines[["top", "right"]].set_visible(False)
    plt.setp(ax_price.get_xticklabels(), visible=False)

    ax_vol.set_ylabel("Обсяг", fontsize=9)
    ax_vol.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _:
            f"{x/1e9:.1f}B" if x >= 1e9 else
            f"{x/1e6:.0f}M" if x >= 1e6 else
            f"{x/1e3:.0f}K"))
    ax_vol.grid(axis="y", linestyle="--", alpha=0.3)
    ax_vol.spines[["top", "right"]].set_visible(False)
    ax_vol.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

# ── Головна фігура ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    4, 1, figsize=(14, 11),
    gridspec_kw={"height_ratios": [3, 1, 3, 1], "hspace": 0.12},
)
fig.patch.set_facecolor("white")

for i, ev in enumerate(EVENTS):
    draw_event(
        axes[i*2], axes[i*2+1],
        ev["df"], ev["event_dt"],
        ev["label"], ev["panel_title"], ev["event_color"],
    )

fig.add_artist(plt.Line2D(
    [0.04, 0.96], [0.505, 0.505],
    transform=fig.transFigure,
    color="#cccccc", linewidth=1.2, linestyle="--",
))

plt.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
print(f"\nГрафік збережено: {OUTPUT_FILE}")
plt.show()