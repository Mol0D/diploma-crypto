import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Crypto Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

COLORS = {
    "market": "#1f77b4",
    "events": "#ff7f0e",
    "combined": "#2ca02c",
}

FGI_PREFIXES = ("fg_", "fear_greed", "extreme_")


def is_fgi(col: str) -> bool:
    return any(col.startswith(p) for p in FGI_PREFIXES)


@st.cache_data
def load_summary(horizon: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / f"eval_{horizon}_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_fold_results(horizon: str, cfg: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / f"eval_{horizon}_{cfg}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    path = DATA_DIR / "dataset_1h.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


@st.cache_resource
def load_model(cfg: str, horizon: str) -> dict | None:
    path = MODELS_DIR / f"lgbm_{cfg}_{horizon}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Налаштування")
    horizon = st.selectbox("Горизонт прогнозу", ["4h", "1h"], key="horizon")
    st.divider()
    selected_cfgs = st.multiselect(
        "Конфігурації (вкладка 2)",
        ["market", "events", "combined"],
        default=["market", "events", "combined"],
    )
    st.divider()
    symbol = st.selectbox("Символ (вкладка 4)", ["BTC", "ETH"], key="symbol")
    horizon4 = st.selectbox("Горизонт (вкладка 4)", ["4h", "1h"], key="horizon4")
    cfg4 = st.selectbox(
        "Конфігурація (вкладка 4)", ["combined", "market", "events"], key="cfg4"
    )
    n_pts = st.slider(
        "Кількість точок (вкладка 4)", min_value=100, max_value=500, value=168, step=1
    )

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Огляд експерименту",
        "📈 Результати по фолдах",
        "🔍 Важливість ознак",
        "🔮 Прогноз vs Реальність",
    ]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Огляд експерименту
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header(f"Огляд експерименту — горизонт {horizon}")
    st.markdown(
        "Порівняння трьох конфігурацій ознак: **Market** / **Events** / **Combined**"
    )

    summary = load_summary(horizon)
    if summary is None:
        st.warning(
            f"Файл `reports/eval_{horizon}_summary.csv` не знайдено. "
            "Запустіть `07_compare_feature_sets.py`."
        )
        st.stop()

    # ── Розрахунок Δ vs Market ──
    market_row = summary[summary["set"] == "market"]
    combined_row = summary[summary["set"] == "combined"]

    pinball_market = float(market_row["pinball_q0.50"].iloc[0])
    pinball_combined = float(combined_row["pinball_q0.50"].iloc[0])
    coverage_combined = float(combined_row["pi_90_coverage"].iloc[0])
    width_combined = float(combined_row["pi_90_avg_width"].iloc[0])

    delta_pct = (pinball_combined - pinball_market) / pinball_market * 100

    # ── st.metric ────────────────────────────────────────────────────────────
    st.subheader("Ключові метрики Combined vs Market")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Покращення Combined vs Market (Pinball q0.50, %)",
        f"{abs(delta_pct):.1f}%",
        delta=f"{'↓' if delta_pct < 0 else '↑'} {delta_pct:.2f}%",
        delta_color="inverse",
    )
    col2.metric("Coverage Combined (90% PI)", f"{coverage_combined * 100:.1f}%")
    col3.metric("Середня ширина PI Combined", f"{width_combined:.5f}")

    st.divider()

    # ── Зведена таблиця ──────────────────────────────────────────────────────
    st.subheader("Зведена таблиця метрик")

    display = summary[
        ["set", "pinball_q0.50", "pi_90_coverage", "pi_90_avg_width"]
    ].copy()
    display.columns = [
        "Конфігурація",
        "Pinball q0.50",
        "Coverage 90%",
        "Ширина PI",
    ]
    display["Δ vs Market (Pinball %)"] = display["Pinball q0.50"].apply(
        lambda v: f"{(v - pinball_market) / pinball_market * 100:+.1f}%"
    )
    display["Coverage 90%"] = display["Coverage 90%"].apply(lambda v: f"{v*100:.1f}%")

    def highlight_best(s: pd.Series) -> list[str]:
        is_best = s == s.min() if s.name == "Pinball q0.50" else s == s.max()
        return [
            "background-color: #d4edda; color: #155724; font-weight: bold"
            if b
            else ""
            for b in is_best
        ]

    styled = (
        display.style.apply(highlight_best, subset=["Pinball q0.50"])
        .apply(highlight_best, subset=["Ширина PI"])
        .format({"Pinball q0.50": "{:.6f}", "Ширина PI": "{:.6f}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # ── Bar chart — Pinball q0.50 ─────────────────────────────────────────────
    st.subheader("Порівняння Pinball q0.50 по конфігураціях")

    bar_df = summary.sort_values("pinball_q0.50")
    fig_bar = go.Figure(
        go.Bar(
            x=bar_df["pinball_q0.50"],
            y=bar_df["set"],
            orientation="h",
            marker_color=[COLORS.get(s, "#888") for s in bar_df["set"]],
            text=[f"{v:.5f}" for v in bar_df["pinball_q0.50"]],
            textposition="outside",
        )
    )
    fig_bar.update_layout(
        xaxis_title="Pinball Loss (q0.50)",
        yaxis_title="",
        height=300,
        margin=dict(l=20, r=60, t=20, b=40),
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Результати по фолдах
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header(f"Результати по фолдах — горизонт {horizon}")

    if not selected_cfgs:
        st.info("Оберіть хоча б одну конфігурацію у боковій панелі.")
        st.stop()

    folds_data: dict[str, pd.DataFrame] = {}
    for cfg in selected_cfgs:
        df_fold = load_fold_results(horizon, cfg)
        if df_fold is None:
            st.warning(
                f"Файл `reports/eval_{horizon}_{cfg}.csv` не знайдено. "
                "Запустіть `07_compare_feature_sets.py`."
            )
        else:
            folds_data[cfg] = df_fold

    if not folds_data:
        st.stop()

    def make_fold_chart(
        metric: str,
        title: str,
        yaxis_title: str,
        hline: float | None = None,
        hline_label: str = "",
    ) -> go.Figure:
        fig = go.Figure()
        for cfg, df in folds_data.items():
            folds = df["split"].astype(int)
            vals = df[metric]
            mean_val = vals.mean()
            fig.add_trace(
                go.Scatter(
                    x=folds,
                    y=vals,
                    mode="lines+markers",
                    name=cfg.capitalize(),
                    line=dict(color=COLORS.get(cfg, "#888"), width=2),
                    marker=dict(size=7),
                )
            )
            fig.add_hline(
                y=mean_val,
                line_dash="dot",
                line_color=COLORS.get(cfg, "#888"),
                opacity=0.5,
                annotation_text=f"Середнє {cfg}: {mean_val:.4f}",
                annotation_position="right",
            )
        if hline is not None:
            fig.add_hline(
                y=hline,
                line_dash="dash",
                line_color="red",
                annotation_text=hline_label,
                annotation_position="right",
            )
        fig.update_layout(
            title=title,
            xaxis_title="Фолд",
            yaxis_title=yaxis_title,
            height=350,
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#eee", dtick=1),
            yaxis=dict(showgrid=True, gridcolor="#eee"),
            legend=dict(orientation="h", y=-0.2),
        )
        return fig

    st.subheader("Pinball Loss q0.50 по фолдах")
    st.plotly_chart(
        make_fold_chart("pinball_q0.50", "", "Pinball q0.50"),
        use_container_width=True,
    )
    st.divider()

    st.subheader("Coverage 90% PI по фолдах")
    st.plotly_chart(
        make_fold_chart(
            "pi_90_coverage",
            "",
            "Coverage",
            hline=0.90,
            hline_label="Номінальний рівень 90%",
        ),
        use_container_width=True,
    )
    st.divider()

    st.subheader("Середня ширина PI по фолдах")
    st.plotly_chart(
        make_fold_chart("pi_90_avg_width", "", "Ширина PI"),
        use_container_width=True,
    )
    st.divider()

    # ── Аналіз проблемних фолдів ──────────────────────────────────────────────
    st.subheader("Ключове спостереження")
    worst_folds = []
    for cfg, df in folds_data.items():
        worst_idx = df["pinball_q0.50"].idxmax()
        worst_folds.append(
            (cfg, int(df.loc[worst_idx, "split"]), df.loc[worst_idx, "pinball_q0.50"])
        )
    worst_folds.sort(key=lambda x: x[2], reverse=True)
    worst_cfg, worst_fold, worst_val = worst_folds[0]

    st.info(
        f"**Найпроблематичніший фолд:** Фолд {worst_fold} у конфігурації "
        f"**{worst_cfg.capitalize()}** (Pinball q0.50 = {worst_val:.5f}). "
        "Це може свідчити про підвищену волатильність або зміну режиму ринку "
        "у відповідному часовому вікні, що ускладнює апроксимацію квантилів."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Важливість ознак
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Важливість ознак — lgbm_combined_4h")

    bundle = load_model("combined", "4h")
    if bundle is None:
        st.warning(
            "Файл `models/lgbm_combined_4h.joblib` не знайдено. "
            "Запустіть `07_compare_feature_sets.py`."
        )
        st.stop()

    feature_cols = bundle["feature_cols"]
    # Усереднюємо важливість по всіх квантильних моделях
    importances = np.mean(
        [bundle["models"][q].feature_importances_ for q in bundle["models"]], axis=0
    )
    total_imp = importances.sum()

    fi_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df["type"] = fi_df["feature"].apply(lambda c: "FGI" if is_fgi(c) else "Ринкова")
    fi_df["color"] = fi_df["type"].map({"FGI": "#ff7f0e", "Ринкова": "#1f77b4"})
    fi_df["pct"] = fi_df["importance"] / total_imp * 100

    top20 = fi_df.head(20).sort_values("importance", ascending=True)

    # ── Горизонтальний bar chart топ-20 ──────────────────────────────────────
    st.subheader("Топ-20 ознак за важливістю")
    fig_fi = go.Figure(
        go.Bar(
            x=top20["importance"],
            y=top20["feature"],
            orientation="h",
            marker_color=top20["color"],
            text=top20["importance"].apply(lambda v: f"{v:.0f}"),
            textposition="outside",
            customdata=np.stack([top20["type"], top20["pct"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Importance: %{x:.0f}<br>"
                "Тип: %{customdata[0]}<br>"
                "Частка: %{customdata[1]:.1f}%<extra></extra>"
            ),
        )
    )
    # Легенда вручну
    for label, color in [("FGI", "#ff7f0e"), ("Ринкова", "#1f77b4")]:
        fig_fi.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                orientation="h",
                marker_color=color,
                name=label,
                showlegend=True,
            )
        )
    fig_fi.update_layout(
        height=600,
        margin=dict(l=20, r=80, t=20, b=40),
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eee", title="Importance (сума по деревах)"),
        yaxis=dict(title=""),
        legend=dict(title="Тип ознаки", orientation="v"),
        barmode="overlay",
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # ── Частка FGI ────────────────────────────────────────────────────────────
    st.subheader("Частка FGI у загальній важливості")
    fgi_imp = fi_df[fi_df["type"] == "FGI"]["importance"].sum()
    fgi_pct = fgi_imp / total_imp * 100

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Частка FGI ознак", f"{fgi_pct:.1f}%")
        st.metric("Частка Ринкових ознак", f"{100 - fgi_pct:.1f}%")

    with col_b:
        pie_labels = ["FGI", "Ринкова"]
        pie_values = [fgi_imp, total_imp - fgi_imp]
        fig_pie = go.Figure(
            go.Pie(
                labels=pie_labels,
                values=pie_values,
                marker_colors=["#ff7f0e", "#1f77b4"],
                hole=0.4,
                textinfo="label+percent",
            )
        )
        fig_pie.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Таблиця топ-10 ────────────────────────────────────────────────────────
    st.subheader("Топ-10 ознак")
    top10 = fi_df.head(10)[["feature", "importance", "pct", "type"]].copy()
    top10.insert(0, "Ранг", range(1, 11))
    top10.columns = ["Ранг", "Ознака", "Importance", "Частка (%)", "Тип"]
    top10["Частка (%)"] = top10["Частка (%)"].round(2)
    top10["Importance"] = top10["Importance"].round(0).astype(int)

    def color_type(val: str) -> str:
        if val == "FGI":
            return "color: #ff7f0e; font-weight: bold"
        return "color: #1f77b4"

    st.dataframe(
        top10.style.applymap(color_type, subset=["Тип"]),
        use_container_width=True,
        hide_index=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Прогноз vs Реальність
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header(f"Прогноз vs Реальність — {symbol}, горизонт {horizon4}, {cfg4}")

    dataset = load_dataset()
    if dataset is None:
        st.warning(
            "Файл `data/dataset_1h.csv` не знайдено. "
            "Запустіть `02_make_dataset.py`."
        )
        st.stop()

    model_bundle = load_model(cfg4, horizon4)
    if model_bundle is None:
        st.warning(
            f"Файл `models/lgbm_{cfg4}_{horizon4}.joblib` не знайдено. "
            "Запустіть `07_compare_feature_sets.py`."
        )
        st.stop()

    # ── Підготовка даних ──────────────────────────────────────────────────────
    target_col = f"y_r_{horizon4}"
    feat_cols = model_bundle["feature_cols"]

    sym_df = dataset[dataset["symbol"] == symbol].copy()
    sym_df = sym_df.dropna(subset=feat_cols + [target_col])
    sym_df = sym_df.tail(n_pts).reset_index(drop=True)

    if len(sym_df) == 0:
        st.warning(f"Немає даних для символу {symbol}.")
        st.stop()

    X = sym_df[feat_cols].values
    y_true = sym_df[target_col].values
    timestamps = sym_df["timestamp"]

    models = model_bundle["models"]
    q_keys = {0.05: 0.05, 0.50: 0.5, 0.95: 0.95}
    preds = {}
    for q_label, q_key in q_keys.items():
        if q_key in models:
            preds[q_label] = models[q_key].predict(X)
        else:
            st.warning(f"Квантиль {q_key} відсутній у моделі.")
            st.stop()

    q05, q50, q95 = preds[0.05], preds[0.50], preds[0.95]

    # ── Метрики ───────────────────────────────────────────────────────────────
    def pinball_loss(y: np.ndarray, q_hat: np.ndarray, alpha: float) -> float:
        err = y - q_hat
        return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))

    pb50 = pinball_loss(y_true, q50, 0.50)
    coverage = float(np.mean((y_true >= q05) & (y_true <= q95)))
    avg_width = float(np.mean(q95 - q05))

    # ── Line chart ────────────────────────────────────────────────────────────
    st.subheader("Прогнозний інтервал та медіанний прогноз")

    fig_pred = go.Figure()

    # Заливка PI
    fig_pred.add_trace(
        go.Scatter(
            x=list(timestamps) + list(timestamps[::-1]),
            y=list(q95) + list(q05[::-1]),
            fill="toself",
            fillcolor="rgba(173, 216, 230, 0.4)",
            line=dict(color="rgba(173,216,230,0)"),
            name="PI 90% [q0.05–q0.95]",
            hoverinfo="skip",
        )
    )
    # Межі PI (тонкі)
    fig_pred.add_trace(
        go.Scatter(
            x=timestamps,
            y=q05,
            mode="lines",
            line=dict(color="#add8e6", width=1, dash="dot"),
            name="q0.05",
            showlegend=False,
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=timestamps,
            y=q95,
            mode="lines",
            line=dict(color="#add8e6", width=1, dash="dot"),
            name="q0.95",
            showlegend=False,
        )
    )
    # Медіанний прогноз
    fig_pred.add_trace(
        go.Scatter(
            x=timestamps,
            y=q50,
            mode="lines",
            line=dict(color="#ff7f0e", width=1.5),
            name="Медіана q0.50",
        )
    )
    # Реальна дохідність
    fig_pred.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_true,
            mode="lines",
            line=dict(color="#1f77b4", width=1.5),
            name="Реальна дохідність",
        )
    )

    fig_pred.update_layout(
        xaxis_title="Час (UTC)",
        yaxis_title=f"Log-дохідність ({horizon4})",
        height=450,
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=True, gridcolor="#eee"),
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=20, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.divider()

    # ── Метрики під графіком ──────────────────────────────────────────────────
    st.subheader("Метрики на вибраному вікні")
    m1, m2, m3 = st.columns(3)
    m1.metric("Pinball Loss q0.50", f"{pb50:.6f}")
    m2.metric(
        "Empirical Coverage (90% PI)",
        f"{coverage * 100:.1f}%",
        delta=f"{(coverage - 0.90) * 100:+.1f}pp vs 90%",
        delta_color="normal",
    )
    m3.metric("Середня ширина PI", f"{avg_width:.5f}")
