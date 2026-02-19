# Мінімально працюючий пайплайн (BTC/ETH, 1h дані, квантільний прогноз)

Це “скелет” коду, який:
1) завантажує погодинні ціни BTC/ETH з CoinGecko,
2) будує датасет з лагами/ролінг-статистиками,
3) навчає LightGBM-квантільні моделі (0.05/0.50/0.95),
4) рахує pinball loss + coverage і будує простий графік.

## Запуск (локально)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt

python code/01_fetch_prices.py
python code/02_make_dataset.py
python code/03_train_quantile_gbm.py
python code/04_evaluate.py
```

Артефакти:
- `data/prices_1h.csv` — сирі ціни,
- `data/dataset_1h.csv` — фічі + таргет,
- `models/lgbm_quantiles_4h.joblib` — збережені моделі,
- `reports/eval_4h.csv`, `reports/pred_vs_true_4h.png` — звіт/графік.

## Запуск у Google Colab

У проєкті є готовий ноутбук: `code/run_colab.ipynb`.
Логіка проста: завантажуєш zip цього проєкту в Colab, ноутбук розпаковує його в `project/`, ставить залежності та запускає скрипти.

## Що далі додати для диплома
- Подієві ознаки: новини/соцмережі → тональність/ембедінги → агрегація у вікнах 1h/4h.
- Бейзлайни: наївний (0-return), ARIMA/ETS, лінійна квантільна регресія.
- Абляції: лише market-features vs market+event features.
