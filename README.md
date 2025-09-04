# Predictive Churn Modeling — Sample Project

A beginner-friendly, production-style template showing **end‑to‑end churn prediction**:
data generation (or bring your own CSV), preprocessing, model training, evaluation,
and batch inference — all using a clean, reusable **scikit‑learn Pipeline**.

> You can upload this as-is to GitHub and extend it later with real data.

## 🔧 Tech
- Python, pandas, numpy
- scikit-learn Pipelines (OneHotEncoder + StandardScaler)
- Logistic Regression, Random Forest, Gradient Boosting
- Joblib model persistence
- Matplotlib charts (ROC, Confusion Matrix, Feature Importance)

## 📁 Repo Layout
```
.
├── data/
│   ├── raw/                # input CSV(s). A synthetic example is provided.
│   └── processed/          # train/test folds saved here
├── models/                 # serialized pipelines (.joblib)
├── reports/
│   └── figures/            # ROC, Confusion Matrix, Feature Importance
├── scripts/
│   └── generate_sample_data.py
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── evaluate.py
│   ├── train.py
│   └── infer.py
├── tests/
│   └── test_data_prep.py
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

```bash
# 1) Create and activate venv (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Option A) Use the included synthetic dataset
python scripts/generate_sample_data.py --n_rows 5000

# 4) Train models and pick the best
python src/train.py --raw-data data/raw/churn.csv --fig-dir reports/figures --model-out models/best_model.joblib

# 5) Run batch inference on new customers (sample is generated too)
python src/infer.py --model models/best_model.joblib --input data/raw/churn_scoring_sample.csv --output predictions.csv
```

## 🧰 Swap in your own data

Replace `data/raw/churn.csv` with your dataset. Required:
- A binary target column named **`Churn`** (values 0/1 or Yes/No).
- Other feature columns (numeric or categorical).

If your target column has different values (e.g., "Yes"/"No"), the code will map them to 1/0.

## 🧪 Minimal test

Run a quick smoke test on preprocessing:
```bash
python -m pytest -q
```

## 📈 What gets produced

- `models/best_model.joblib` — a single, end-to-end **Pipeline** (preprocessing + model).
- `reports/figures/roc_curve.png` — ROC curve on hold-out set.
- `reports/figures/confusion_matrix.png` — normalized confusion matrix.
- `reports/figures/feature_importance.png` — top 15 features.
- `data/processed/` — train/test CSV splits to make results reproducible.
- `reports/metrics.json` — key metrics (Accuracy, Precision, Recall, F1, ROC-AUC).

## 🗂️ CLI help
Each script supports `--help` for available options.
