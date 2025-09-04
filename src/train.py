from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.config import Config
from src.data_prep import read_data, clean_dataframe, split_xy, build_preprocess
from src.evaluate import compute_metrics, save_metrics, plot_roc, plot_confusion, plot_feature_importance

MODELS = {
    "logreg": LogisticRegression(max_iter=1000, n_jobs=None),
    "rf": RandomForestClassifier(n_estimators=400, max_depth=None, random_state=Config().seed, n_jobs=-1),
    "gb": GradientBoostingClassifier(random_state=Config().seed)
}

def cv_score(model, X, y, preprocess):
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config().seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=str, default="data/raw/churn.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fig-dir", type=str, default="reports/figures")
    parser.add_argument("--metrics-out", type=str, default="reports/metrics.json")
    parser.add_argument("--model-out", type=str, default="models/best_model.joblib")
    args = parser.parse_args()

    # 1) Load & clean
    df = read_data(args.raw_data)
    df = clean_dataframe(df)

    # 2) Train/test split (stratified)
    X, y = split_xy(df, Config().target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=Config().seed, stratify=y
    )

    # Save processed splits for reproducibility
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv("data/processed/train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv("data/processed/test.csv", index=False)

    # 3) Build preprocessing
    preprocess, num_cols, cat_cols = build_preprocess(X_train)

    # 4) CV to select best model by ROC-AUC
    best_name, best_cv = None, -np.inf
    for name, model in MODELS.items():
        score = cv_score(model, X_train, y_train, preprocess)
        print(f"CV ROC-AUC ({name}) = {score:.4f}")
        if score > best_cv:
            best_cv, best_name = score, name

    best_model = MODELS[best_name]
    print(f"Selected model: {best_name} (CV ROC-AUC={best_cv:.4f})")

    # 5) Fit pipeline on training data
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", best_model)])
    pipeline.fit(X_train, y_train)

    # 6) Evaluate on holdout
    y_prob = pipeline.predict_proba(X_test)[:,1]
    metrics = compute_metrics(y_test, y_prob, threshold=args.threshold)
    print("Test metrics:", json.dumps(metrics, indent=2))

    # 7) Save metrics and plots
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, args.metrics_out)
    plot_roc(y_test, y_prob, f"{args.fig_dir}/roc_curve.png")
    plot_confusion(y_test, y_prob, f"{args.fig_dir}/confusion_matrix.png", threshold=args.threshold)
    plot_feature_importance(pipeline, f"{args.fig_dir}/feature_importance.png", top_k=15)

    # 8) Persist end-to-end pipeline
    Path(Path(args.model_out).parent).mkdir(parents=True, exist_ok=True)
    dump(pipeline, args.model_out)
    print(f"Saved model to {args.model_out}")

if __name__ == "__main__":
    main()
