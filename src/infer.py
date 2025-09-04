from __future__ import annotations
import argparse
import pandas as pd
from joblib import load
from pathlib import Path
from src.data_prep import clean_dataframe, split_xy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="CSV of customers to score (no Churn column)")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    pipe = load(args.model)
    df = pd.read_csv(args.input)
    df = clean_dataframe(df)
    X, _ = split_xy(df, target_col="Churn")

    proba = pipe.predict_proba(X)[:,1]
    pred = (proba >= args.threshold).astype(int)

    out = df.copy()
    out["Churn_Probability"] = proba
    out["Prediction"] = pred

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote predictions -> {args.output} (n={len(out)})")

if __name__ == "__main__":
    main()
