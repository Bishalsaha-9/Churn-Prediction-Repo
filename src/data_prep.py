from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_DEFAULT = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize target
    if "Churn" in df.columns:
        if df["Churn"].dtype == "O":
            df["Churn"] = df["Churn"].str.strip().str.lower().map({"yes":1, "no":0})
    # Ensure numeric
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop duplicates
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset=["customerID"], keep="last")

    return df

def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = None
    if target_col in df.columns:
        y = df[target_col].astype(int)
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])
    return X, y

def build_preprocess(X: pd.DataFrame, numeric_cols: List[str] = None) -> ColumnTransformer:
    if numeric_cols is None:
        numeric_cols = [c for c in X.columns if c in NUMERIC_DEFAULT or pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return preprocess, numeric_cols, cat_cols
