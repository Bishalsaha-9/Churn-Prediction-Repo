
import pandas as pd
from src.data_prep import clean_dataframe, split_xy, build_preprocess

def test_preprocess_smoke():
    df = pd.DataFrame({
        "customerID": ["C1","C2"],
        "SeniorCitizen":[0,1],
        "tenure":[5,40],
        "MonthlyCharges":[55.3, 80.1],
        "TotalCharges":[300.0, 1800.5],
        "gender":["Male","Female"],
        "Churn":[0,1]
    })
    df = clean_dataframe(df)
    X, y = split_xy(df, "Churn")
    pre, num, cat = build_preprocess(X)
    assert set(num).issubset(set(X.columns))
    assert len(cat) == len(X.columns) - len(num)
