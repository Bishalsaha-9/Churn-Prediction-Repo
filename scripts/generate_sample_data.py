import argparse
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def make_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n_rows)
    senior = rng.integers(0, 2, size=n_rows)  # 0/1
    partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.45, 0.55])
    dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7])

    tenure = rng.integers(0, 73, size=n_rows)  # months
    phone_service = rng.choice(["Yes", "No"], size=n_rows, p=[0.9, 0.1])
    multiple_lines = np.where(phone_service == "No",
                              "No phone service",
                              rng.choice(["Yes", "No"], size=n_rows, p=[0.4, 0.6]))

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.4, 0.5, 0.1])

    def with_internet(x):
        return np.where(internet_service == "No", "No internet service", x)

    online_security  = with_internet(rng.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7]))
    online_backup    = with_internet(rng.choice(["Yes", "No"], size=n_rows, p=[0.4, 0.6]))
    device_protect   = with_internet(rng.choice(["Yes", "No"], size=n_rows, p=[0.35, 0.65]))
    tech_support     = with_internet(rng.choice(["Yes", "No"], size=n_rows, p=[0.25, 0.75]))
    streaming_tv     = with_internet(rng.choice(["Yes", "No"], size=n_rows, p=[0.5, 0.5]))

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.20])
    paperless = rng.choice(["Yes", "No"], size=n_rows, p=[0.6, 0.4])
    payment_method = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                                size=n_rows, p=[0.45, 0.2, 0.2, 0.15])

    # Base monthly charges by internet service
    base = np.select(
        [internet_service == "No", internet_service == "DSL", internet_service == "Fiber optic"],
        [20, 50, 75],
        default=50
    )

    # Add-ons cost
    addon = (
        (online_security == "Yes").astype(int) * 5 +
        (online_backup == "Yes").astype(int)   * 5 +
        (device_protect == "Yes").astype(int)  * 5 +
        (tech_support == "Yes").astype(int)    * 7 +
        (streaming_tv == "Yes").astype(int)    * 8 +
        (multiple_lines == "Yes").astype(int)  * 5
    )

    monthly_charges = base + addon + RNG.normal(0, 2, size=n_rows)
    monthly_charges = np.clip(monthly_charges, 15, None)

    total_charges = monthly_charges * tenure + RNG.normal(0, 20, size=n_rows)
    total_charges = np.clip(total_charges, 0, None)

    # Churn probability: higher for month-to-month, higher charges, fiber optic, short tenure,
    # electronic check, no tech support, no online security. Seniors churn slightly more.
    logits = (
        -2.0
        + 1.0 * (contract == "Month-to-month").astype(float)
        + 0.6 * (internet_service == "Fiber optic").astype(float)
        + 0.8 * (payment_method == "Electronic check").astype(float)
        - 0.03 * tenure
        + 0.01 * (monthly_charges - monthly_charges.mean())
        + 0.5 * (tech_support == "No").astype(float)
        + 0.4 * (online_security == "No").astype(float)
        + 0.2 * senior
    )
    prob = logistic(logits)
    churn = (RNG.random(n_rows) < prob).astype(int)

    df = pd.DataFrame({
        "customerID": [f"C{100000+i}" for i in range(n_rows)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protect,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": np.round(monthly_charges, 2),
        "TotalCharges": np.round(total_charges, 2),
        "Churn": churn
    })

    # Introduce a few missing values to simulate reality
    mask = RNG.random(n_rows) < 0.01
    df.loc[mask, "TotalCharges"] = np.nan
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=5000)
    parser.add_argument("--out", type=str, default="data/raw/churn.csv")
    parser.add_argument("--scoring_out", type=str, default="data/raw/churn_scoring_sample.csv")
    args = parser.parse_args()

    df = make_dataset(args.n_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # make a small scoring (no target) sample
    scoring = df.drop(columns=["Churn"]).sample(25, random_state=42)
    scoring.to_csv(args.scoring_out, index=False)

    print(f"Wrote {args.out} with shape={df.shape}")
    print(f"Wrote scoring sample {args.scoring_out} with shape={scoring.shape}")

if __name__ == "__main__":
    main()
