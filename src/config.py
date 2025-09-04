from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    seed: int = 42
    target_col: str = "Churn"
