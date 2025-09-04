from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
)

def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }

def save_metrics(metrics: Dict[str, float], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

def plot_roc(y_true, y_prob, out_png: str):
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_confusion(y_true, y_prob, out_png: str, threshold: float = 0.5):
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No churn","Churn"])
    disp.plot(values_format=".2f", colorbar=False)
    plt.title("Confusion Matrix (normalized)")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def get_feature_names(preprocess) -> Tuple[list, list, list]:
    # Extract transformed feature names from ColumnTransformer
    num_names = preprocess.transformers_[0][2]
    cat_names = preprocess.transformers_[1][2]
    ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
    cat_expanded = list(ohe.get_feature_names_out(cat_names))
    return list(num_names), list(cat_names), cat_expanded

def plot_feature_importance(pipeline, out_png: str, top_k: int = 15):
    import numpy as np
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    num_names, cat_names, cat_expanded = get_feature_names(preprocess)
    feature_names = num_names + cat_expanded

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # LogisticRegression: take absolute value of coefficients
        coef = model.coef_.ravel()
        importances = np.abs(coef)
    else:
        return  # silently skip if not supported

    idx = np.argsort(importances)[::-1][:top_k]
    names_top = [feature_names[i] for i in idx]
    imps_top = importances[idx]

    plt.figure()
    plt.barh(range(len(names_top)), imps_top)
    plt.yticks(range(len(names_top)), names_top)
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
