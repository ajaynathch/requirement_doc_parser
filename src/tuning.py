"""Hyperparameter tuning for the classical baseline (GridSearchCV).

Compares the model BEFORE tuning (default hyperparameters) against AFTER tuning
(best hyperparameters found by an exhaustive grid search with stratified k-fold
cross-validation), on the same held-out test set.

Outputs:
    outputs/metrics/tuning.json          before/after metrics + best params
    outputs/figures/tuning_before_after.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config as C
from . import evaluation as E


def _load(split):
    return pd.read_csv(C.DATASET_DIR / f"{split}.csv")


def _pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode")),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


# Grid over both the feature representation and the classifier regularization.
PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "tfidf__sublinear_tf": [True, False],
    "clf__C": [0.1, 1.0, 10.0],
}

# The "before" configuration = library defaults for the same pipeline.
DEFAULT_PARAMS = {
    "tfidf__ngram_range": (1, 1),
    "tfidf__min_df": 1,
    "tfidf__sublinear_tf": False,
    "clf__C": 1.0,
}


def run() -> dict:
    train, val, test = _load("train"), _load("val"), _load("test")
    Xtr = train["text"].tolist() + val["text"].tolist()
    ytr = np.concatenate([train["label_id"].to_numpy(), val["label_id"].to_numpy()])
    Xte, yte = test["text"].tolist(), test["label_id"].to_numpy()

    n_min = int(np.bincount(ytr).min())
    folds = max(2, min(C.CV_FOLDS, n_min))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.SEED)

    # ---- BEFORE tuning: default hyperparameters ----
    before = _pipeline().set_params(**DEFAULT_PARAMS)
    before.fit(Xtr, ytr)
    m_before = E.compute_metrics(yte, before.predict(Xte))

    # ---- GridSearch ----
    grid = GridSearchCV(_pipeline(), PARAM_GRID, scoring="f1_macro",
                        cv=skf, n_jobs=-1, refit=True)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    m_after = E.compute_metrics(yte, best.predict(Xte))

    result = {
        "cv_folds": folds,
        "grid_size": int(len(grid.cv_results_["params"])),
        "before_params": {k: str(v) for k, v in DEFAULT_PARAMS.items()},
        "before_test": {"accuracy": m_before["accuracy"],
                        "macro_f1": m_before["macro_f1"],
                        "weighted_f1": m_before["weighted_f1"]},
        "best_params": {k: str(v) for k, v in grid.best_params_.items()},
        "best_cv_macro_f1": float(grid.best_score_),
        "after_test": {"accuracy": m_after["accuracy"],
                       "macro_f1": m_after["macro_f1"],
                       "weighted_f1": m_after["weighted_f1"]},
    }
    (C.METRIC_DIR / "tuning.json").write_text(json.dumps(result, indent=2))

    # ---- before/after bar chart ----
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    before_v = [result["before_test"][m] for m in metrics]
    after_v = [result["after_test"][m] for m in metrics]
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, before_v, w, label="Before tuning", color="#9db4c0")
    ax.bar(x + w/2, after_v, w, label="After tuning", color="#1a3c6e")
    ax.set_xticks(x); ax.set_xticklabels(["Accuracy", "Macro-F1", "Weighted-F1"])
    ax.set_ylim(0, 1); ax.set_title("Logistic Regression: before vs. after tuning")
    ax.legend()
    for i, (b, a) in enumerate(zip(before_v, after_v)):
        ax.text(i - w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "tuning_before_after.png", dpi=150)
    plt.close(fig)

    print(f"[tuning] grid size: {result['grid_size']} combos, {folds}-fold CV")
    print(f"[tuning] best params: {grid.best_params_}")
    print(f"[tuning] before test macro-F1: {m_before['macro_f1']:.3f}")
    print(f"[tuning] after  test macro-F1: {m_after['macro_f1']:.3f}")
    return result


if __name__ == "__main__":
    run()
