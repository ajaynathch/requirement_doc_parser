"""Feature importance + learning curve for the classical model.

Feature importance:
    A TF-IDF + Logistic Regression pipeline exposes one weight per (class, term).
    The largest positive weights are the terms that most push a sentence toward a
    class; near-zero weights are terms with little influence.  We plot the top
    terms per class and report the least-influential terms.

Learning curve:
    Shows how test-generalization (CV macro-F1) changes as training-set size
    grows -- evidence of whether more data would help.

Outputs:
    outputs/figures/feature_importance.png
    outputs/figures/learning_curve.png
    outputs/metrics/feature_importance.json
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
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline

from . import config as C


def _load(split):
    return pd.read_csv(C.DATASET_DIR / f"{split}.csv")


def _fit_full():
    train, val = _load("train"), _load("val")
    X = train["text"].tolist() + val["text"].tolist()
    y = np.concatenate([train["label_id"].to_numpy(), val["label_id"].to_numpy()])
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                  min_df=1, strip_accents="unicode")),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(X, y)
    return pipe, X, y


def top_features(pipe, top_n=8) -> dict:
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_          # (n_classes, n_features)
    classes = pipe.named_steps["clf"].classes_
    out = {}
    for row, cls in zip(coef, classes):
        order = np.argsort(row)[::-1][:top_n]
        out[C.ID2LABEL[int(cls)]] = [(vocab[i], float(row[i])) for i in order]
    return out


def plot_top_features(top: dict) -> None:
    labels = [l for l in C.LABELS if l in top]
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.5), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, lab in zip(axes, labels):
        terms, weights = zip(*top[lab])
        y = np.arange(len(terms))[::-1]
        ax.barh(y, weights, color="#1a3c6e")
        ax.set_yticks(y); ax.set_yticklabels(terms, fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("LogReg weight", fontsize=8)
    fig.suptitle("Top TF-IDF features per class (Logistic Regression weights)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(C.FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_learning_curve(pipe, X, y) -> dict:
    n_min = int(np.bincount(y).min())
    folds = max(2, min(5, n_min))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.SEED)
    sizes, train_sc, test_sc = learning_curve(
        pipe, X, y, cv=skf, scoring="f1_macro",
        train_sizes=np.linspace(0.3, 1.0, 5), n_jobs=-1,
        random_state=C.SEED, shuffle=True,
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(sizes, train_sc.mean(1), "o-", label="Training", color="#1a3c6e")
    ax.plot(sizes, test_sc.mean(1), "o-", label="Cross-val", color="#e0a800")
    ax.fill_between(sizes, test_sc.mean(1) - test_sc.std(1),
                    test_sc.mean(1) + test_sc.std(1), alpha=0.15, color="#e0a800")
    ax.set(xlabel="Training samples", ylabel="Macro-F1",
           title="Learning curve (TF-IDF + Logistic Regression)")
    ax.legend(); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)
    return {"train_sizes": [int(s) for s in sizes],
            "cv_macro_f1": [float(s) for s in test_sc.mean(1)]}


def main() -> None:
    pipe, X, y = _fit_full()
    top = top_features(pipe)
    plot_top_features(top)
    lc = plot_learning_curve(pipe, X, y)

    (C.METRIC_DIR / "feature_importance.json").write_text(
        json.dumps({"top_features": top, "learning_curve": lc}, indent=2))

    print("[featimp] top features per class:")
    for lab, feats in top.items():
        print(f"   {lab:12s} {', '.join(t for t, _ in feats[:5])}")
    print(f"[featimp] learning curve CV macro-F1: {lc['cv_macro_f1']}")


if __name__ == "__main__":
    main()
