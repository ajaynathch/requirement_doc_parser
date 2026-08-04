
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import build_dataset as B
from . import config as C
from . import evaluation as E


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_df=0.9,
        strip_accents="unicode",
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "nb": Pipeline([("tfidf", _tfidf()), ("clf", MultinomialNB())]),
        "logreg": Pipeline([
            ("tfidf", _tfidf()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        # LinearSVC has no predict_proba; we softmax its decision_function for
        # ROC/PR (see _scores).  This avoids CalibratedClassifierCV, whose inner
        # CV breaks on classes with only 1-2 samples.
        "linsvm": Pipeline([
            ("tfidf", _tfidf()),
            ("clf", LinearSVC(class_weight="balanced")),
        ]),
    }


def _scores(model, X) -> np.ndarray:
    """Class-probability-like scores for ROC/PR, whatever the estimator."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    margins = model.decision_function(X)
    if margins.ndim == 1:  # binary edge case
        margins = np.column_stack([-margins, margins])
    return softmax(margins, axis=1)


def _load(split: str) -> pd.DataFrame:
    path = C.DATASET_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `python -m src.build_dataset`.")
    return pd.read_csv(path)


def run(mask_modals: bool = False) -> dict:
    train, val, test = _load("train"), _load("val"), _load("test")

    def prep(df):
        text = df["text"].map(B.mask_modal_cues) if mask_modals else df["text"]
        return text.tolist(), df["label_id"].to_numpy()

    Xtr, ytr = prep(train)
    Xval, yval = prep(val)
    Xte, yte = prep(test)

    # ---- cross-validated model selection on the training split ----
    n_min = np.bincount(ytr).min() if len(ytr) else 0
    folds = max(2, min(C.CV_FOLDS, int(n_min))) if n_min >= 2 else 2
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.SEED)

    cv_results = {}
    for name, model in candidate_models().items():
        try:
            scores = cross_val_score(model, Xtr, ytr, cv=skf,
                                     scoring="f1_macro", n_jobs=-1)
            cv_results[name] = float(scores.mean())
            print(f"[baseline] CV macro-F1  {name:8s} = {scores.mean():.3f} "
                  f"(+/- {scores.std():.3f})")
        except Exception as exc:  # tiny/degenerate folds
            cv_results[name] = float("nan")
            print(f"[baseline] CV failed for {name}: {exc}")

    best_name = max(cv_results, key=lambda k: (cv_results[k] if cv_results[k] == cv_results[k] else -1))
    print(f"[baseline] selected model: {best_name}")

    # ---- refit best on train+val, evaluate on test ----
    best = candidate_models()[best_name]
    best.fit(Xtr + Xval, np.concatenate([ytr, yval]))

    y_pred = best.predict(Xte)
    y_score = _scores(best, Xte)

    tag = f"baseline_{best_name}" + ("_masked" if mask_modals else "")
    metrics = E.compute_metrics(yte, y_pred)
    metrics["cv_macro_f1"] = cv_results
    metrics["selected_model"] = best_name
    metrics["mask_modals"] = mask_modals

    E.save_metrics(metrics, tag)
    E.save_confusion_matrix(yte, y_pred, tag)
    E.save_roc_pr_curves(yte, y_score, tag)
    E.print_summary(metrics, tag)
    return {"tag": tag, "metrics": metrics}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-modals", action="store_true",
                    help="blank modal trigger words to test context learning")
    args = ap.parse_args()
    run(mask_modals=args.mask_modals)
