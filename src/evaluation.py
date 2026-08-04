"""Shared evaluation utilities: metrics, confusion matrix, ROC / PR curves.

Both the classical baseline and the transformer route their predictions
through here so every model is judged on identical, reproducible metrics --
exactly what the EECE5644 rubric expects (per-class precision/recall/F1,
macro averages, confusion matrix, and ROC/PR analysis).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from . import config as C


def compute_metrics(y_true, y_pred, labels=None) -> dict:
    labels = labels or list(range(len(C.LABELS)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=C.LABELS,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": report,
    }


def save_confusion_matrix(y_true, y_pred, model_name: str) -> Path:
    labels = list(range(len(C.LABELS)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=C.LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Confusion matrix -- {model_name}")
    fig.tight_layout()
    out = C.FIG_DIR / f"confusion_{model_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_roc_pr_curves(y_true, y_score, model_name: str) -> dict[str, Path]:
    """One-vs-rest ROC and PR curves (macro).  y_score: (n_samples, n_classes)."""
    n_classes = len(C.LABELS)
    y_true = np.asarray(y_true)
    present = sorted(set(y_true.tolist()))
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    y_score = np.asarray(y_score)

    # ---- ROC ----
    fig, ax = plt.subplots(figsize=(6, 5))
    aucs = {}
    for c in present:
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_score[:, c])
        auc = roc_auc_score(y_bin[:, c], y_score[:, c])
        aucs[C.LABELS[c]] = auc
        ax.plot(fpr, tpr, label=f"{C.LABELS[c]} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"ROC (one-vs-rest) -- {model_name}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    roc_path = C.FIG_DIR / f"roc_{model_name}.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    # ---- PR ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in present:
        if y_bin[:, c].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, c], y_score[:, c])
        ax.plot(rec, prec, label=C.LABELS[c])
    ax.set(xlabel="Recall", ylabel="Precision",
           title=f"Precision-Recall -- {model_name}")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    pr_path = C.FIG_DIR / f"pr_{model_name}.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)

    return {"roc": roc_path, "pr": pr_path, "auc_per_class": aucs}


def save_metrics(metrics: dict, model_name: str) -> Path:
    out = C.METRIC_DIR / f"metrics_{model_name}.json"
    out.write_text(json.dumps(metrics, indent=2))
    return out


def print_summary(metrics: dict, model_name: str) -> None:
    print(f"\n=== {model_name} ===")
    print(f"accuracy    : {metrics['accuracy']:.4f}")
    print(f"macro F1    : {metrics['macro_f1']:.4f}")
    print(f"weighted F1 : {metrics['weighted_f1']:.4f}")
    print("per-class F1:")
    for lab in C.LABELS:
        row = metrics["per_class"].get(lab)
        if row:
            print(f"   {lab:12s} P={row['precision']:.2f} "
                  f"R={row['recall']:.2f} F1={row['f1-score']:.2f} "
                  f"(n={int(row['support'])})")


def save_comparison(all_metrics: dict[str, dict]) -> Path:
    """Bar chart + json comparing models on accuracy / macro-F1."""
    names = list(all_metrics)
    acc = [all_metrics[n]["accuracy"] for n in names]
    mf1 = [all_metrics[n]["macro_f1"] for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 3, 5))
    ax.bar(x - w / 2, acc, w, label="Accuracy")
    ax.bar(x + w / 2, mf1, w, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison")
    ax.legend()
    for i, (a, m) in enumerate(zip(acc, mf1)):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, m + 0.01, f"{m:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig_path = C.FIG_DIR / "model_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    summary = {n: {"accuracy": all_metrics[n]["accuracy"],
                   "macro_f1": all_metrics[n]["macro_f1"],
                   "weighted_f1": all_metrics[n]["weighted_f1"]} for n in names}
    (C.METRIC_DIR / "comparison.json").write_text(json.dumps(summary, indent=2))
    return fig_path
