
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config as C

EDA_DIR = C.FIG_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].astype(str)
    feats = pd.DataFrame({
        "char_len": text.str.len(),
        "word_count": text.str.split().map(len),
        "avg_word_len": text.str.replace(r"\s+", "", regex=True).str.len()
                        / text.str.split().map(len).clip(lower=1),
        "num_digits": text.str.count(r"\d"),
        "num_acronyms": text.str.count(r"\b[A-Z]{2,}\b"),
        "has_negation": text.str.contains(r"\bnot\b", case=False).astype(int),
        "has_ref": text.str.contains(r"\b(?:refer|section|figure)\b", case=False).astype(int),
        "has_shall_must": text.str.contains(r"\b(?:shall|must|required)\b", case=False).astype(int),
        "has_should": text.str.contains(r"\bshould\b", case=False).astype(int),
        "has_may": text.str.contains(r"\b(?:may|optional|can)\b", case=False).astype(int),
    })
    feats["label"] = df["label"].values
    feats["label_id"] = df["label_id"].values
    return feats


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().reindex(C.LABELS).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="viridis",
                hue=counts.index, legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, int(v), ha="center", fontweight="bold")
    ax.set(title="Class distribution (modality labels)",
           xlabel="", ylabel="number of sentences")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(feats: pd.DataFrame) -> None:
    num = feats.drop(columns=["label"])
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Correlation heatmap of engineered features (+ label_id)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "feature_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_feature_by_class(feats: pd.DataFrame) -> None:
    num_cols = [c for c in feats.columns if c not in ("label", "label_id")]
    means = feats.groupby("label")[num_cols].mean().reindex(C.LABELS)
    # z-score each feature across classes so scales are comparable
    z = (means - means.mean()) / means.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(z.T, annot=means.T, fmt=".1f", cmap="mako",
                cbar_kws={"label": "z-score across classes"}, ax=ax)
    ax.set(title="Mean engineered-feature value by class (annotated with raw mean)",
           xlabel="", ylabel="")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "feature_by_class_heatmap.png", dpi=150)
    plt.close(fig)


def plot_length_by_class(feats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=feats, x="label", y="word_count", order=C.LABELS,
                palette="viridis", hue="label", legend=False, ax=ax)
    ax.set(title="Sentence length (words) by class", xlabel="", ylabel="word count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "length_by_class.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(C.DATASET_DIR / "labeled.csv")
    feats = engineer_features(df)

    plot_class_distribution(df)
    plot_correlation_heatmap(feats)
    plot_feature_by_class(feats)
    plot_length_by_class(feats)

    # quick textual summary used in the report
    print(f"[eda] samples: {len(df)}  classes: {df['label'].nunique()}")
    print(f"[eda] duplicates in raw parse handled during dataset build")
    print("[eda] class counts:\n", df["label"].value_counts().to_string())
    corr = feats.drop(columns=["label"]).corr(numeric_only=True)["label_id"].drop("label_id")
    print("[eda] |correlation| of features with label_id (desc):")
    print(corr.abs().sort_values(ascending=False).round(3).to_string())
    print(f"[eda] figures written to {EDA_DIR}")


if __name__ == "__main__":
    main()
