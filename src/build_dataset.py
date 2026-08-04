
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def label_sentence(text: str) -> str:
    low = f" {text.lower()} "
    for label, phrases in C.MODALITY_RULES:
        for phrase in phrases:
            # word-boundary match so 'may' does not fire inside 'maybe'
            if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", low):
                return label
    return "INFORMATIVE"


def mask_modal_cues(text: str) -> str:
    out = text
    for _label, phrases in C.MODALITY_RULES:
        for phrase in sorted(phrases, key=len, reverse=True):
            out = re.sub(rf"(?<![a-zA-Z]){re.escape(phrase)}(?![a-zA-Z])",
                         "[MODAL]", out, flags=re.IGNORECASE)
    return out


def build_dataframe(records_json: Path = C.PARSED_DIR / "all_records.json") -> pd.DataFrame:
    if not records_json.exists():
        raise FileNotFoundError(
            f"{records_json} not found. Run `python -m src.parse` first."
        )
    records = json.loads(records_json.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["label"] = df["text"].map(label_sentence)
    df["label_id"] = df["label"].map(C.LABEL2ID)
    return df


def _safe_split(df: pd.DataFrame, test_size: float):
    
    n_classes = df["label"].nunique()
    n_test = int(round(len(df) * test_size))
    can_stratify = (df["label"].value_counts().min() >= 2
                    and n_test >= n_classes
                    and (len(df) - n_test) >= n_classes)
    strat = df["label"] if can_stratify else None
    if strat is None:
        print(f"[dataset] (note) too few samples to stratify a {test_size:.0%} "
              f"split cleanly; using a random split for this stage.")
    return train_test_split(df, test_size=test_size, random_state=C.SEED,
                            stratify=strat)


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_val, test = _safe_split(df, C.TEST_SIZE)
    val_frac = C.VAL_SIZE / (1.0 - C.TEST_SIZE)
    train, val = _safe_split(train_val, val_frac)
    return {"train": train, "val": val, "test": test}


def main() -> None:
    df = build_dataframe()
    df.to_csv(C.DATASET_DIR / "labeled.csv", index=False)

    stats = df["label"].value_counts().to_dict()
    (C.DATASET_DIR / "label_stats.json").write_text(json.dumps(stats, indent=2))

    splits = make_splits(df)
    for name, part in splits.items():
        part.to_csv(C.DATASET_DIR / f"{name}.csv", index=False)

    print(f"[dataset] {len(df)} labeled sentences")
    print("[dataset] class distribution:")
    for lab in C.LABELS:
        print(f"          {lab:12s} {stats.get(lab, 0)}")
    print("[dataset] splits:",
          {k: len(v) for k, v in splits.items()})
    print(f"[dataset] written to {C.DATASET_DIR}")


if __name__ == "__main__":
    main()
