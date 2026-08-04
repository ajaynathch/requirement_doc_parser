"""Phase 4b -- fine-tune DistilBERT for requirement classification.

This is the deep-learning half of the project.  A pretrained DistilBERT encoder
is fine-tuned with a classification head over the modality labels, using the
HuggingFace Trainer.  Runs on Apple-Silicon MPS, CUDA, or CPU automatically.

Compared against the classical TF-IDF baseline in scripts/compare.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from . import build_dataset as B
from . import config as C
from . import evaluation as E


class WeightedTrainer(Trainer):
    """Trainer with a class-weighted cross-entropy loss.

    Requirement corpora are heavily imbalanced (descriptive/INFORMATIVE prose
    dominates, and modalities such as RECOMMENDED are rare).  Un-weighted loss
    lets the model collapse onto the majority class and score ~0 macro-F1 on the
    minority classes.  Weighting the loss inversely to class frequency forces the
    encoder to attend to the rare-but-important requirement types.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = (self._class_weights.to(outputs.logits.device)
                  if self._class_weights is not None else None)
        loss = nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load(split: str, mask_modals: bool) -> pd.DataFrame:
    df = pd.read_csv(C.DATASET_DIR / f"{split}.csv")
    if mask_modals:
        df = df.copy()
        df["text"] = df["text"].map(B.mask_modal_cues)
    return df


def _to_hf(df: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=C.MAX_LEN),
                batched=True)
    keep = {"input_ids", "attention_mask", "labels"}
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


def run(epochs: int = 6, batch_size: int = 16, lr: float = 3e-5,
        mask_modals: bool = False, class_weighting: bool = True) -> dict:
    device = _device()
    print(f"[transformer] device = {device}  class_weighting = {class_weighting}")

    tokenizer = AutoTokenizer.from_pretrained(C.TRANSFORMER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        C.TRANSFORMER_NAME,
        num_labels=len(C.LABELS),
        id2label=C.ID2LABEL,
        label2id=C.LABEL2ID,
    )

    train_df = _load("train", mask_modals)
    train = _to_hf(train_df, tokenizer)
    val = _to_hf(_load("val", mask_modals), tokenizer)

    class_weights = None
    if class_weighting:
        present = train_df["label_id"].to_numpy()
        w = compute_class_weight("balanced", classes=np.unique(present), y=present)
        full = np.ones(len(C.LABELS), dtype="float32")
        for cls, weight in zip(np.unique(present), w):
            full[cls] = weight
        class_weights = torch.tensor(full, dtype=torch.float32)
        print("[transformer] class weights:",
              {C.LABELS[i]: round(float(full[i]), 2) for i in range(len(C.LABELS))})
    test_df = _load("test", mask_modals)
    test = _to_hf(test_df, tokenizer)

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        m = E.compute_metrics(labels, preds)
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    tag = "distilbert" + ("_masked" if mask_modals else "")
    args = TrainingArguments(
        output_dir=str(C.MODEL_DIR / tag),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        seed=C.SEED,
        report_to="none",
        # Trainer auto-selects MPS/CUDA/CPU; keep training on CPU only if neither.
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    # ---- evaluate on held-out test set ----
    pred = trainer.predict(test)
    logits = pred.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = test_df["label_id"].to_numpy()
    # softmax for ROC/PR
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_score = ex / ex.sum(axis=1, keepdims=True)

    metrics = E.compute_metrics(y_true, y_pred)
    metrics["mask_modals"] = mask_modals
    metrics["class_weighting"] = class_weighting
    metrics["device"] = device
    metrics["hyperparams"] = {"epochs": epochs, "batch_size": batch_size, "lr": lr}

    E.save_metrics(metrics, tag)
    E.save_confusion_matrix(y_true, y_pred, tag)
    E.save_roc_pr_curves(y_true, y_score, tag)
    E.print_summary(metrics, tag)

    trainer.save_model(str(C.MODEL_DIR / tag / "best"))
    tokenizer.save_pretrained(str(C.MODEL_DIR / tag / "best"))
    return {"tag": tag, "metrics": metrics}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--mask-modals", action="store_true")
    ap.add_argument("--no-class-weight", action="store_true",
                    help="disable class-weighted loss (ablation)")
    args = ap.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        mask_modals=args.mask_modals, class_weighting=not args.no_class_weight)
