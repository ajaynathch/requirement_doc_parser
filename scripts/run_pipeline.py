"""End-to-end driver: parse -> dataset -> baseline -> transformer -> compare.

Usage (from project root, with the venv active):
    python -m scripts.run_pipeline                 # full run
    python -m scripts.run_pipeline --skip-transformer
    python -m scripts.run_pipeline --mask-modals   # ablation: hide modal cues
    python -m scripts.run_pipeline --epochs 4
"""
from __future__ import annotations

import argparse

from src import baseline, build_dataset, evaluation, parse, transformer_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-parse", action="store_true",
                    help="reuse existing data/parsed artifacts")
    ap.add_argument("--skip-transformer", action="store_true",
                    help="run only the classical baseline")
    ap.add_argument("--mask-modals", action="store_true",
                    help="ablation: blank modal trigger words")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    # Phase 1 + 2 ---------------------------------------------------------
    if not args.skip_parse:
        parse.parse_all()
    build_dataset.main()

    # Phase 4 -------------------------------------------------------------
    all_metrics: dict[str, dict] = {}

    base = baseline.run(mask_modals=args.mask_modals)
    all_metrics[base["tag"]] = base["metrics"]

    if not args.skip_transformer:
        tr = transformer_model.run(
            epochs=args.epochs, batch_size=args.batch_size,
            mask_modals=args.mask_modals,
        )
        all_metrics[tr["tag"]] = tr["metrics"]

    # Comparison ----------------------------------------------------------
    fig = evaluation.save_comparison(all_metrics)
    print(f"\n[pipeline] comparison chart -> {fig}")
    print("[pipeline] done. See outputs/figures and outputs/metrics.")


if __name__ == "__main__":
    main()
