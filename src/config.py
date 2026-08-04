"""Central configuration: paths, label schema, and modality lexicon.

Keeping every tunable in one place makes the pipeline reproducible and keeps
the label taxonomy identical across the classical and transformer models.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # drop the NVMe spec PDF/DOCX here
PARSED_DIR = DATA_DIR / "parsed"      # Docling markdown + json land here
DATASET_DIR = DATA_DIR / "dataset"    # generated train/val/test CSVs

OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
METRIC_DIR = OUT_DIR / "metrics"
MODEL_DIR = OUT_DIR / "models"

for _d in (RAW_DIR, PARSED_DIR, DATASET_DIR, FIG_DIR, METRIC_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Label schema
# --------------------------------------------------------------------------- #
# We classify each extracted sentence by its *normative modality* -- the
# strongest signal a specification gives us without manual annotation.
# This is the classic "requirement identification / obligation typing" task
# in requirements-engineering NLP.
#
#   MANDATORY   : "shall", "must", "is required to"          (a hard requirement)
#   PROHIBITED  : "shall not", "must not", "may not"         (a hard prohibition)
#   RECOMMENDED : "should", "recommended"                    (soft guidance)
#   OPTIONAL    : "may", "can", "optional"                   (permitted behaviour)
#   INFORMATIVE : none of the above                          (descriptive text)
LABELS = ["MANDATORY", "PROHIBITED", "RECOMMENDED", "OPTIONAL", "INFORMATIVE"]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}

# Ordered longest-first so "shall not" is matched before "shall".
MODALITY_RULES: list[tuple[str, list[str]]] = [
    ("PROHIBITED", ["shall not", "must not", "may not", "should not", "cannot", "is prohibited"]),
    ("MANDATORY", ["shall", "must", "is required to", "are required to", "required to", "is mandatory"]),
    ("RECOMMENDED", ["should", "recommended", "ought to"]),
    ("OPTIONAL", ["may", "optional", "can optionally", "is permitted", "is allowed"]),
]

# --------------------------------------------------------------------------- #
# Model / training
# --------------------------------------------------------------------------- #
TRANSFORMER_NAME = "distilbert-base-uncased"
MAX_LEN = 128
SEED = 5644
TEST_SIZE = 0.20
VAL_SIZE = 0.10          # fraction of the *full* dataset held out for validation
CV_FOLDS = 5             # stratified CV for the classical baseline
