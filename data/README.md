# Data folder

## What to drop here

Put the NVMe specification pages you want to use into **`data/raw/`**:

```
data/raw/nvme_spec_pages.pdf        <-- your file(s); PDF / DOCX / HTML all work
```

You can add more than one document; every file in `data/raw/` is parsed and
pooled into one dataset.

## What the pipeline generates

```
data/raw/        # INPUT  -- your spec documents (you provide these)
data/parsed/     # Docling output: <doc>.md, <doc>.records.json, all_records.json
data/dataset/    # labeled.csv, train/val/test.csv, label_stats.json
```

## How sentences get labeled (weak supervision)

A specification has no gold-standard class labels, so we derive them from the
document's own **normative modality**:

| Label        | Triggered by (word-boundary match)                     |
|--------------|--------------------------------------------------------|
| MANDATORY    | shall, must, is/are required to, is mandatory          |
| PROHIBITED   | shall not, must not, may not, should not, cannot       |
| RECOMMENDED  | should, recommended, ought to                          |
| OPTIONAL     | may, optional, can optionally, is permitted/allowed    |
| INFORMATIVE  | none of the above (descriptive prose)                  |

The exact lexicon lives in [`src/config.py`](../src/config.py)
(`MODALITY_RULES`) — edit it there if you want a different taxonomy.

### Want to hand-correct the labels?

The weak labels are a starting point. To use *gold* labels instead, open
`data/dataset/labeled.csv`, fix the `label` column, and re-run only the model
steps (`--skip-parse`). Keeping `label` and `label_id` consistent matters —
`label_id` must equal the index of `label` in `config.LABELS`.
