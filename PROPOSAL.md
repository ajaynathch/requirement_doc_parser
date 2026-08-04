# Project Proposal — Automated Requirement Classification from Technical Specifications

**Course:** EECE5644: Machine Learning and Pattern Recognition
**Author:** Chittela Ajayanath (ajayanath.c@northeastern.edu)
**Date:** July 8, 2026

---

## 1. Problem Statement

Hardware and firmware specifications (NVMe, PCIe, CXL, and similar standards) are
dense documents in which every sentence carries a different level of obligation.
A single page mixes **hard requirements** ("the controller *shall* …"),
**prohibitions** ("*shall not* …"), **recommendations** ("*should* …"),
**optional behavior** ("*may* …"), and purely **descriptive** prose. Engineers
who write validation tests, trace coverage, or generate implementation stubs must
first separate the *normative* requirements from the informative text and
identify each requirement's obligation level. Today this is done manually and is
slow, inconsistent, and error-prone.

**Goal.** Build a machine-learning system that, given the text of a technical
specification, automatically (1) extracts candidate requirement sentences and
(2) classifies each one by its **normative modality**:
`MANDATORY`, `PROHIBITED`, `RECOMMENDED`, `OPTIONAL`, or `INFORMATIVE`.

This is the requirement-understanding stage of a larger "document-to-code" agent:
a reliable classifier is what turns raw parsed spec text into the structured,
typed requirements that downstream test-generation or code-review tools consume.

## 2. Motivation and Relevance to the Course

This is a supervised **multi-class text classification / pattern-recognition**
problem and exercises the core EECE5644 toolkit end to end:

- **Feature representation** — TF-IDF n-gram vectors vs. learned contextual
  embeddings.
- **Generative vs. discriminative models** — Multinomial Naive Bayes (a MAP
  classifier over word counts) against Logistic Regression and a linear SVM.
- **Deep learning** — fine-tuning a pretrained Transformer (DistilBERT) with a
  classification head, including class-imbalance handling via a weighted loss.
- **Model selection and evaluation** — stratified k-fold cross-validation,
  per-class precision/recall/F1, macro/weighted averaging, confusion matrices,
  and one-vs-rest ROC / precision–recall analysis.

It also connects theory to a real systems-engineering workflow, matching the
author's background in SSD firmware validation and storage/SoC specifications.

## 3. Data

**Source.** The public **NVM Express (NVMe) Base Specification, Revision 2.3**.
The document is parsed with **Docling**, which converts the PDF into structured
Markdown/JSON while preserving section headers and page provenance. Body prose is
segmented into individual sentences, yielding one record per sentence with its
page number, section title, and block type.

**Labeling (weak supervision).** Specifications have no gold-standard class
labels, so labels are derived from the document's own **normative modal
language** using a curated keyword lexicon (e.g., "shall"/"must" → `MANDATORY`,
"shall not"/"must not" → `PROHIBITED`, "should" → `RECOMMENDED`, "may"/"optional"
→ `OPTIONAL`, otherwise `INFORMATIVE`). This is a standard weak-supervision
approach in requirements-engineering NLP.

**Guarding against trivial keyword matching.** Because the labels come from modal
keywords, a model could "cheat" by memorizing those words. To demonstrate the
model learns sentence *context* rather than a lookup rule, an ablation
(`--mask-modals`) blanks the modal trigger word and re-evaluates; accuracy is
reported both with and without masking.

**Scale.** A representative extract currently yields ~125 labeled sentences with
a realistic, heavy class imbalance (`INFORMATIVE` dominates; `RECOMMENDED` and
`PROHIBITED` are rare). The corpus is expanded simply by adding more spec pages,
and imbalance is addressed methodologically (see §4).

## 4. Proposed Approach

```
NVMe spec (PDF)
      │  Docling  (Phase 1: document understanding)
      ▼
structured sentences  ─►  weak modality labeling  ─►  stratified train/val/test
      │
      ├── Classical baseline:  TF-IDF  +  {Naive Bayes, Logistic Regression, Linear SVM}
      │                        model selection by stratified k-fold CV
      │
      └── Deep model:          DistilBERT fine-tune, class-weighted cross-entropy
                               (Apple-Silicon MPS / CUDA / CPU)
      ▼
Evaluation & comparison:  P/R/F1 (per-class + macro/weighted),
                          confusion matrices, ROC & PR curves
```

**Handling class imbalance.** The Transformer uses a class-weighted
cross-entropy loss (weights inversely proportional to class frequency) so the
rare-but-important modalities are not ignored; the classical models use
`class_weight="balanced"`. Macro-F1 is the headline metric so minority classes
count equally.

**Deliverables.** A reproducible pipeline (parse → dataset → train → evaluate),
saved model artifacts, and a set of figures/metrics comparing the classical and
deep approaches, plus the masking ablation.

## 5. Evaluation Plan

- **Primary metric:** macro-averaged F1 (robust to imbalance).
- **Secondary:** accuracy, weighted-F1, per-class precision/recall/F1.
- **Diagnostics:** confusion matrices and one-vs-rest ROC / PR curves per model.
- **Model selection:** stratified k-fold cross-validation on the training split
  for the classical models; validation-set macro-F1 with best-checkpoint
  selection for the Transformer.
- **Ablation:** modal-keyword masking to test context learning.
- **Comparison:** classical vs. deep model on a common held-out test set.

## 6. Preliminary Results (proof of concept)

The full pipeline already runs end-to-end on the NVMe 2.3 extract (held-out test
set, 25 sentences):

| Model | Accuracy | Weighted-F1 | Macro-F1 |
|-------|:--------:|:-----------:|:--------:|
| TF-IDF + Logistic Regression | 0.88 | 0.88 | 0.86 |
| DistilBERT (class-weighted) | **0.96** | **0.94** | 0.74 |

The Transformer already outperforms the classical baseline on accuracy and
weighted-F1 and is perfect on the well-represented classes; its lower macro-F1
is driven entirely by the two rarest classes (1–2 examples), which motivates the
data-expansion and imbalance-handling work in the full project.

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Small / imbalanced dataset | Add more spec pages; class-weighted loss; report macro-F1; consider merging ultra-rare classes |
| Labels reflect keywords, not meaning | Modal-masking ablation to prove context learning |
| High variance on a tiny test set | Larger corpus; fixed seeds; report CV mean ± std |
| Weak labels are noisy | Optional manual correction pass on `labeled.csv` for a gold test set |

## 8. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Docling parsing + weak-labeling pipeline (**done**) |
| 2 | Classical baselines + cross-validated model selection (**done**) |
| 3 | DistilBERT fine-tuning + imbalance handling (**done, proof of concept**) |
| 4 | Corpus expansion, hyperparameter tuning, masking ablation |
| 5 | Full evaluation, error analysis, figures |
| 6 | Final report and presentation |

## 9. Expected Outcomes

A reproducible, well-evaluated system that classifies specification sentences by
normative modality, with a rigorous comparison of classical pattern-recognition
methods against a fine-tuned Transformer, quantifying the accuracy/data trade-off
between them and demonstrating (via ablation) that the learned model captures
context beyond surface keywords.

## References (to be finalized)

1. NVM Express Base Specification, Revision 2.3, NVM Express, Inc.
2. Docling: Document parsing toolkit (IBM Research).
3. V. Sanh et al., "DistilBERT, a distilled version of BERT," 2019.
4. J. Cleland-Huang et al., work on automated classification of non-functional
   requirements (PROMISE NFR dataset).
5. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, 2011.
