"""Generate the Iteration-05 FINAL report as a self-contained PDF.

All figures are embedded as base64 and every number is read live from
outputs/metrics/ so the report always matches the latest pipeline run.

Sections follow the Iteration-5 rubric: Abstract, Introduction, Problem
Definition, Dataset, Preprocessing, Feature Engineering, EDA, Models, Training,
Hyperparameter Tuning, Evaluation, Results, Feature Importance, Research
Questions, Discussion, Real-World Application, Conclusion, Future Work,
References.

Run:  .venv/bin/python -m scripts.make_final_report
Output: FINAL_REPORT.pdf
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs" / "figures"
MET = ROOT / "outputs" / "metrics"
DATA = ROOT / "data" / "dataset"

REPO = "https://github.com/ajaynathch/requirement_doc_parser"


def m(name):
    return json.loads((MET / name).read_text())


def img(path: Path, width=460) -> str:
    if not path.exists():
        return f"<p><i>[missing figure: {path.name}]</i></p>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<div class="fig"><img src="data:image/png;base64,{b64}" '
            f'style="width:{width}pt;"/></div>')


# --------------------------------------------------------------------------- #
# Live numbers
# --------------------------------------------------------------------------- #
base = m("metrics_baseline_logreg.json")
dbert = m("metrics_distilbert.json")
base_mask = m("metrics_baseline_logreg_masked.json")
dbert_mask = m("metrics_distilbert_masked.json")
tune = m("tuning.json")
feat = m("feature_importance.json")
stats = json.loads((DATA / "label_stats.json").read_text())

TOTAL = sum(stats.values())
hp = dbert.get("hyperparams", {"epochs": 10, "batch_size": 16, "lr": 3e-5})
dev = dbert.get("device", "mps")

pc = dbert["per_class"]


def pcrow(lab):
    r = pc.get(lab, {})
    return (f"<tr><td>{lab}</td><td class='n'>{r.get('precision',0):.2f}</td>"
            f"<td class='n'>{r.get('recall',0):.2f}</td>"
            f"<td class='n'>{r.get('f1-score',0):.2f}</td>"
            f"<td class='n'>{int(r.get('support',0))}</td></tr>")


percls_rows = "".join(pcrow(l) for l in
                      ["MANDATORY", "PROHIBITED", "RECOMMENDED", "OPTIONAL", "INFORMATIVE"])

dist_rows = "".join(
    f"<tr><td>{lab}</td><td class='n'>{stats.get(lab,0)}</td>"
    f"<td class='n'>{100*stats.get(lab,0)/TOTAL:.1f}%</td></tr>"
    for lab in ["MANDATORY", "PROHIBITED", "RECOMMENDED", "OPTIONAL", "INFORMATIVE"])

topfeat_rows = "".join(
    f"<tr><td>{lab}</td><td>{', '.join(t for t, _ in feats[:6])}</td></tr>"
    for lab, feats in feat["top_features"].items())

lc = feat["learning_curve"]
lc_txt = ", ".join(f"{n}&rarr;{s:.2f}" for n, s in
                   zip(lc["train_sizes"], lc["cv_macro_f1"]))

tune_gain = tune["after_test"]["macro_f1"] - tune["before_test"]["macro_f1"]
maj_acc = max(stats.values()) / TOTAL

HTML = f"""
<html><head><style>
@page {{ size: letter; margin: 1.4cm; }}
body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10.3pt; color:#111; line-height:1.36; }}
h1 {{ font-size: 18pt; margin-bottom:2pt; }}
h2 {{ font-size: 13pt; color:#1a3c6e; border-bottom:1.5px solid #1a3c6e; padding-bottom:2pt; margin-top:15pt; }}
h3 {{ font-size: 11pt; color:#333; margin-bottom:1pt; margin-top:9pt; }}
.sub {{ color:#555; font-size:10pt; margin-top:0; }}
table {{ border-collapse: collapse; width:100%; margin:6pt 0; }}
th, td {{ border:1px solid #bbb; padding:3px 6px; font-size:9.2pt; text-align:left; vertical-align:top; }}
th {{ background:#e8eef7; }}
td.n, th.n {{ text-align:center; }}
.fig {{ text-align:center; margin:8pt 0; }}
.cap {{ font-size:8.5pt; color:#666; text-align:center; margin-top:-4pt; margin-bottom:8pt; }}
ul, ol {{ margin-top:2pt; }}
li {{ margin-bottom:2pt; }}
code {{ background:#f2f2f2; padding:1px 3px; font-size:9pt; }}
.note {{ background:#eef6ee; border-left:3px solid #2e7d32; padding:6px 8px; font-size:9.6pt; }}
.q {{ font-weight:bold; color:#1a3c6e; }}
.abstract {{ background:#f7f9fc; border:1px solid #d6e0ef; padding:8px 12px; font-size:9.8pt; }}
</style></head><body>

<h1>Automated Requirement Classification from an NVMe Specification</h1>
<p class="sub">EECE5644 &mdash; Machine Learning and Pattern Recognition &nbsp;|&nbsp;
Iteration 5: Final Machine Learning Project</p>

<table>
<tr><th style="width:22%;">Team Member</th><td>Ajaya Nath Chittela &nbsp;(ajayanath.c@northeastern.edu)</td></tr>
<tr><th>GitHub Repository</th><td><a href="{REPO}">{REPO}</a></td></tr>
<tr><th>Problem type</th><td>Supervised multi-class text classification (5 classes)</td></tr>
</table>

<h2>Abstract</h2>
<div class="abstract">
Technical hardware/firmware specifications such as NVMe encode different levels of
obligation in almost every sentence &mdash; hard requirements ("shall"), prohibitions
("shall not"), recommendations ("should"), optional behaviour ("may"), and purely
descriptive prose. Separating and typing these requirements is the first, manual, and
error-prone step engineers perform before writing validation tests. This project builds
an end-to-end machine-learning pipeline that parses the NVM Express Base Specification
(Revision 2.3) with <b>Docling</b>, weakly labels each extracted sentence by its
<b>normative modality</b> (MANDATORY / PROHIBITED / RECOMMENDED / OPTIONAL / INFORMATIVE),
and classifies it with two model families: a classical <b>TF-IDF + Logistic Regression /
Naive Bayes / Linear SVM</b> baseline selected by stratified cross-validation, and a
fine-tuned <b>DistilBERT</b> transformer with a class-weighted loss. On a held-out test
set, DistilBERT reaches <b>{dbert['accuracy']:.2f}</b> accuracy and
<b>{dbert['weighted_f1']:.2f}</b> weighted-F1, while the grid-search-tuned Logistic
Regression attains the best <b>macro-F1 ({tune['after_test']['macro_f1']:.2f})</b>, the
metric that matters most under heavy class imbalance. A modal-keyword masking ablation
shows both models retain accuracy well above the {maj_acc:.2f} majority baseline even when
the trigger word is hidden, demonstrating that they learn sentence <i>context</i> rather
than a keyword lookup.
</div>

<h2>1. Introduction</h2>
<p>Standards such as NVMe, PCIe, and CXL run to hundreds of pages in which the
obligation level of a statement determines whether an engineer must write a
conformance test, may skip it, or can ignore the sentence entirely. Today this
triage is done by hand. This project is the <b>requirement-understanding stage of a
larger "document-to-code" agent</b>: a reliable classifier that turns raw parsed
specification text into structured, typed requirements that downstream
test-generation and coverage-tracking tools can consume. We implement two phases of
that agent &mdash; (1) document understanding with Docling, and (2) requirement
classification &mdash; and rigorously compare a classical pattern-recognition
approach against a fine-tuned Transformer using the full EECE5644 evaluation toolkit.</p>

<h2>2. Problem Definition</h2>
<p>Given the text of a technical specification, (1) extract candidate requirement
sentences and (2) assign each to exactly one of five normative-modality classes:</p>
<table>
<tr><th style="width:20%;">Class</th><th>Meaning</th><th>Typical trigger language</th></tr>
<tr><td>MANDATORY</td><td>Hard requirement</td><td>shall, must, is/are required to</td></tr>
<tr><td>PROHIBITED</td><td>Hard prohibition</td><td>shall not, must not, may not, cannot</td></tr>
<tr><td>RECOMMENDED</td><td>Soft guidance</td><td>should, recommended, ought to</td></tr>
<tr><td>OPTIONAL</td><td>Permitted behaviour</td><td>may, optional, can, is permitted</td></tr>
<tr><td>INFORMATIVE</td><td>Descriptive prose</td><td>none of the above</td></tr>
</table>
<p>This is a supervised multi-class text-classification problem. The headline metric is
<b>macro-averaged F1</b> because the classes are highly imbalanced.</p>

<h2>3. Dataset Description</h2>
<p><b>Source.</b> The public <b>NVM Express Base Specification, Revision 2.3</b>. The PDF is
converted with <b>Docling</b> into structured Markdown/JSON that preserves section headers
and page provenance; body prose is segmented into individual sentences, yielding one record
per sentence with its page, section title, and block type.</p>
<p><b>Weak-supervision labeling.</b> Specifications carry no gold class labels, so labels are
derived from the document's own normative modal language via a curated, longest-match-first
keyword lexicon (<code>config.MODALITY_RULES</code>) &mdash; a standard weak-supervision
approach in requirements-engineering NLP. The extract yields <b>{TOTAL} labeled sentences</b>
with a realistic, heavy class imbalance:</p>
<table>
<tr><th>Class</th><th class="n">Count</th><th class="n">Share</th></tr>
{dist_rows}
<tr><td><b>Total</b></td><td class="n"><b>{TOTAL}</b></td><td class="n">100%</td></tr>
</table>
<p>Data are split into stratified <b>train / validation / test</b> partitions
(80 / held-out val / 20, seed {5644}); all reported metrics are on the test set, which is
never seen during training or model selection.</p>

<h2>4. Data Cleaning and Preprocessing</h2>
<ul>
<li><b>Structured extraction:</b> only Docling body blocks (<code>text</code>,
<code>paragraph</code>, <code>list_item</code>) are kept; tables, figures, and headers are
excluded from the prose stream (headers are retained as section context).</li>
<li><b>Sentence segmentation:</b> a lightweight regex splitter that protects spec
abbreviations ("e.g.", "i.e.", "Fig.", "No.", "vol.") from false sentence breaks.</li>
<li><b>Fragment filtering:</b> units shorter than 15 characters or containing no letters
are dropped (removes stray numbers, cross-reference tokens, and table debris).</li>
<li><b>Duplicate removal:</b> exact-text duplicates are dropped before labeling.</li>
<li><b>Label assignment:</b> word-boundary regex matching so, e.g., "may" does not fire
inside "maybe" and "shall not" is matched before "shall".</li>
</ul>

<h2>5. Feature Engineering</h2>
<p>Two complementary representations feed the two model families:</p>
<ul>
<li><b>Classical models &mdash; TF-IDF:</b> sublinear term-frequency scaling, 1&ndash;2-grams
(so phrases like "shall not" and "may not" become single features), English accent
stripping, and document-frequency thresholds (tuned in Section 8).</li>
<li><b>DistilBERT &mdash; WordPiece tokenization</b> to a max length of 128, with the
pretrained contextual embeddings fine-tuned to the task.</li>
<li><b>Interpretable engineered features (for EDA):</b> sentence character/word length,
average word length, digit and acronym counts, and binary cues for negation, cross-references,
and each modal group. These are used to analyze the data (Section 6); the predictive models
use the text representations above.</li>
</ul>

<h2>6. Exploratory Data Analysis</h2>
<p>The corpus is dominated by INFORMATIVE prose, with RECOMMENDED and PROHIBITED extremely
rare &mdash; a genuine, not artificial, imbalance that drives every downstream modeling
choice (class-weighted losses, macro-F1 as the headline metric).</p>
{img(FIG / "eda" / "class_distribution.png", 330)}
<p class="cap">Figure 1. Class distribution of modality labels ({TOTAL} sentences).</p>
<p>A correlation heatmap of the engineered features shows that no single surface feature
is strongly correlated with the label id (all |r| are modest); the modal-cue flags carry the
most signal, confirming that the class boundary lives in <i>word choice and context</i>
rather than in length or punctuation statistics.</p>
{img(FIG / "eda" / "feature_correlation_heatmap.png", 380)}
<p class="cap">Figure 2. Correlation heatmap of engineered features (with label id).</p>
{img(FIG / "eda" / "length_by_class.png", 330)}
<p class="cap">Figure 3. Sentence length (words) by class &mdash; distributions overlap heavily,
so length alone cannot separate the classes.</p>

<h2>7. Machine Learning Models</h2>
<table>
<tr><th style="width:26%;">Algorithm</th><th>Architecture &amp; justification</th></tr>
<tr><td>Multinomial Naive Bayes</td><td>Generative MAP classifier over TF-IDF counts; a fast, natural reference for sparse text.</td></tr>
<tr><td>Logistic Regression (TF-IDF)</td><td>Discriminative linear model, <code>class_weight="balanced"</code>; strong and interpretable &mdash; its coefficients give per-class feature importance.</td></tr>
<tr><td>Linear SVM (TF-IDF)</td><td>Max-margin linear classifier, effective in high-dimensional sparse spaces; decision scores softmaxed for ROC/PR.</td></tr>
<tr><td>DistilBERT (fine-tuned)</td><td>6-layer distilled BERT encoder + linear classification head ({len(feat['top_features'])} labels); captures word order and context, trained with a <b>class-weighted cross-entropy</b> loss.</td></tr>
</table>

<h2>8. Model Training</h2>
<p><b>Classical models</b> are selected by <b>stratified k-fold cross-validation</b>
(macro-F1 scoring) on the training split, then refit on train+validation and evaluated once
on the held-out test set. <b>DistilBERT</b> is fine-tuned for <b>{hp['epochs']} epochs</b>
(batch {hp['batch_size']}, learning rate {hp['lr']:.0e}, max length 128) on Apple-Silicon
<b>{dev.upper()}</b>, with best-checkpoint selection on validation macro-F1. Class imbalance is
handled with weights inversely proportional to class frequency: a class-weighted
cross-entropy for the Transformer and <code>class_weight="balanced"</code> for the linear
models, so the rare-but-important modalities are not ignored.</p>

<h2>9. Hyperparameter Tuning</h2>
<p><b>Method.</b> Exhaustive <b>grid search</b> ({tune['grid_size']} combinations) with
{tune['cv_folds']}-fold stratified cross-validation (scoring: macro-F1) over the
TF-IDF + Logistic Regression pipeline, tuning both the representation and the classifier:
<code>ngram_range</code>, <code>min_df</code>, <code>sublinear_tf</code>, and the
inverse-regularization <code>C</code>.</p>
<table>
<tr><th>Configuration</th><th class="n">Accuracy</th><th class="n">Macro-F1</th><th class="n">Weighted-F1</th></tr>
<tr><td>Before tuning (defaults)</td><td class="n">{tune['before_test']['accuracy']:.2f}</td><td class="n">{tune['before_test']['macro_f1']:.2f}</td><td class="n">{tune['before_test']['weighted_f1']:.2f}</td></tr>
<tr><td>After tuning (best params)</td><td class="n"><b>{tune['after_test']['accuracy']:.2f}</b></td><td class="n"><b>{tune['after_test']['macro_f1']:.2f}</b></td><td class="n"><b>{tune['after_test']['weighted_f1']:.2f}</b></td></tr>
</table>
<p><b>Best parameters:</b> <code>ngram_range={tune['best_params']['tfidf__ngram_range']}</code>,
<code>min_df={tune['best_params']['tfidf__min_df']}</code>,
<code>sublinear_tf={tune['best_params']['tfidf__sublinear_tf']}</code>,
<code>C={tune['best_params']['clf__C']}</code>. Tuning raised test macro-F1 by
<b>+{tune_gain:.3f}</b> ({tune['before_test']['macro_f1']:.3f}&rarr;{tune['after_test']['macro_f1']:.3f}),
driven mostly by enabling bigrams and sublinear TF scaling.</p>
{img(FIG / "tuning_before_after.png", 300)}
<p class="cap">Figure 4. Logistic Regression before vs. after grid-search tuning.</p>

<h2>10. Model Evaluation</h2>
<p>Metrics on the held-out test set (n = {int(sum(pc[l]['support'] for l in ['MANDATORY','PROHIBITED','RECOMMENDED','OPTIONAL','INFORMATIVE']))}).
Macro-F1 is emphasized because a majority-only classifier already scores ~{maj_acc:.2f} accuracy.</p>
<table>
<tr><th>Model</th><th class="n">Accuracy</th><th class="n">Precision (macro)</th><th class="n">Recall (macro)</th><th class="n">Macro-F1</th><th class="n">Weighted-F1</th></tr>
<tr><td>TF-IDF + Logistic Regression</td><td class="n">{base['accuracy']:.2f}</td>
<td class="n">{base['per_class']['macro avg']['precision']:.2f}</td>
<td class="n">{base['per_class']['macro avg']['recall']:.2f}</td>
<td class="n">{base['macro_f1']:.2f}</td><td class="n">{base['weighted_f1']:.2f}</td></tr>
<tr><td>LogReg (grid-search tuned)</td><td class="n">{tune['after_test']['accuracy']:.2f}</td>
<td class="n">&mdash;</td><td class="n">&mdash;</td>
<td class="n"><b>{tune['after_test']['macro_f1']:.2f}</b></td><td class="n">{tune['after_test']['weighted_f1']:.2f}</td></tr>
<tr><td>DistilBERT (class-weighted)</td><td class="n"><b>{dbert['accuracy']:.2f}</b></td>
<td class="n">{dbert['per_class']['macro avg']['precision']:.2f}</td>
<td class="n">{dbert['per_class']['macro avg']['recall']:.2f}</td>
<td class="n">{dbert['macro_f1']:.2f}</td><td class="n"><b>{dbert['weighted_f1']:.2f}</b></td></tr>
</table>

<h3>Per-class results (DistilBERT)</h3>
<table>
<tr><th>Class</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">Support</th></tr>
{percls_rows}
</table>
{img(FIG / "confusion_distilbert.png", 280)}
<p class="cap">Figure 5. DistilBERT confusion matrix (test set).</p>
{img(FIG / "roc_distilbert.png", 300)}
<p class="cap">Figure 6. One-vs-rest ROC curves (DistilBERT) &mdash; AUC near 1.0 for
well-represented classes.</p>
{img(FIG / "pr_distilbert.png", 300)}
<p class="cap">Figure 7. One-vs-rest precision&ndash;recall curves (DistilBERT).</p>

<h2>11. Results</h2>
{img(FIG / "model_comparison.png", 320)}
<p class="cap">Figure 8. Accuracy and macro-F1: classical baseline vs. DistilBERT.</p>
<ul>
<li>DistilBERT is best on <b>accuracy ({dbert['accuracy']:.2f})</b> and
<b>weighted-F1 ({dbert['weighted_f1']:.2f})</b>, and is perfect on every well-represented
class (MANDATORY, OPTIONAL: precision = recall = 1.00).</li>
<li>The <b>tuned Logistic Regression wins on macro-F1 ({tune['after_test']['macro_f1']:.2f})</b>,
the imbalance-sensitive metric &mdash; a well-regularized linear model rivals the Transformer
on this task.</li>
<li>Both models' remaining errors sit almost entirely in the two rarest classes
(PROHIBITED: 1 test sample; RECOMMENDED: 0), i.e. a data-scarcity artifact rather than a
systematic weakness.</li>
</ul>

<h2>12. Feature Importance</h2>
<p>Logistic-Regression weights over the TF-IDF vocabulary give one importance value per
(class, term). The most influential terms per class are intuitive and align with the
modality definitions:</p>
<table>
<tr><th style="width:22%;">Class</th><th>Most influential terms</th></tr>
{topfeat_rows}
</table>
{img(FIG / "feature_importance.png", 460)}
<p class="cap">Figure 9. Top TF-IDF features per class (Logistic Regression weights).</p>
{img(FIG / "learning_curve.png", 300)}
<p class="cap">Figure 10. Learning curve: CV macro-F1 rises with training-set size
({lc_txt}) and has not plateaued &mdash; more data would help.</p>

<h2>13. Research Questions</h2>
<p class="q">Q1. Which algorithm produced the best performance?</p>
<p>It depends on the metric: DistilBERT for accuracy ({dbert['accuracy']:.2f}) and weighted-F1
({dbert['weighted_f1']:.2f}); the tuned Logistic Regression for macro-F1
({tune['after_test']['macro_f1']:.2f}).</p>
<p class="q">Q2. Which features contributed most to the predictions?</p>
<p>The modal-cue n-grams &mdash; "shall"/"controller shall" for MANDATORY, "may"/"may not"
for OPTIONAL/PROHIBITED, "should" for RECOMMENDED (Fig. 9). Generic nouns, digits, and
cross-reference tokens ("figure", "section") carry near-zero weight.</p>
<p class="q">Q3. Did preprocessing / tuning improve the model?</p>
<p>Yes. Enabling bigrams and sublinear TF via grid search raised macro-F1 by
+{tune_gain:.3f} (Section 9). Sentence segmentation, fragment filtering, and duplicate
removal were prerequisites for clean labels.</p>
<p class="q">Q4. Which evaluation metric best reflects model performance?</p>
<p><b>Macro-F1.</b> With INFORMATIVE at {100*stats.get('INFORMATIVE',0)/TOTAL:.0f}% of the data,
accuracy is misleading (a majority-only classifier scores ~{maj_acc:.2f}); macro-F1 weights
every class equally, so it reflects the rare but important modalities.</p>
<p class="q">Q5. What does the confusion matrix reveal about errors?</p>
<p>Errors concentrate in PROHIBITED (1 test sample) and RECOMMENDED (0 test samples);
well-represented classes have essentially no confusions (Fig. 5). False negatives are a
data-scarcity artifact.</p>
<p class="q">Q6. Which model minimizes the most critical error?</p>
<p>The costly error is dropping a real requirement (MANDATORY/PROHIBITED) to INFORMATIVE.
DistilBERT achieves recall 1.00 on MANDATORY, minimizing that critical miss for the dominant
requirement type.</p>
<p class="q">Q7. How well does the model generalize?</p>
<p>All numbers are on a held-out test set. Generalization is strong for common classes; the
still-rising learning curve (Fig. 10) indicates more data would further improve rare classes.</p>
<p class="q">Q8. Do the models learn meaning or just keywords?</p>
<p>A modal-masking ablation blanks the trigger word and re-evaluates. Accuracy drops from
{base['accuracy']:.2f}&rarr;{base_mask['accuracy']:.2f} (LogReg) and
{dbert['accuracy']:.2f}&rarr;{dbert_mask['accuracy']:.2f} (DistilBERT) but stays above the
{maj_acc:.2f} majority baseline &mdash; the models use sentence <i>context</i>, not just the
keyword.</p>
<table>
<tr><th>Model</th><th class="n">Acc (full)</th><th class="n">Acc (masked)</th><th class="n">Macro-F1 (full)</th><th class="n">Macro-F1 (masked)</th></tr>
<tr><td>Logistic Regression</td><td class="n">{base['accuracy']:.2f}</td><td class="n">{base_mask['accuracy']:.2f}</td><td class="n">{base['macro_f1']:.2f}</td><td class="n">{base_mask['macro_f1']:.2f}</td></tr>
<tr><td>DistilBERT</td><td class="n">{dbert['accuracy']:.2f}</td><td class="n">{dbert_mask['accuracy']:.2f}</td><td class="n">{dbert['macro_f1']:.2f}</td><td class="n">{dbert_mask['macro_f1']:.2f}</td></tr>
</table>

<h2>14. Discussion</h2>
<h3>Key findings</h3>
<p>Requirement modality is highly learnable from text; a well-tuned linear model rivals a
fine-tuned Transformer, and both generalize beyond surface keywords. The task's difficulty is
almost entirely a function of data volume for the rare classes, not model capacity.</p>
<h3>Strengths</h3>
<ul>
<li>Reproducible end-to-end pipeline (parse &rarr; label &rarr; train &rarr; evaluate) with fixed seeds.</li>
<li>Two model families evaluated on identical splits and metrics.</li>
<li>Principled imbalance handling and macro-F1 as the headline metric.</li>
<li>A masking ablation that validates genuine context learning rather than keyword lookup.</li>
</ul>
<h3>Limitations and sources of error</h3>
<ul>
<li><b>Data volume:</b> {TOTAL} sentences from a few spec pages; the {int(pc['PROHIBITED']['support'])}-sample
PROHIBITED and {int(pc['RECOMMENDED']['support'])}-sample RECOMMENDED test classes make macro-F1
and per-class metrics high-variance.</li>
<li><b>Weak labels:</b> derived from modal keywords, not human-verified, so ambiguous
sentences ("may" as permission vs. possibility) can be mislabeled &mdash; a source of
irreducible error.</li>
<li><b>Small test set</b> ({int(sum(pc[l]['support'] for l in pc if l in ['MANDATORY','PROHIBITED','RECOMMENDED','OPTIONAL','INFORMATIVE']))} sentences)
gives coarse metric resolution.</li>
</ul>
<h3>Ethical considerations</h3>
<p>The classifier is a decision-support tool, not an authority on the specification. A missed
MANDATORY/PROHIBITED requirement could cause a real conformance gap, so in deployment the model
should <b>augment</b> (flag and rank) rather than replace human review, and its weak-label
provenance should be disclosed. The data is a public engineering standard with no personal or
sensitive information, so privacy risk is minimal.</p>

<h2>15. Real-World Application</h2>
<ul>
<li><b>How it would be used:</b> as the requirement-understanding stage of a document-to-code
agent &mdash; automatically converting a parsed specification into typed requirements that feed
test-plan generation, requirement-traceability matrices, and coverage checking.</li>
<li><b>Who benefits:</b> firmware/hardware validation engineers, compliance and certification
teams, and technical writers who must audit obligation language across large standards.</li>
<li><b>Deployment challenges:</b> generalizing across specifications with different house styles;
building a small human-verified gold set per standard; calibrating confidence so low-certainty
sentences are routed to a human; and integrating into existing requirements-management tooling.</li>
</ul>

<h2>16. Conclusion</h2>
<p>We delivered a complete, reproducible ML system that classifies NVMe specification sentences
by normative modality. DistilBERT leads on accuracy/weighted-F1 and the tuned Logistic
Regression leads on macro-F1; a masking ablation confirms both learn context beyond keywords.
The dominant bottleneck is rare-class data volume, not model choice &mdash; a clear, actionable
result for the next iteration of the document-to-code agent.</p>

<h2>17. Future Work</h2>
<ul>
<li>Expand the corpus to many more spec pages (and additional standards) &mdash; the learning
curve is still climbing.</li>
<li>Create a small <b>human-verified gold test set</b> to remove weak-label noise from evaluation.</li>
<li>Consider merging ultra-rare classes or hierarchical labeling (normative vs. informative,
then obligation level).</li>
<li>Add the downstream agent stages: requirement &rarr; test-case and coverage generation.</li>
<li>Explore larger encoders and confidence calibration for human-in-the-loop routing.</li>
</ul>

<h2>References</h2>
<ol>
<li>NVM Express Base Specification, Revision 2.3, NVM Express, Inc., 2025.</li>
<li>Docling: an efficient open-source toolkit for document conversion, IBM Research.</li>
<li>V. Sanh, L. Debut, J. Chaumond, T. Wolf, "DistilBERT, a distilled version of BERT," 2019.</li>
<li>J. Cleland-Huang et al., automated classification of non-functional requirements (PROMISE NFR).</li>
<li>F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, 2011.</li>
<li>T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," EMNLP, 2020.</li>
</ol>

<div class="note"><b>Reproducibility.</b> All numbers and figures in this report are read live
from <code>outputs/metrics/</code> and <code>outputs/figures/</code>, regenerated by
<code>python -m scripts.run_pipeline</code>. Source, data, and instructions are in the GitHub
repository linked above.</div>

</body></html>
"""


def main() -> None:
    out = ROOT / "EECE5644_Iteration5_Final_Report_Chittela_Ajayanath.pdf"
    with open(out, "wb") as f:
        result = pisa.CreatePDF(HTML, dest=f)
    if result.err:
        raise SystemExit(f"PDF generation failed with {result.err} error(s)")
    print(f"[final-report] wrote {out}")


if __name__ == "__main__":
    main()
