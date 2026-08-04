"""Phase 1 -- parse an NVMe specification document with Docling.

Pipeline:
    raw PDF/DOCX  ->  Docling  ->  markdown + structured json
                                ->  sentence-level records with:
                                      * page number   (from Docling provenance)
                                      * section title  (nearest preceding header)
                                      * block type     (paragraph / list_item / ...)

The output is a tidy list of text records that Phase 2 turns into a labeled
dataset.  We deliberately keep parsing and labeling separate so the raw
extraction can be inspected before any ML assumptions are baked in.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from docling.document_converter import DocumentConverter

from . import config as C

# Docling block labels that carry running prose (vs. tables, figures, headers).
_BODY_LABELS = {"text", "paragraph", "list_item"}
_HEADER_LABELS = {"section_header", "title"}

# A lightweight sentence splitter.  We avoid heavyweight NLP deps and instead
# split on sentence punctuation while protecting common spec abbreviations and
# the "e.g." / "i.e." forms that would otherwise cause false splits.
_ABBREV = r"(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bFig)(?<!\bNo)(?<!\bvol)"
_SENT_SPLIT = re.compile(rf"{_ABBREV}(?<=[.;:])\s+(?=[A-Z0-9])")


@dataclass
class TextRecord:
    """One extracted unit of prose from the document."""

    text: str
    page: int | None
    section: str
    block_type: str
    source: str


def _split_sentences(block: str) -> list[str]:
    block = re.sub(r"\s+", " ", block).strip()
    if not block:
        return []
    parts = _SENT_SPLIT.split(block)
    return [p.strip() for p in parts if p.strip()]


def find_documents(raw_dir: Path = C.RAW_DIR) -> list[Path]:
    """Return every parseable document dropped into data/raw."""
    exts = {".pdf", ".docx", ".html", ".htm", ".pptx", ".md"}
    return sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in exts)


def parse_document(path: Path) -> tuple[str, list[TextRecord]]:
    """Convert one document; return (markdown, sentence-level records)."""
    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document

    markdown = doc.export_to_markdown()

    records: list[TextRecord] = []
    current_section = "(preamble)"

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        label = getattr(label, "value", label)  # enum -> str
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue

        if label in _HEADER_LABELS:
            current_section = text.strip()
            continue
        if label not in _BODY_LABELS:
            continue

        page = None
        prov = getattr(item, "prov", None)
        if prov:
            page = getattr(prov[0], "page_no", None)

        for sentence in _split_sentences(text):
            # Skip fragments that are almost certainly not requirement prose.
            if len(sentence) < 15 or not re.search(r"[a-zA-Z]", sentence):
                continue
            records.append(
                TextRecord(
                    text=sentence,
                    page=page,
                    section=current_section,
                    block_type=label,
                    source=path.name,
                )
            )

    return markdown, records


def parse_all(raw_dir: Path = C.RAW_DIR, parsed_dir: Path = C.PARSED_DIR) -> list[TextRecord]:
    """Parse every document in raw_dir, writing markdown/json artifacts."""
    docs = find_documents(raw_dir)
    if not docs:
        raise FileNotFoundError(
            f"No documents found in {raw_dir}. "
            "Drop the NVMe spec pages (PDF/DOCX) there and re-run."
        )

    all_records: list[TextRecord] = []
    for path in docs:
        print(f"[parse] {path.name} ...")
        markdown, records = parse_document(path)

        stem = path.stem
        (parsed_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
        (parsed_dir / f"{stem}.records.json").write_text(
            json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8"
        )
        print(f"[parse]   -> {len(records)} sentence records "
              f"(markdown: {parsed_dir / (stem + '.md')})")
        all_records.extend(records)

    (parsed_dir / "all_records.json").write_text(
        json.dumps([asdict(r) for r in all_records], indent=2), encoding="utf-8"
    )
    print(f"[parse] total {len(all_records)} records from {len(docs)} document(s).")
    return all_records


if __name__ == "__main__":
    parse_all()
