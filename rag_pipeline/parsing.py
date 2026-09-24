"""
Stage 1: PDF -> one text record per page.

Swept knob: cfg.parser ("pypdf" | "pdfplumber" | "pymupdf").
Every parser returns the same shape (list of PageRecord), so everything
downstream is parser-agnostic and Stage 1 is a clean swap.

Page numbers are 1-based PHYSICAL PDF pages (page 1 = first page of the file),
not the page number printed in the footer. Citations and the eval set must use
the same convention, so it is fixed here once.

Run `python -m rag_pipeline.parsing` for a first look at all three parsers.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from rag_pipeline.config import BASELINE, PipelineConfig

import pdfplumber
from pypdf import PdfReader

try:  # PyMuPDF >= 1.24.3 exposes "pymupdf"; older versions only "fitz"
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf

# All parser imports live at the top on purpose: importing inside the parser
# functions would bill the one-off import cost to whichever parser runs first
# and skew the timing column.

DATA_DIR = Path("data/raw")

# doc_id -> (filename, business_line). A registry of inputs, not a tunable knob.
SOURCES: dict[str, tuple[str, str]] = {
    "contract_of_carriage": ("Delta_contract_of_carriage.pdf", "passenger"),
    "cargo_tariff": ("Delta_Cargo_Shipping_Rules_Tariff.pdf", "cargo"),
}


@dataclass(frozen=True)
class PageRecord:
    doc_id: str
    source_file: str
    business_line: str
    page: int   # 1-based physical PDF page
    text: str
    parser: str


# ---- the three parsers: same input, same output shape ----------------------
def _pypdf_pages(path: Path) -> list[str]:
    return [p.extract_text() or "" for p in PdfReader(str(path)).pages]


def _pdfplumber_pages(path: Path) -> list[str]:
    with pdfplumber.open(str(path)) as pdf:
        return [p.extract_text() or "" for p in pdf.pages]


def _pymupdf_pages(path: Path) -> list[str]:
    with pymupdf.open(str(path)) as doc:
        return [p.get_text("text") for p in doc]


_PARSERS = {"pypdf": _pypdf_pages, "pdfplumber": _pdfplumber_pages, "pymupdf": _pymupdf_pages}


def _path(doc_id: str) -> Path:
    path = DATA_DIR / SOURCES[doc_id][0]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run from the repo root; sources live in data/raw/.")
    return path


def parse_document(doc_id: str, cfg: PipelineConfig) -> list[PageRecord]:
    # Fail loudly rather than silently ignore a knob: otherwise a run's slug
    # would claim table extraction happened when it didn't.
    if cfg.extract_tables or cfg.strip_headers_footers:
        raise NotImplementedError("extract_tables / strip_headers_footers are not implemented yet.")
    filename, business_line = SOURCES[doc_id]
    texts = _PARSERS[cfg.parser](_path(doc_id))
    return [
        PageRecord(doc_id, filename, business_line, i, text, cfg.parser)
        for i, text in enumerate(texts, start=1)
    ]


def parse_all(cfg: PipelineConfig) -> list[PageRecord]:
    return [rec for doc_id in SOURCES for rec in parse_document(doc_id, cfg)]


# ---- diagnostic: a property of the PDF itself, not of any parser -----------
def image_coverage(doc_id: str) -> list[float]:
    """Fraction of each page's area covered by placed images (0.0 to 1.0).
    Near 1.0 = the page is essentially a picture (e.g. a scan)."""
    out = []
    with pymupdf.open(str(_path(doc_id))) as doc:
        for page in doc:
            area = page.rect.get_area()
            covered = sum((pymupdf.Rect(info["bbox"]) & page.rect).get_area()
                          for info in page.get_image_info())
            out.append(min(covered / area, 1.0) if area else 0.0)
    return out


if __name__ == "__main__":
    EMPTY = 50  # chars; a page with less than this has essentially no text layer
    for doc_id in SOURCES:
        cov = image_coverage(doc_id)
        print(f"\n=== {doc_id}: {len(cov)} pages | pages >80% covered by images: "
              f"{sum(c > 0.8 for c in cov)} ===")
        for parser in _PARSERS:
            t0 = time.perf_counter()
            recs = parse_document(doc_id, BASELINE.with_(parser=parser))
            ms = (time.perf_counter() - t0) * 1000
            chars = [len(r.text.strip()) for r in recs]
            print(f"  {parser:10s} pages={len(recs):3d}  total_chars={sum(chars):7d}  "
                  f"median_chars/page={statistics.median(chars):7.0f}  "
                  f"near-empty pages={sum(c < EMPTY for c in chars):3d}  time={ms:7.0f} ms")
