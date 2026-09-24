"""
Step 1 inspection: write each parser's full text to disk for side-by-side
reading, and count cheap automatic red flags per document.

Output goes to _cache/parse_inspect/ (gitignored: it's derived from the PDFs).
Run from the repo root:  python -m eval.inspect_parsing
"""
from __future__ import annotations

import re
from pathlib import Path

from rag_pipeline.config import BASELINE
from rag_pipeline.parsing import SOURCES, parse_document

OUT = Path("_cache/parse_inspect")
PARSERS = ["pypdf", "pdfplumber", "pymupdf"]

# Each flag counts one kind of extraction damage. None is proof of a problem
# on its own; they tell you WHERE to look when you read the text.
FLAGS = {
    "replacement_chars": lambda t: t.count("\ufffd"),                 # undecodable glyphs
    "cid_codes": lambda t: len(re.findall(r"\(cid:\d+\)", t)),        # font mapping failures
    "glued_words": lambda t: sum(
    len(w) > 30 and "http" not in w and "www." not in w
    and sum(c.isalpha() for c in w) > 0.8 * len(w)
    for w in t.split()), 
    "hyphen_linebreaks": lambda t: len(re.findall(r"\w-\n\w", t)),   # words split across lines
    "ligatures": lambda t: len(re.findall("[\ufb00-\ufb06]", t)),    # fi/fl as single glyphs
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    header = f"{'doc':22s} {'parser':10s} " + " ".join(f"{k:>18s}" for k in FLAGS)
    print(header)
    print("-" * len(header))
    for doc_id in SOURCES:
        for parser in PARSERS:
            pages = parse_document(doc_id, BASELINE.with_(parser=parser))
            body = "\n".join(f"\n===== page {p.page} =====\n{p.text}" for p in pages)
            (OUT / f"{doc_id}__{parser}.txt").write_text(body, encoding="utf-8")
            counts = {k: sum(f(p.text) for p in pages) for k, f in FLAGS.items()}
            print(f"{doc_id:22s} {parser:10s} " + " ".join(f"{v:>18d}" for v in counts.values()))
    print(f"\nFull text written to {OUT.resolve()}")
