"""
Step 1 diagnostic: the parsers extract the same characters, so where do they
put them in a different ORDER? Compares whitespace-stripped text page by page
against PyMuPDF and shows the first point of divergence.

Run from the repo root:  python -m eval.diff_order
"""
from __future__ import annotations

from rag_pipeline.config import BASELINE
from rag_pipeline.parsing import SOURCES, parse_document

REFERENCE = "pymupdf"
OTHERS = ["pypdf", "pdfplumber"]
CONTEXT = 50  # characters of context to print either side of the divergence


def squash(text: str) -> str:
    return "".join(c for c in text if not c.isspace())


if __name__ == "__main__":
    for doc_id in SOURCES:
        ref = [squash(p.text) for p in parse_document(doc_id, BASELINE.with_(parser=REFERENCE))]
        for other in OTHERS:
            cmp = [squash(p.text) for p in parse_document(doc_id, BASELINE.with_(parser=other))]
            diff_pages = [i + 1 for i, (a, b) in enumerate(zip(ref, cmp)) if a != b]
            same_chars = [sorted(ref[p - 1]) == sorted(cmp[p - 1]) for p in diff_pages]
            print(f"\n=== {doc_id}: {other} vs {REFERENCE} ===")
            print(f"pages in different order: {diff_pages or 'none'}")
            print(f"  ...of which same characters, just reordered: {sum(same_chars)}/{len(diff_pages)}")
            if diff_pages:
                p = diff_pages[0]
                a, b = ref[p - 1], cmp[p - 1]
                k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
                print(f"  first divergence on page {p} at character {k}:")
                print(f"    {REFERENCE:10s} ...{a[max(0, k - CONTEXT):k + CONTEXT]}...")
                print(f"    {other:10s} ...{b[max(0, k - CONTEXT):k + CONTEXT]}...")
