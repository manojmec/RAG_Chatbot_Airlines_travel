"""
Single source of truth for every experimental knob.

One PipelineConfig instance == one fully specified run. Ablation runners build
variants with `base.with_(field=value)`; modules read everything from the config
object they are handed. If you find a tunable number hard-coded inside a module,
it belongs here instead.

Run `python -m rag_pipeline.config` to smoke-test this file.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Literal, Optional, get_args, get_origin, get_type_hints

# Stage numbers match EVALUATION_METHODOLOGY.md Part C.
PARSING, CHUNKING, EMBEDDING, VECTORSTORE, RETRIEVAL, FUSION, RERANK, GENERATION = range(1, 9)


def knob(default, stage: int, *, affects_results: bool = True):
    """Declare a config field and record which ablation stage owns it.

    stage:           drives per-stage cache keys (see stage_key()).
    affects_results: False for knobs that change speed but never outputs
                     (batch size, device). They are left out of slugs so they
                     never split a results table or invalidate a cache.
    """
    return field(default=default, metadata={"stage": stage, "affects_results": affects_results})


@dataclass(frozen=True)
class PipelineConfig:
    # ---- Stage 1: parsing ---------------------------------------------------
    parser: Literal["pypdf", "pdfplumber", "pymupdf"] = knob("pymupdf", PARSING)
    # Structured table extraction (pdfplumber, or PyMuPDF's page.find_tables()).
    extract_tables: bool = knob(False, PARSING)
    # Remove running headers/footers repeated on every page. Decide whether this
    # matters AFTER you have eyeballed the Step 1 output, not before.
    strip_headers_footers: bool = knob(False, PARSING)

    # ---- Stage 2: chunking --------------------------------------------------
    # "fixed+overlap" is not its own strategy: it is "fixed" with chunk_overlap > 0.
    # "structural" = split on the documents' own rule/section headings.
    chunk_strategy: Literal["fixed", "recursive", "semantic", "structural"] = knob("recursive", CHUNKING)
    chunk_size: int = knob(500, CHUNKING)      # tokens; sweep 300 / 500 / 800
    chunk_overlap: int = knob(50, CHUNKING)    # tokens; sweep 0 / 50 / 100
    # Tokenizer used to MEASURE chunk size. Held fixed so "500 tokens" means the
    # same thing in every run whatever the embedding model.
    # WARNING: embedding models truncate silently. all-MiniLM-L6-v2 stops at 256
    # word-pieces; bge-small and multi-qa-mpnet at 512. An 800-token chunk is
    # partly invisible to them. Remember this when reading Stage 2 x Stage 3 results.
    token_encoding: str = knob("cl100k_base", CHUNKING)
    # Semantic chunking only: split where adjacent-sentence distance exceeds
    # this percentile.
    semantic_breakpoint_percentile: float = knob(95.0, CHUNKING)
    # Semantic chunking uses its OWN embedding model, pinned separately, so that
    # changing the Stage 3 retrieval model never silently changes the chunks too.
    semantic_chunker_model: str = knob("sentence-transformers/all-MiniLM-L6-v2", CHUNKING)

    # ---- Stage 3: embeddings ------------------------------------------------
    # Candidates: sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5,
    # text-embedding-3-small (OpenAI API), sentence-transformers/multi-qa-mpnet-base-dot-v1
    # Model-specific quirks (e.g. bge's query prefix) live in a registry inside
    # embeddings.py: they are properties of the model, not independent knobs.
    embedding_model: str = knob("BAAI/bge-small-en-v1.5", EMBEDDING)
    # Note: multi-qa-mpnet-base-dot-v1 was trained for raw dot product;
    # normalising turns it into cosine. Worth one check when you reach Stage 3.
    normalize_embeddings: bool = knob(True, EMBEDDING)
    embedding_batch_size: int = knob(32, EMBEDDING, affects_results=False)
    device: str = knob("cpu", EMBEDDING, affects_results=False)

    # ---- Stage 4: vector store ----------------------------------------------
    vector_store: Literal["chroma", "faiss"] = knob("chroma", VECTORSTORE)
    distance: Literal["cosine"] = knob("cosine", VECTORSTORE)  # fixed, not swept

    # ---- Stage 5: retrieval mode --------------------------------------------
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = knob("dense", RETRIEVAL)
    # Depth of the ranked list each retriever returns. Metrics are computed on
    # this list, so it must be >= 10 for MRR@10; 20 feeds the Stage 7 reranker.
    candidate_k: int = knob(20, RETRIEVAL)
    bm25_tokenizer: Literal["simple", "stemmed"] = knob("simple", RETRIEVAL)
    bm25_k1: float = knob(1.5, RETRIEVAL)     # rank-bm25 defaults; recorded, not swept
    bm25_b: float = knob(0.75, RETRIEVAL)
    # Business-line routing (restrict search to passenger / cargo / financial).
    # NOT one of the 8 required stages. Baseline is "none": let retrieval route
    # on its own, which is exactly what the Stage 5 per-line breakdown measures.
    routing: Literal["none", "keyword", "llm"] = knob("none", RETRIEVAL)

    # ---- Stage 6: fusion (hybrid only) --------------------------------------
    fusion: Literal["rrf", "weighted"] = knob("rrf", FUSION)
    rrf_k: int = knob(60, FUSION)
    alpha: float = knob(0.5, FUSION)          # weighted only: dense weight; sweep >= 3 values
    score_norm: Literal["minmax", "zscore"] = knob("minmax", FUSION)

    # ---- Stage 7: reranking -------------------------------------------------
    reranker: Optional[str] = knob(None, RERANK)  # None | "cross-encoder/ms-marco-MiniLM-L-6-v2"
    final_k: int = knob(5, RERANK)                # chunks handed to the generator (3 or 5)

    # ---- Stage 8: generation ------------------------------------------------
    # Verify model IDs against provider docs when you get here; pin dated
    # snapshots so a silent model update can't change your results.
    generator_provider: Literal["anthropic", "openai", "ollama"] = knob("anthropic", GENERATION)
    generator_model: str = knob("claude-haiku-4-5-20251001", GENERATION)
    temperature: float = knob(0.0, GENERATION)
    max_output_tokens: int = knob(512, GENERATION)
    # The prompt template is a hidden variable. Bump this whenever you edit it,
    # or two "identical" runs will silently differ.
    prompt_version: str = knob("v1", GENERATION)

    # ---- validation ---------------------------------------------------------
    def __post_init__(self) -> None:
        errors: list[str] = []
        hints = get_type_hints(type(self))
        for f in fields(self):
            hint, value = hints[f.name], getattr(self, f.name)
            if get_origin(hint) is Literal and value not in get_args(hint):
                errors.append(f"{f.name}={value!r} not in {get_args(hint)}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            errors.append("need 0 <= chunk_overlap < chunk_size")
        if self.candidate_k < 10:
            errors.append("candidate_k must be >= 10 or MRR@10 is undefined")
        if not 0 < self.final_k <= self.candidate_k:
            errors.append("need 0 < final_k <= candidate_k")
        if not 0.0 <= self.alpha <= 1.0:
            errors.append("alpha must be in [0, 1]")
        if self.temperature != 0.0:
            errors.append("methodology fixes generator temperature at 0")
        if errors:
            raise ValueError("Invalid PipelineConfig:\n  " + "\n  ".join(errors))

    # ---- identity -----------------------------------------------------------
    def _inactive(self) -> set[str]:
        """Knobs that cannot affect this run (e.g. alpha when using RRF).
        Excluded from hashes so two genuinely identical runs get the same slug."""
        off: set[str] = set()
        if self.chunk_strategy != "semantic":
            off |= {"semantic_breakpoint_percentile", "semantic_chunker_model"}
        if self.retrieval_mode == "dense":
            off |= {"bm25_tokenizer", "bm25_k1", "bm25_b"}
        if self.retrieval_mode != "hybrid":
            off |= {"fusion", "rrf_k", "alpha", "score_norm"}
        elif self.fusion == "rrf":
            off |= {"alpha", "score_norm"}
        else:
            off.add("rrf_k")
        return off

    def result_knobs(self, upto: int = GENERATION) -> dict:
        """Every active, result-affecting knob owned by stages 1..upto."""
        off = self._inactive()
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.metadata["affects_results"] and f.metadata["stage"] <= upto and f.name not in off
        }

    def stage_key(self, upto: int) -> str:
        """Cache key for artefacts produced by stages 1..upto.
        Parsed text -> stage_key(PARSING), chunks -> stage_key(CHUNKING),
        embeddings -> stage_key(EMBEDDING). Changing the reranker therefore
        never throws away your parsed PDFs or embeddings."""
        blob = json.dumps(self.result_knobs(upto), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]

    @property
    def slug(self) -> str:
        """Readable run id + full hash; safe as a Windows filename."""
        readable = "_".join([
            self.parser,
            f"{self.chunk_strategy}{self.chunk_size}o{self.chunk_overlap}",
            self.embedding_model.split("/")[-1],
            self.vector_store,
            self.retrieval_mode,
            "rerank" if self.reranker else "norerank",
            self.generator_model,
        ])
        readable = re.sub(r"[^A-Za-z0-9._-]+", "-", readable)  # ':' etc. are illegal on Windows
        return f"{readable}__{self.stage_key(GENERATION)}"

    # ---- experiment helpers -------------------------------------------------
    def with_(self, **changes) -> "PipelineConfig":
        """Return a variant. Re-runs validation."""
        return replace(self, **changes)

    def changed_fields(self, other: "PipelineConfig") -> dict[str, tuple]:
        """{knob: (self_value, other_value)} for every result-affecting difference.
        Ablation runners use this to refuse comparisons where more than one
        stage's knobs moved."""
        a, b = self.result_knobs(), other.result_knobs()
        return {k: (a.get(k), b.get(k)) for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


@dataclass(frozen=True)
class JudgeConfig:
    """The evaluation judge, held FIXED across every generator comparison.
    Deliberately separate from PipelineConfig so no ablation can vary it.
    Pick it at Stage 8, ideally from a different model family than your
    generator candidates (LLM judges tend to favour their own family)."""
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = ""          # left blank on purpose: choose it deliberately at Stage 8
    temperature: float = 0.0

    def require_set(self) -> None:
        if not self.model:
            raise ValueError("JudgeConfig.model is unset. Choose the judge before running generation evals.")


BASELINE = PipelineConfig()


if __name__ == "__main__":
    print("baseline slug:", BASELINE.slug)
    for name, stage in [("parse", PARSING), ("chunk", CHUNKING), ("embed", EMBEDDING)]:
        print(f"  {name:5s} cache key: {BASELINE.stage_key(stage)}")
    variant = BASELINE.with_(chunk_size=800)
    print("changed vs baseline:", BASELINE.changed_fields(variant))
    try:
        BASELINE.with_(parser="PyMuPDF")
    except ValueError as e:
        print("validation works ->", str(e).splitlines()[1].strip())
