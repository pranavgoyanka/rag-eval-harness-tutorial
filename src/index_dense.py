"""
=============================================================================
PHASE 2 (part 2 of 3) — Dense index
=============================================================================

WHAT YOU'RE BUILDING
    artifacts/chunks.jsonl -> a searchable matrix of chunk vectors.

THE BIG IDEA OF THIS FILE
    You will implement dense search TWICE:

      1. numpy_search()  — six lines. This is what a vector database actually
         does at its core: one matrix multiply and a sort.
      2. build_faiss() / faiss_search() — the same thing via FAISS.

    Then the self-check asserts the two produce IDENTICAL results.

    Why bother with both? Because "vector database" gets talked about as if it
    were an exotic piece of infrastructure, and at this scale it demonstrably
    is not — it's `matrix @ query`. Knowing that, you can answer the question
    that actually matters in a design review: *when* does the numpy version
    stop being enough? (Answer below, in build_faiss's docstring.)

WHAT TO IMPLEMENT
    Three stubs: numpy_search, build_faiss, faiss_search.

RUN IT
    uv run python -m src.index_dense               # build the index
    uv run python -m src.index_dense --self-check  # numpy == faiss
    uv run python -m src.index_dense --search "how fast is the NB-1"

DEPENDS ON: Phase 1 output (artifacts/chunks.jsonl) and a working server.
=============================================================================
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from src.config import Config, load_config
from src.embed import Embedder
from src.ingest import Chunk, load_chunks

# FAISS is imported lazily. It is an optional accelerator here, not a hard
# requirement — if the wheel is unavailable for your Python version, every
# numpy path in this file still works and Phases 3/4 run unaffected.
try:
    import faiss  # type: ignore

    HAVE_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    HAVE_FAISS = False


# =============================================================================
# TODO(Phase 2) #2 — numpy_search
# =============================================================================


def numpy_search(qvec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return the top-k most similar rows of `matrix` to `qvec`.

    ---------------------------------------------------------------------------
    PRECONDITION
    ---------------------------------------------------------------------------
    Both `qvec` (shape (d,)) and every row of `matrix` (shape (n, d)) are
    ALREADY L2-normalized by embed.py. That is what makes the next line valid.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. scores = matrix @ qvec
         Shape (n, d) @ (d,) -> (n,). Because both sides are unit vectors, each
         entry IS the cosine similarity with that chunk. One BLAS call.

      2. Take the k largest. Two options:
           * np.argsort(-scores)[:k]                 — O(n log n), simple
           * np.argpartition(-scores, k)[:k], then sort those k — O(n), the
             right call when n is large
         At n=45 the difference is unmeasurable; use argsort and note in a
         comment that argpartition is the scaling answer.

      3. Return [(row_index, float(score)), ...] sorted by score DESCENDING.
         Cast to a Python float — np.float32 leaks into your JSON output later
         and json.dumps refuses to serialize it.

      4. Guard k > n: np.argsort on a short array won't error, it just returns
         fewer results, which is the behavior you want. But make sure your
         caller can handle a short list.

    ---------------------------------------------------------------------------
    THINK ABOUT
    ---------------------------------------------------------------------------
    This function reads the entire matrix on every query — an EXHAUSTIVE search.
    Cost is O(n*d). At n=45, d=768 that's ~35k multiply-adds, i.e. microseconds.
    At n=10,000,000 it is not, and that is the moment an ANN index earns its
    complexity. See build_faiss().
    """
    # TODO(Phase 2): implement per the recipe above.
    raise NotImplementedError("Phase 2: numpy_search")


# =============================================================================
# TODO(Phase 2) #3 — build_faiss
# =============================================================================


def build_faiss(matrix: np.ndarray):  # -> faiss.Index
    """Build a FAISS index over the chunk vectors.

    ---------------------------------------------------------------------------
    WHICH INDEX, AND WHY IT MATTERS
    ---------------------------------------------------------------------------
    FAISS offers a family of index types. The two axes are:

      EXACT vs APPROXIMATE
        * IndexFlatIP / IndexFlatL2 — brute force. Compares the query to every
          vector. Always returns the true top-k. O(n*d) per query.
        * IndexIVFFlat — partitions the space into `nlist` cells via k-means and
          only searches the `nprobe` nearest cells. Sublinear, but can MISS a
          true neighbor sitting just across a cell boundary.
        * IndexHNSWFlat — a navigable small-world graph; greedy descent from an
          entry point. Very fast, high recall, but the index is large in RAM and
          slow to build.

      The approximate ones trade RECALL for LATENCY. That trade is only worth
      making when latency is actually a problem.

      INNER PRODUCT vs L2
        * IndexFlatIP maximizes the dot product a.b
        * IndexFlatL2 minimizes squared euclidean distance ||a-b||^2
        Since embed.py normalized everything, these rank IDENTICALLY (expand
        ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b = 2 - 2a.b when both are unit
        length — so minimizing L2 distance is exactly maximizing the dot
        product). We use IP because the returned SCORE is then directly the
        cosine similarity, which is a number you can reason about and threshold.
        An L2 index would hand you back 2-2cos, which you'd have to convert.

    FOR THIS PROJECT: n=45. Use IndexFlatIP. It is exact, has zero parameters to
    tune, and needs no training step. Reaching for HNSW here would be a
    resume-driven decision, and the right answer in a design review is to say
    so — and to be able to say at what n you'd change your mind.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Guard: if not HAVE_FAISS, raise RuntimeError with a clear message.
      2. dim = matrix.shape[1]
      3. index = faiss.IndexFlatIP(dim)
      4. index.add(matrix)
         FAISS requires a C-contiguous float32 array. embed.py already returns
         float32; if you ever slice the matrix, use np.ascontiguousarray().
      5. Return the index.

      Note there is NO index.train() call — Flat indexes need no training. IVF
      would require index.train(matrix) before add(), and forgetting it is the
      classic first FAISS bug.
    """
    # TODO(Phase 2): implement per the recipe above.
    raise NotImplementedError("Phase 2: build_faiss")


# =============================================================================
# TODO(Phase 2) #4 — faiss_search
# =============================================================================


def faiss_search(index, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Query a FAISS index. Must return the same shape as numpy_search().

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. FAISS searches in BATCHES, so it wants a 2-D query array even for one
         query. Reshape (d,) -> (1, d) and make it float32:
             q = np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32)

      2. scores, idxs = index.search(q, k)
         Both come back with shape (1, k). Take row 0 of each.

      3. FAISS pads with index -1 when it has fewer than k results. DROP those,
         or you will index into your chunk list with -1 and silently return the
         LAST chunk in your corpus as a match. This is a genuinely nasty bug
         because it produces plausible-looking output.

      4. Return [(int(idx), float(score)), ...], already sorted descending
         (FAISS returns them sorted).
    """
    # TODO(Phase 2): implement per the recipe above.
    raise NotImplementedError("Phase 2: faiss_search")


# =============================================================================
# GIVEN: build / save / load
# =============================================================================


def build(cfg: Config) -> tuple[np.ndarray, list[Chunk]]:
    """Embed every chunk and persist the vector matrix + metadata."""
    chunks = load_chunks(cfg.path("chunks"))
    print(f"embedding {len(chunks)} chunks with {cfg.models.embedding} ...")

    emb = Embedder(cfg)
    t0 = time.perf_counter()
    try:
        # is_query=False -> the passage prefix. See embed.py on asymmetric models.
        matrix = emb.embed([c.text for c in chunks], is_query=False)
    finally:
        emb.close()
    dt = time.perf_counter() - t0

    vec_path = cfg.path("dense_vectors")
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, matrix)

    # The metadata file is what turns row indices back into chunk ids. The
    # matrix alone is useless; row 7 means nothing without this mapping.
    cfg.path("dense_meta").write_text(
        json.dumps(
            {
                "model": cfg.models.embedding,
                "dim": int(matrix.shape[1]),
                "count": int(matrix.shape[0]),
                "chunk_ids": [c.chunk_id for c in chunks],
            },
            indent=2,
        )
    )

    print(f"embedded in {dt:.1f}s -> matrix {matrix.shape} "
          f"({matrix.nbytes / 1024:.0f} KiB)")

    if HAVE_FAISS:
        try:
            index = build_faiss(matrix)
            faiss.write_index(index, str(cfg.path("faiss_index")))
            print(f"faiss index written, ntotal={index.ntotal}")
        except NotImplementedError:
            print("faiss index skipped (build_faiss not implemented yet)")
    else:
        print("faiss not installed — numpy path only (this is fine)")

    return matrix, chunks


def load_dense(cfg: Config) -> tuple[np.ndarray, list[str]]:
    """Load the vector matrix and its chunk_id mapping.

    Asserts the two agree. A matrix/metadata mismatch means one of them is stale
    and every search result would be misattributed — fail here, loudly, rather
    than serving wrong citations for the rest of the project.
    """
    vec_path, meta_path = cfg.path("dense_vectors"), cfg.path("dense_meta")
    if not vec_path.exists():
        raise FileNotFoundError(
            f"{vec_path} not found. Run `python -m src.index_dense` first."
        )
    matrix = np.load(vec_path)
    meta = json.loads(meta_path.read_text())
    assert matrix.shape[0] == meta["count"] == len(meta["chunk_ids"]), (
        f"stale index: matrix has {matrix.shape[0]} rows, metadata claims "
        f"{meta['count']}. Rebuild with `python -m src.index_dense`."
    )
    if meta["model"] != cfg.models.embedding:
        raise ValueError(
            f"index was built with {meta['model']!r} but config says "
            f"{cfg.models.embedding!r}. Vectors from different models are not "
            "comparable. Rebuild the index."
        )
    return matrix, meta["chunk_ids"]


def load_faiss(cfg: Config):
    """Load the persisted FAISS index, or None if unavailable."""
    p = cfg.path("faiss_index")
    if not HAVE_FAISS or not p.exists():
        return None
    return faiss.read_index(str(p))


# =============================================================================
# GIVEN: search entry point used by Phase 3
# =============================================================================


def search(cfg: Config, query: str, k: int) -> list[tuple[str, float]]:
    """Dense retrieval: query string -> [(chunk_id, cosine_score), ...].

    Prefers FAISS when present, falls back to numpy. Phase 3 calls this.
    """
    matrix, chunk_ids = load_dense(cfg)
    emb = Embedder(cfg)
    try:
        qvec = emb.embed_query(query)
    finally:
        emb.close()

    index = load_faiss(cfg)
    if index is not None:
        try:
            hits = faiss_search(index, qvec, k)
        except NotImplementedError:
            hits = numpy_search(qvec, matrix, k)
    else:
        hits = numpy_search(qvec, matrix, k)

    return [(chunk_ids[i], s) for i, s in hits]


# =============================================================================
# GIVEN: self-check
# =============================================================================


def self_check(cfg: Config) -> None:
    matrix, chunk_ids = load_dense(cfg)
    chunks = {c.chunk_id: c for c in load_chunks(cfg.path("chunks"))}

    # --- 1. reload integrity ------------------------------------------------
    assert matrix.shape[0] == len(chunk_ids), "matrix/metadata length mismatch"
    assert matrix.dtype == np.float32, f"expected float32, got {matrix.dtype}"
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), (
        "stored vectors are not unit length — l2_normalize is not being applied"
    )
    print(f"  [1/3] ok  {matrix.shape[0]} unit-length vectors reloaded, "
          f"dim={matrix.shape[1]}")

    # --- 2. numpy and faiss agree ------------------------------------------
    index = load_faiss(cfg)
    if index is None:
        print("  [2/3] SKIP  faiss unavailable or index not built")
    else:
        assert index.ntotal == matrix.shape[0], (
            f"faiss ntotal={index.ntotal} != {matrix.shape[0]} rows. "
            "The index is stale; rebuild it."
        )
        rng = np.random.default_rng(42)
        k = min(5, matrix.shape[0])
        for trial in range(5):
            q = rng.normal(size=matrix.shape[1]).astype(np.float32)
            q /= np.linalg.norm(q)
            a = numpy_search(q, matrix, k)
            b = faiss_search(index, q, k)
            assert [i for i, _ in a] == [i for i, _ in b], (
                f"trial {trial}: numpy returned {[i for i, _ in a]} but faiss "
                f"returned {[i for i, _ in b]}"
            )
            for (_, sa), (_, sb) in zip(a, b):
                assert abs(sa - sb) < 1e-5, f"score mismatch {sa} vs {sb}"
        print(f"  [2/3] ok  numpy and faiss agree on top-{k} for 5 random queries "
              f"(ntotal={index.ntotal})")

    # --- 3. a known query retrieves the right document ---------------------
    q = "What is the top speed of the NB-1 Courier?"
    hits = search(cfg, q, 5)
    top_docs = [chunks[cid].doc for cid, _ in hits]
    assert "product-nb1.md" in top_docs, (
        f"{q!r} did not retrieve product-nb1.md in top-5; got {top_docs}"
    )
    print(f"  [3/3] ok  {q!r} -> {hits[0][0]} (score {hits[0][1]:.3f})")

    print("\nPHASE 2 (dense) SELF-CHECK PASSED")


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    cfg = load_config()
    try:
        if "--self-check" in sys.argv:
            self_check(cfg)
        elif "--search" in sys.argv:
            query = sys.argv[sys.argv.index("--search") + 1]
            chunks = {c.chunk_id: c for c in load_chunks(cfg.path("chunks"))}
            print(f"query: {query!r}\n")
            for rank, (cid, score) in enumerate(search(cfg, query, 5), start=1):
                c = chunks[cid]
                print(f"{rank}. {score:.4f}  {cid}  ({' > '.join(c.heading_path)})")
                print(f"     {c.body[:110].replace(chr(10), ' ')} ...")
        else:
            build(cfg)
            print("\nNext: uv run python -m src.index_dense --self-check")
    except FileNotFoundError as exc:
        print(f"Missing prerequisite: {exc}")
        return 1
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}")
        print("\nPhase 2 (dense) stubs, in order:")
        print("  1. l2_normalize()  src/embed.py")
        print("  2. numpy_search()  src/index_dense.py")
        print("  3. build_faiss()   src/index_dense.py")
        print("  4. faiss_search()  src/index_dense.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
