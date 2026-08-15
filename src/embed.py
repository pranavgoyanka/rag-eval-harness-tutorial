"""
=============================================================================
PHASE 2 (part 1 of 3) — Embeddings
=============================================================================

WHAT YOU'RE BUILDING
    text -> a fixed-length vector of floats, via the server's /v1/embeddings.

WHAT AN EMBEDDING ACTUALLY IS
    A function that maps text to a point in R^d (here d=768) such that texts
    with similar MEANING land near each other, regardless of shared words. That
    "regardless of shared words" is the whole point, and also the whole weakness:
      * strength — "How fast does the courier robot go?" matches a chunk saying
        "Top speed | 24 km/h" with no words in common.
      * weakness — "NB-1" and "NB-2", or "E204" and "E317", are near-identical
        strings that mean completely different things, and the embedding cannot
        reliably tell them apart. That is why Phase 2 also builds a BM25 index
        (index_sparse.py) and Phase 3 fuses the two.

WHAT TO IMPLEMENT
    One stub: l2_normalize(). Everything else is plumbing and is written.

RUN IT
    uv run python -m src.embed --self-check

REQUIRES A WORKING SERVER. Run `python -m src.config --check` first.
=============================================================================
"""

from __future__ import annotations

import hashlib
import sys

import httpx
import numpy as np

from src.config import Config, load_config

# =============================================================================
# TODO(Phase 2) #1 — l2_normalize
# =============================================================================


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Scale each row of `mat` to unit length.

    ---------------------------------------------------------------------------
    THE CONCEPT: why normalizing lets you replace cosine with a dot product
    ---------------------------------------------------------------------------
    Similarity between two vectors is measured by the cosine of the angle
    between them:

        cosine(a, b) = (a . b) / (||a|| * ||b||)          in [-1, 1]

    The magnitudes in the denominator are noise for our purposes — a longer
    chunk may produce a longer vector without being more or less relevant. We
    only care about direction.

    So: divide every vector by its own L2 norm, ||v|| = sqrt(sum(v_i^2)). Now
    every vector has length 1, both denominators are 1, and

        cosine(a, b) == a . b

    A plain inner product. This is not just tidiness — it is what lets
    index_dense.py use FAISS's `IndexFlatIP` (Inner Product) rather than an L2
    index, and it is why the whole dense search collapses into a single matrix
    multiply.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Accept either a 1-D vector (d,) or a 2-D matrix (n, d). Reshaping a
         1-D input to (1, d) at the top and squeezing at the end is the tidiest
         way; searching with a single query vector is the common case.
      2. norms = np.linalg.norm(mat, axis=1, keepdims=True)
         `keepdims=True` is what makes the division broadcast correctly:
         (n, d) / (n, 1) works, (n, d) / (n,) raises.
      3. GUARD AGAINST ZERO. A zero vector (an empty string, or a server hiccup)
         gives a norm of 0 and produces NaNs, which then poison every score in
         the index and surface much later as a baffling "everything ranks 0".
         Replace zero norms with 1.0 before dividing.
      4. Return float32. FAISS requires float32; passing float64 raises a type
         error at index.add() time.

    Returns: array of the same shape as the input, each row of length 1.
    """
    # TODO(Phase 2): implement per the recipe above.
    raise NotImplementedError("Phase 2: l2_normalize")


# =============================================================================
# GIVEN: the embedding client
# =============================================================================


class Embedder:
    """Thin client over the server's /v1/embeddings endpoint, with a disk cache.

    Why a cache: embedding is deterministic and the corpus barely changes, but
    you will rebuild the index dozens of times while iterating on chunking. The
    cache makes every rebuild after the first one essentially free.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = cfg.models.embedding
        self.dim = cfg.models.embedding_dim
        self.cache_dir = cfg.path("embed_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            base_url=cfg.server.base_url,
            timeout=cfg.server.timeout,
            headers={"Authorization": f"Bearer {cfg.server.api_key}"},
        )

    # -- cache ------------------------------------------------------------
    def _cache_key(self, text: str) -> str:
        # The model name is part of the key: switching embedding models must
        # invalidate the cache, or you end up with an index containing vectors
        # from two different vector spaces. That failure is silent and awful.
        h = hashlib.sha256(f"{self.model}\x00{text}".encode()).hexdigest()
        return h[:32]

    def _cache_get(self, text: str) -> np.ndarray | None:
        p = self.cache_dir / f"{self._cache_key(text)}.npy"
        return np.load(p) if p.exists() else None

    def _cache_put(self, text: str, vec: np.ndarray) -> None:
        np.save(self.cache_dir / f"{self._cache_key(text)}.npy", vec)

    # -- HTTP -------------------------------------------------------------
    def _post_batch(self, texts: list[str]) -> list[np.ndarray]:
        try:
            resp = self._client.post(
                "/embeddings", json={"model": self.model, "input": texts}
            )
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Could not reach {self.cfg.server.base_url}/embeddings: {exc}\n"
                "Run `python -m src.config --check` to diagnose."
            ) from exc

        if resp.status_code == 501:
            raise RuntimeError(
                "Server returned 501: it was not started with --embeddings, or "
                f"the model {self.model!r} is not an embedding model.\n"
                "Run `python -m src.config --check` — it prints the llama-swap "
                "config snippet you need."
            )
        resp.raise_for_status()

        data = resp.json()["data"]
        # The API does not guarantee ordering, but it does return an `index`
        # field. Sort by it rather than trusting the wire order.
        data.sort(key=lambda d: d["index"])
        return [np.asarray(d["embedding"], dtype=np.float32) for d in data]

    # -- public API -------------------------------------------------------
    def embed(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        """Embed a list of texts. Returns a normalized (n, dim) float32 array.

        ---------------------------------------------------------------------
        ASYMMETRIC MODELS AND PREFIXES — a real footgun
        ---------------------------------------------------------------------
        Modern retrieval embedders (nomic, BGE, E5, GTE) are trained with an
        instruction prefix that tells the model which SIDE of the retrieval pair
        a text is on:

            queries   -> "search_query: How fast is the NB-1?"
            documents -> "search_document: | Top speed | 24 km/h |"

        This exists because the two sides look nothing alike — a query is a
        short question, a document is a long declarative passage — and training
        them into the same region of space requires telling the model which is
        which.

        Get the prefixes backwards, or omit them, and NOTHING ERRORS. You just
        lose a chunk of recall, and you will spend an afternoon blaming your
        chunker. Hence the explicit `is_query` argument: this function refuses
        to guess.

        If you switch to a symmetric model (e.g. all-MiniLM-L6-v2), set both
        prefixes to "" in config.yaml.
        """
        prefix = (
            self.cfg.embedding.query_prefix
            if is_query
            else self.cfg.embedding.passage_prefix
        )
        prefixed = [prefix + t for t in texts]

        out: list[np.ndarray | None] = [None] * len(prefixed)
        pending: list[int] = []
        for i, t in enumerate(prefixed):
            cached = self._cache_get(t)
            if cached is not None:
                out[i] = cached
            else:
                pending.append(i)

        bs = self.cfg.embedding.batch_size
        for start in range(0, len(pending), bs):
            # Serial on purpose: the backing llama-server runs with --parallel 1,
            # so concurrent requests would queue anyway and only make the failure
            # modes harder to read.
            idxs = pending[start : start + bs]
            vecs = self._post_batch([prefixed[i] for i in idxs])
            for i, v in zip(idxs, vecs):
                self._cache_put(prefixed[i], v)
                out[i] = v
            if len(pending) > bs:
                print(f"  embedded {min(start + bs, len(pending))}/{len(pending)}")

        mat = np.vstack([v for v in out if v is not None])
        if mat.shape[1] != self.dim:
            raise ValueError(
                f"Server returned {mat.shape[1]}-dim vectors but config.yaml says "
                f"{self.dim}. Fix models.embedding_dim, then delete "
                f"{self.cache_dir} and re-embed."
            )
        return l2_normalize(mat)

    def embed_query(self, text: str) -> np.ndarray:
        """Convenience: embed one query, return a 1-D (dim,) vector."""
        return self.embed([text], is_query=True)[0]

    def close(self) -> None:
        self._client.close()


# =============================================================================
# GIVEN: self-check
# =============================================================================


def self_check(cfg: Config) -> None:
    # --- 1. l2_normalize is correct (no server needed) ----------------------
    rng = np.random.default_rng(0)
    m = rng.normal(size=(5, 8)).astype(np.float32)
    n = l2_normalize(m)
    assert n.shape == m.shape, f"shape changed: {m.shape} -> {n.shape}"
    assert n.dtype == np.float32, f"expected float32, got {n.dtype}"
    lengths = np.linalg.norm(n, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5), f"rows not unit length: {lengths}"
    print("  [1/4] ok  l2_normalize produces unit-length float32 rows")

    # --- 2. zero-vector guard ----------------------------------------------
    z = np.zeros((2, 8), dtype=np.float32)
    zn = l2_normalize(z)
    assert not np.isnan(zn).any(), "zero vector produced NaNs — guard the divide"
    print("  [2/4] ok  zero vectors do not produce NaNs")

    # --- 3. dot product of normalized vectors == cosine ---------------------
    a, b = m[0], m[1]
    manual_cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    dot_of_norm = float(n[0] @ n[1])
    assert abs(manual_cos - dot_of_norm) < 1e-5, (
        f"dot of normalized ({dot_of_norm}) != cosine ({manual_cos})"
    )
    print("  [3/4] ok  dot product of normalized vectors equals cosine")

    # --- 4. server round-trip ----------------------------------------------
    emb = Embedder(cfg)
    try:
        v = emb.embed_query("How fast is the NB-1 Courier?")
        assert v.shape == (cfg.models.embedding_dim,), f"bad query shape {v.shape}"
        # Sanity: a related passage should score higher than an unrelated one.
        pas = emb.embed(
            [
                "NB-1 Courier general specifications: Top speed | 24 km/h",
                "Volume discounts apply to orders above 50 units.",
            ],
            is_query=False,
        )
        related, unrelated = float(v @ pas[0]), float(v @ pas[1])
        assert related > unrelated, (
            f"related passage scored {related:.3f} <= unrelated {unrelated:.3f}. "
            "Check your query_prefix/passage_prefix in config.yaml."
        )
        print(f"  [4/4] ok  server round-trip; related={related:.3f} > "
              f"unrelated={unrelated:.3f}")
    finally:
        emb.close()

    print("\nPHASE 2 (embed) SELF-CHECK PASSED")


if __name__ == "__main__":
    cfg = load_config()
    try:
        if "--self-check" in sys.argv:
            self_check(cfg)
        else:
            e = Embedder(cfg)
            q = sys.argv[1] if len(sys.argv) > 1 else "hello"
            v = e.embed_query(q)
            print(f"{q!r} -> shape={v.shape} norm={np.linalg.norm(v):.4f}")
            print(f"first 8 dims: {v[:8]}")
            e.close()
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}")
        print("\nFill in l2_normalize() in src/embed.py — the docstring has the recipe.")
        raise SystemExit(1) from None
