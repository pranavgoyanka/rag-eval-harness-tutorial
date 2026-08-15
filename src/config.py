"""
Phase 0/1 — Configuration loading and environment preflight.

THIS FILE IS FULLY IMPLEMENTED. There is nothing to fill in here. Read it once
to learn the shape of the config, then run the preflight check:

    uv run python -m src.config --check

That command is the gate for the whole project. It must print `chat: ok` and
`embeddings: ok` before Phase 2 will work. Phase 1 (chunking) needs neither and
can be done offline.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of src/. Everything in config.yaml uses paths relative to
# this, so the CLIs behave the same regardless of your working directory.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


# =============================================================================
# Dataclasses
#
# Why not just pass the raw dict around? Because `cfg.retrieval.top_k` fails
# loudly on a typo, while `cfg["retrieval"]["topk"]` fails at 2am in Phase 6.
# =============================================================================


@dataclass
class ServerCfg:
    base_url: str
    api_key: str
    timeout: int


@dataclass
class ModelsCfg:
    chat: str
    embedding: str
    embedding_dim: int


@dataclass
class ChunkingCfg:
    corpus_dir: str
    heading_levels: list[int]
    min_tokens: int
    max_tokens: int
    prepend_heading_path: bool


@dataclass
class PathsCfg:
    artifacts_dir: str
    chunks: str
    dense_vectors: str
    dense_meta: str
    faiss_index: str
    bm25_index: str
    embed_cache: str


@dataclass
class EmbeddingCfg:
    batch_size: int
    query_prefix: str
    passage_prefix: str


@dataclass
class RetrievalCfg:
    top_k: int
    overfetch: int
    rrf_k: int
    min_score: float


@dataclass
class GenerationCfg:
    temperature: float
    max_tokens: int
    context_token_budget: int
    enable_thinking: bool


@dataclass
class VerificationCfg:
    """Phase 5. Parsed now so the config shape is stable, but NOTHING reads
    these fields until you write src/self_correct.py. Don't be confused by the
    dangling values — they are a promise, not dead code."""

    enabled: bool
    sim_threshold: float
    jaccard_threshold: float
    max_iterations: int


@dataclass
class EvalCfg:
    """Phase 6. Same deal as VerificationCfg — declared early, used later."""

    golden_set: str
    output: str


@dataclass
class Config:
    server: ServerCfg
    models: ModelsCfg
    chunking: ChunkingCfg
    paths: PathsCfg
    embedding: EmbeddingCfg
    retrieval: RetrievalCfg
    generation: GenerationCfg
    verification: VerificationCfg
    eval: EvalCfg
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def path(self, key: str) -> Path:
        """Resolve a `paths.*` entry to an absolute Path under the repo root."""
        return ROOT / getattr(self.paths, key)


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    """Parse config.yaml into the dataclass tree above.

    Environment overrides (useful in CI or when the server moves):
      RAG_BASE_URL   -> server.base_url
      RAG_CHAT_MODEL -> models.chat
      RAG_EMBED_MODEL-> models.embedding
    """
    raw = yaml.safe_load(Path(path).read_text())

    server = ServerCfg(**raw["server"])
    models = ModelsCfg(**raw["models"])

    if env := os.environ.get("RAG_BASE_URL"):
        server.base_url = env
    if env := os.environ.get("RAG_CHAT_MODEL"):
        models.chat = env
    if env := os.environ.get("RAG_EMBED_MODEL"):
        models.embedding = env

    return Config(
        server=server,
        models=models,
        chunking=ChunkingCfg(**raw["chunking"]),
        paths=PathsCfg(**raw["paths"]),
        embedding=EmbeddingCfg(**raw["embedding"]),
        retrieval=RetrievalCfg(**raw["retrieval"]),
        generation=GenerationCfg(**raw["generation"]),
        verification=VerificationCfg(**raw["verification"]),
        eval=EvalCfg(**raw["eval"]),
        _raw=raw,
    )


# =============================================================================
# Preflight check
# =============================================================================

_EMBED_SETUP_HELP = """
  Your server does not expose /v1/embeddings yet. This is expected on a first
  run — llama-server only serves embeddings when launched with --embeddings,
  and a chat model cannot serve them at all. You need to register a SECOND
  model on the llama-swap router.

  Add this to your llama-swap config (models: section) and reload the router:

      nomic-embed-text:
        cmd: >
          /path/to/llama.cpp/build/bin/llama-server
          --port ${PORT}
          --alias nomic-embed-text
          --hf-repo nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0
          --embeddings
          --pooling mean
          --ctx-size 8192
          --n-gpu-layers 99

  Notes:
    * --embeddings is what turns on the /v1/embeddings route.
    * --pooling mean turns per-token vectors into one vector per input. Without
      it you get a shape you cannot index.
    * Your router reported max_instances: 4, so this model can stay loaded
      alongside qwen3.8-27b; you do not have to swap between them.
    * nomic-embed-text-v1.5 is 768-dimensional. If you pick a different model,
      update models.embedding_dim in config.yaml to match.
"""


def _check(cfg: Config) -> int:
    """Probe the server and report what works. Returns a process exit code."""
    import httpx

    print("Resolved configuration")
    print(f"  base_url        {cfg.server.base_url}")
    print(f"  chat model      {cfg.models.chat}")
    print(f"  embedding model {cfg.models.embedding} (dim={cfg.models.embedding_dim})")
    print(f"  corpus          {ROOT / cfg.chunking.corpus_dir}")
    print()

    failures = 0
    client = httpx.Client(
        base_url=cfg.server.base_url,
        timeout=cfg.server.timeout,
        headers={"Authorization": f"Bearer {cfg.server.api_key}"},
    )

    # --- 1. Can we reach the server at all? --------------------------------
    try:
        resp = client.get("/models")
        resp.raise_for_status()
        ids = [m["id"] for m in resp.json().get("data", [])]
        print(f"models: ok ({len(ids)} available)")
        for i in ids:
            print(f"  - {i}")
    except Exception as exc:  # noqa: BLE001 - preflight wants every failure
        print(f"models: FAILED — {type(exc).__name__}: {exc}")
        print(
            "\n  Cannot reach the server. If you are running inside a container,\n"
            "  remember that 'localhost' is the container, not your host machine.\n"
            "  Try:  export RAG_BASE_URL=http://host.docker.internal:8101/v1\n"
        )
        return 1
    print()

    if cfg.models.chat not in ids:
        print(f"WARNING: chat model {cfg.models.chat!r} is not in the list above.")
    if cfg.models.embedding not in ids:
        print(f"WARNING: embedding model {cfg.models.embedding!r} is not in the list above.")
    print()

    # --- 2. Chat completions ----------------------------------------------
    try:
        body: dict[str, Any] = {
            "model": cfg.models.chat,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        if not cfg.generation.enable_thinking:
            # See config.yaml: qwen3.8-27b reasons by default.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        resp = client.post("/chat/completions", json=body)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        print(f"chat: ok — model replied {text.strip()[:40]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"chat: FAILED — {type(exc).__name__}: {exc}")
        failures += 1
    print()

    # --- 3. Embeddings -----------------------------------------------------
    try:
        resp = client.post(
            "/embeddings",
            json={"model": cfg.models.embedding, "input": "hello"},
        )
        if resp.status_code == 501:
            print("embeddings: FAILED — server replied 501 (not supported)")
            print(_EMBED_SETUP_HELP)
            failures += 1
        else:
            resp.raise_for_status()
            dim = len(resp.json()["data"][0]["embedding"])
            if dim != cfg.models.embedding_dim:
                print(
                    f"embeddings: DIMENSION MISMATCH — server returned {dim}, "
                    f"config says {cfg.models.embedding_dim}."
                )
                print("  Fix models.embedding_dim in config.yaml before indexing.")
                failures += 1
            else:
                print(f"embeddings: ok (dim={dim})")
    except Exception as exc:  # noqa: BLE001
        print(f"embeddings: FAILED — {type(exc).__name__}: {exc}")
        print(_EMBED_SETUP_HELP)
        failures += 1

    client.close()
    print()
    print("PREFLIGHT PASSED" if failures == 0 else f"PREFLIGHT FAILED ({failures} problem(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    cfg = load_config()
    if "--check" in sys.argv:
        raise SystemExit(_check(cfg))
    print(cfg)
    print("\nTip: run `python -m src.config --check` to probe the server.")
