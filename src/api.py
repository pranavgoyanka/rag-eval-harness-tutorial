"""
=============================================================================
PHASE 7 — HTTP API + browser UI
=============================================================================

WHAT YOU'RE BUILDING
    POST /query  -> the full QueryResult as JSON
    GET  /health -> readiness
    GET  /       -> a single-page UI

THIS FILE HAS NO STUBS, AND THAT IS DELIBERATE.
    Every other phase left the conceptually interesting function blank. This one
    doesn't, because there is no new RAG concept here — it is a web server in
    front of code you already wrote. Making you retype FastAPI boilerplate would
    teach you FastAPI, not retrieval.

    There ARE two engineering decisions worth reading, both commented below:
    holding warm clients for the process lifetime, and deciding what NOT to put
    in the response.

RUN IT
    uv run uvicorn src.api:app --port 8000
    open http://localhost:8000

    curl -s localhost:8000/query -H 'content-type: application/json' \\
         -d '{"question":"What is the NB-1 payload capacity?"}' | jq

DEPENDS ON: Phases 1-5.
=============================================================================
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import ROOT, load_config
from src.embed import Embedder
from src.generate import Generator
from src.pipeline import RETRIEVAL_MODES, query

cfg = load_config()

# Module-level state, populated in the lifespan handler below.
_clients: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the model clients once, at startup, and keep them.

    WHY THIS MATTERS: llama-swap loads a model on first use. If you constructed
    a Generator per request, the first request after any idle period would pay
    seconds of model load, and your "why is it sometimes slow?" investigation
    would end here. Warm once, hold the connection, reuse it.

    The same argument applies to Embedder, which additionally holds the on-disk
    embedding cache.
    """
    gen = Generator(cfg)
    emb = Embedder(cfg)
    try:
        gen.warmup()
    except Exception as exc:  # noqa: BLE001 — never block startup on a cold model
        print(f"warmup failed (continuing): {exc}")
    _clients["gen"], _clients["emb"] = gen, emb
    yield
    gen.close()
    emb.close()


app = FastAPI(title="RAG + Eval Harness", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    verify: bool | None = None


class SourceOut(BaseModel):
    chunk_id: str
    source: str
    heading_path: list[str]
    text: str
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    cited: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    citations: list[str]
    invalid_citations: list[int]
    sources: list[SourceOut]
    timings: dict[str, float]
    prompt_tokens: int
    completion_tokens: int
    iterations: int | None = None
    faithfulness: float | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chat_model": cfg.models.chat,
        "embedding_model": cfg.models.embedding,
        "warm": "gen" in _clients,
    }


@app.post("/query", response_model=QueryResponse)
def post_query(req: QueryRequest) -> QueryResponse:
    mode = req.mode if req.mode in RETRIEVAL_MODES else "hybrid"
    res = query(
        cfg,
        req.question,
        generator=_clients.get("gen"),  # type: ignore[arg-type]
        embedder=_clients.get("emb"),  # type: ignore[arg-type]
        mode=mode,
        verify=req.verify,
    )

    # WHAT GOES IN THE RESPONSE, AND WHAT DOESN'T.
    #
    # We return the full chunk text for every retrieved source, not just the
    # cited ones, because this is a DEBUGGING tool as much as a product: seeing
    # what was retrieved and NOT cited is how you diagnose a bad answer, and it
    # is the same view the ablation in Phase 6 is built on.
    #
    # A user-facing product would return citations only. If you ever point this
    # at a corpus with access controls, note that this endpoint hands the caller
    # raw chunk text for anything the retriever surfaced — which is a data
    # leak waiting to happen, and the reason real systems filter by permission
    # BEFORE retrieval rather than after.
    cited = set(res.citations)
    return QueryResponse(
        question=res.question,
        answer=res.answer,
        abstained=res.abstained,
        citations=res.citations,
        invalid_citations=res.invalid_citations,
        sources=[
            SourceOut(
                chunk_id=s.chunk_id,
                source=s.chunk.source,
                heading_path=s.chunk.heading_path,
                text=s.chunk.body,
                rrf_score=s.rrf_score,
                dense_rank=s.dense_rank,
                sparse_rank=s.sparse_rank,
                cited=s.chunk_id in cited,
            )
            for s in res.sources
        ],
        timings=res.timings,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        iterations=res.iterations,
        faithfulness=res.faithfulness,
    )


STATIC = ROOT / "static"
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")
