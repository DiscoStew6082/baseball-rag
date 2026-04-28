"""FastAPI server for Baseball RAG."""

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Baseball RAG API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    intent: str
    sources: list[dict[str, Any]]
    warnings: list[str]
    unsupported: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    from baseball_rag.service import answer

    result = answer(req.question)
    return QueryResponse(**result.to_dict())


@app.get("/sources")
def sources():
    """Return dataset provenance used by DuckDB-backed answers."""
    from baseball_rag.provenance import load_data_manifest

    return load_data_manifest()
