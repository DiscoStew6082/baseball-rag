"""FastAPI server for Baseball RAG."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Baseball RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    from baseball_rag.cli import answer as cli_answer
    result = cli_answer(req.question)
    return QueryResponse(answer=result, sources=[])