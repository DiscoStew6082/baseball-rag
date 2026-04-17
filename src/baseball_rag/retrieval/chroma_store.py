"""Persistent vector store backed by ChromaDB."""
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "baseball_corpus"


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk."""
    text: str
    source: str
    title: str
    score: float


def get_store(persist_dir: Path) -> chromadb.Collection:
    """Open or create the baseball corpus collection."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, top_k: int = 3, persist_dir: Path | None = None) -> list[RetrievedChunk]:
    """Embed query and retrieve top-K relevant chunks.

    Args:
        query: Natural language search query
        top_k: Number of results to return
        persist_dir: Where the ChromaDB is stored. Defaults to data/chroma_db/.

    Returns:
        List of RetrievedChunk objects sorted by relevance (best first).
    """
    if persist_dir is None:
        from baseball_rag.db.lahman import DATA_DIR
        persist_dir = DATA_DIR

    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = get_store(persist_dir)
    query_embedding = embed_fn([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if not results["ids"]:
        return []
    # ChromaDB type stubs over-approximate None; we already verified ids is non-empty above
    for i in range(len(results["ids"][0])):
        doc = results["documents"][0][i]  # type: ignore[index]
        meta = results["metadatas"][0][i]  # type: ignore[index]
        dist = results["distances"][0][i]  # type: ignore[index]
        # ChromaDB L2 distance — lower is better; convert to a 0-1 "score"
        score = max(0.0, 1.0 - dist / 2.0)
        chunks.append(RetrievedChunk(
            text=doc,
            source=str(meta.get("source", "")),
            title=str(meta.get("title", "")),
            score=score,
        ))

    return chunks
