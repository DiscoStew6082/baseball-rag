"""RAG retrieval layer."""

from baseball_rag.retrieval.chroma_store import get_store, retrieve
from baseball_rag.retrieval.strategies import (
    RetrievalStrategy,
    available_strategy_names,
    get_strategy,
)

__all__ = ["RetrievalStrategy", "available_strategy_names", "get_store", "get_strategy", "retrieve"]
