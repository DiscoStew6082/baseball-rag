"""Retrieval strategy objects for benchmarkable search tactics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from baseball_rag.retrieval.chroma_store import RetrievedChunk, retrieve

RetrieveFn = Callable[..., list[RetrievedChunk]]


class RetrievalStrategy(Protocol):
    """A benchmarkable retrieval tactic."""

    @property
    def name(self) -> str:
        """Stable strategy name for reporting."""
        ...

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        persist_dir: Path | None = None,
        player_name: str | None = None,
        player_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a routed query."""


@dataclass(frozen=True)
class SemanticChromaStrategy:
    """Current vector search only."""

    retrieve_fn: RetrieveFn = retrieve
    name: str = "semantic_chroma"

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        persist_dir: Path | None = None,
        player_name: str | None = None,
        player_id: str | None = None,
    ) -> list[RetrievedChunk]:
        search_query = player_name or query
        return _call_retrieve(
            self.retrieve_fn,
            search_query,
            top_k=top_k,
            persist_dir=persist_dir,
            where=None,
        )


@dataclass(frozen=True)
class ExactPlayerIdStrategy:
    """Resolve a player ID first, then use a Chroma metadata filter."""

    retrieve_fn: RetrieveFn = retrieve
    name: str = "exact_player_id"

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        persist_dir: Path | None = None,
        player_name: str | None = None,
        player_id: str | None = None,
    ) -> list[RetrievedChunk]:
        if not player_id:
            return []
        search_query = player_name or query
        return _call_retrieve(
            self.retrieve_fn,
            search_query,
            top_k=1,
            persist_dir=persist_dir,
            where={"player_id": player_id},
        )


@dataclass(frozen=True)
class HybridPlayerBioStrategy:
    """Exact player metadata lookup first, semantic player search fallback."""

    retrieve_fn: RetrieveFn = retrieve
    name: str = "hybrid_player_bio"

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        persist_dir: Path | None = None,
        player_name: str | None = None,
        player_id: str | None = None,
    ) -> list[RetrievedChunk]:
        exact = ExactPlayerIdStrategy(retrieve_fn=self.retrieve_fn).retrieve(
            query,
            top_k=top_k,
            persist_dir=persist_dir,
            player_name=player_name,
            player_id=player_id,
        )
        if exact:
            return exact
        return SemanticChromaStrategy(retrieve_fn=self.retrieve_fn).retrieve(
            query,
            top_k=top_k,
            persist_dir=persist_dir,
            player_name=player_name,
            player_id=player_id,
        )


def available_strategy_names() -> list[str]:
    """Return strategy names in stable benchmark display order."""
    return ["semantic_chroma", "exact_player_id", "hybrid_player_bio"]


def get_strategy(name: str, *, retrieve_fn: RetrieveFn = retrieve) -> RetrievalStrategy:
    """Build a retrieval strategy by name."""
    if name == "semantic_chroma":
        return SemanticChromaStrategy(retrieve_fn=retrieve_fn)
    if name == "exact_player_id":
        return ExactPlayerIdStrategy(retrieve_fn=retrieve_fn)
    if name == "hybrid_player_bio":
        return HybridPlayerBioStrategy(retrieve_fn=retrieve_fn)
    choices = ", ".join(available_strategy_names())
    raise ValueError(f"unknown retrieval strategy {name!r}; choose one of: {choices}")


def _call_retrieve(
    retrieve_fn: RetrieveFn,
    query: str,
    *,
    top_k: int,
    persist_dir: Path | None,
    where: dict | None,
) -> list[RetrievedChunk]:
    if persist_dir is None:
        return retrieve_fn(query, top_k=top_k, where=where)
    return retrieve_fn(query, top_k=top_k, persist_dir=persist_dir, where=where)
