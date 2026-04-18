"""Pytest configuration — mock the LM Studio embedder so tests run without a live server."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def _fake_embed(text: str) -> list[float]:
    """Deterministic 384-dim embedding derived from text content.

    Uses a hash of the input to seed a RNG so every call with the same
    string returns the same vector — but semantically different strings get
    unrelated vectors.  This is enough for testing ChromaDB mechanics
    (ID handling, metadata, scoring order) without needing LM Studio.
    """
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    rng = np.random.default_rng(h)
    return rng.standard_normal(384).tolist()


@pytest.fixture(autouse=True)
def _mock_embedder():
    """Patch baseball_rag.embedder so CI runs without a live LM Studio server."""
    with (
        patch("baseball_rag.embedder.embed", side_effect=_fake_embed),
        # Also patch the module-level alias inside chroma_store
        patch("baseball_rag.retrieval.chroma_store._embedder.embed", side_effect=_fake_embed),
    ):
        yield


# ----------------------------------------------------------------------
# Helper to re-index with mock embeddings — used by tests that need a fresh store.
# ----------------------------------------------------------------------
def rebuild_index(persist_dir: str) -> None:
    """Build the ChromaDB index using fake embeddings (no LM Studio required)."""
    from pathlib import Path

    from baseball_rag.corpus.ingest import build_index

    build_index(Path(persist_dir))


@pytest.fixture
def chroma_db_dir(tmp_path: pytest.TempPathFactory) -> "Path":
    """Build a fresh index with fake embeddings in a temp directory."""
    rebuild_index(str(tmp_path / "chroma"))
    return tmp_path / "chroma"
