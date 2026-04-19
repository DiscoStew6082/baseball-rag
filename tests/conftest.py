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
# Auto-apply markers based on filename patterns — no per-test decoration needed.
# ----------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    """Register marker names so pytest knows about them."""
    config.addinivalue_line("markers", "unit: fast, mocked tests — no external services required")
    config.addinivalue_line(
        "markers", "integration: requires live services (ChromaDB, LM Studio, Gradio)"
    )
    config.addinivalue_line("markers", "slow: long-running or heavy setup")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-tag tests by filename so 'pytest -m unit' skips integration and slow suites."""
    integration = {
        "test_api",
        "test_chroma_store",
        "test_diagram_ui",
        "test_generation",
        "test_gradio",
        "test_pipeline",
        "test_pipeline_tracing_integration",
    }
    slow = {
        "test_corpus_content",
        "test_ingest_player_bios",
        "test_pipeline_tracing",
        "test_run_all_tests",
    }

    for item in items:
        name = item.path.stem
        if name in integration:
            item.add_marker(pytest.mark.integration)
        elif name in slow:
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.unit)


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
