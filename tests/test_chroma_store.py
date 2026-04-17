"""Tests for ChromaDB vector store."""

import pytest

from baseball_rag.retrieval.chroma_store import get_store, retrieve


@pytest.fixture
def chroma_db_dir(tmp_path):
    """Build a fresh index in a temp dir."""
    from baseball_rag.corpus.ingest import build_index
    build_index(tmp_path / "chroma")
    return tmp_path / "chroma"


class TestChromaStore:
    def test_collection_exists(self, chroma_db_dir):
        """The baseball_corpus collection is present after ingest."""
        col = get_store(chroma_db_dir)
        assert col.name == "baseball_corpus"

    def test_retrieve_returns_chunks(self, chroma_db_dir):
        """retrieve() returns a list of RetrievedChunk objects."""
        results = retrieve("RBI definition", top_k=3, persist_dir=chroma_db_dir)
        assert isinstance(results, list)
        assert len(results) <= 3
        for chunk in results:
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "source")
            assert hasattr(chunk, "score")

    def test_retrieve_stat_definitions(self, chroma_db_dir):
        """Querying a stat returns relevant definition in top-3."""
        results = retrieve("what is batting average", top_k=3, persist_dir=chroma_db_dir)
        texts = [r.text.lower() for r in results]
        assert any("batting" in t or "average" in t for t in texts), (
            f"Batting avg content not in top-3: {texts}"
        )

    def test_retrieve_hof_player(self, chroma_db_dir):
        """Querying a Hall of Fame player name returns their bio."""
        results = retrieve("babe ruth biography", top_k=5, persist_dir=chroma_db_dir)
        sources = [r.source for r in results]
        assert any("Babe_Ruth" in s for s in sources), f"Babe Ruth not found: {sources}"

    def test_scores_decrease_with_rank(self, chroma_db_dir):
        """Later results have lower (worse) scores."""
        results = retrieve("home run records", top_k=5, persist_dir=chroma_db_dir)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
