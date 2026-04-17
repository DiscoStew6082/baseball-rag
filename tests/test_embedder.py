"""Tests for embedder client — Phase 3.4."""
from baseball_rag import embedder


class TestEmbedder:
    def test_embed_returns_list_of_floats(self):
        """embed() returns a list of floats, len > 100."""
        result = embedder.embed("hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
        assert len(result) > 100

    def test_embed_consistent(self):
        """Same input yields same output."""
        a = embedder.embed("test string")
        b = embedder.embed("test string")
        assert a == b

