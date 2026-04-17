"""Tests for generation.answer() — Phase 5.5."""
import pytest
from baseball_rag.generation import answer
from baseball_rag.retrieval.chroma_store import RetrievedChunk

class TestGenerationAnswer:
    def test_generate_with_context(self):
        """generate_answer(question, chunks) returns non-empty string mentioning context player."""
        chunks = [
            RetrievedChunk(
                text="Babe Ruth was a legendary baseball player who played for the NY Yankees from 1920-1934. He hit 714 career home runs.",
                source="hof/babe_ruth.md",
                title="Babe Ruth",
                score=0.95,
            ),
        ]
        result = answer("who was babe ruth", chunks)
        assert isinstance(result, str)
        assert len(result) > 10
        # Result should mention the player from context (or fall back to showing doc text)
        assert "ruth" in result.lower() or "babe" in result.lower()