"""Tests for generation.answer() — Phase 5.5."""

from unittest.mock import patch

import pytest

from baseball_rag.generation import answer
from baseball_rag.retrieval.chroma_store import RetrievedChunk


class TestGenerationAnswer:
    def test_generate_with_context(self):
        """generate_answer(question, chunks) returns non-empty string mentioning context player."""
        chunks = [
            RetrievedChunk(
                text=(
                    "Babe Ruth was a legendary baseball player who played for "
                    "the NY Yankees from 1920-1934. He hit 714 career home runs."
                ),
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


class TestGenerationExceptionHandling:
    def test_timeout_error_propagates_not_swallowed(self):
        """TimeoutError from make_request propagates instead of returning silent fallback."""

        def fake_request(prompt):
            raise TimeoutError("LM Studio timed out after 120s")

        with patch("baseball_rag.generation.llm.make_request", fake_request):
            chunks = [
                RetrievedChunk(
                    text="Babe Ruth hit 714 home runs.",
                    source="hof/babe_ruth.md",
                    title="Babe Ruth",
                    score=0.95,
                ),
            ]
            with pytest.raises(TimeoutError, match="timed out"):
                answer("how many HR did Babe Ruth have", chunks)

    def test_json_decode_error_is_not_silenced(self):
        """JSON decode error from LM Studio raises instead of silent fallback."""

        import json

        def fake_request(prompt):
            # Simulate what happens when LM Studio returns garbled non-JSON
            raise json.JSONDecodeError("Expecting value", '{"model": "gemma', 10)

        with patch("baseball_rag.generation.llm.make_request", fake_request):
            chunks = [
                RetrievedChunk(
                    text="Babe Ruth hit 714 home runs.",
                    source="hof/babe_ruth.md",
                    title="Babe Ruth",
                    score=0.95,
                ),
            ]
            with pytest.raises(json.JSONDecodeError):
                answer("how many HR did Babe Ruth have", chunks)
