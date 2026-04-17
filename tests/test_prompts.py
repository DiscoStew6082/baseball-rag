"""Tests for prompt templates."""

from baseball_rag.generation.prompt import (
    build_explanation_prompt,
    build_stat_query_prompt,
)
from baseball_rag.retrieval.chroma_store import RetrievedChunk


def _fake_chunk(title: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(text=text, source=f"{title}.md", title=title, score=0.9)


class TestPromptTemplates:
    def test_stat_query_prompt_has_context(self):
        """Rendered prompt contains retrieved doc text and user question."""
        docs = [_fake_chunk("RBI", "RBI stands for Runs Batted In...")]
        question = "who had the most RBIs in 1962"
        prompt = build_stat_query_prompt(question, docs)

        assert "RBI" in prompt
        assert "who had the most RBIs in 1962" in prompt

    def test_answer_format_has_citation(self):
        """Prompt instructs model to cite sources."""
        docs = [_fake_chunk("HR", "Home runs are hits that clear the fence...")]
        question = "what is a home run"
        prompt = build_stat_query_prompt(question, docs)

        assert "[Source:" in prompt

    def test_explanation_prompt_has_citation(self):
        """General explanation template also cites sources."""
        docs = [_fake_chunk("Babe_Ruth", "Babe Ruth played for the Yankees...")]
        question = "who was babe ruth"
        prompt = build_explanation_prompt(question, docs)

        assert "[Source:" in prompt
        assert "babe" in prompt.lower()

    def test_multiple_docs_concatenated(self):
        """Multiple retrieved docs are all included."""
        docs = [
            _fake_chunk("AVG", "Batting average is H/AB..."),
            _fake_chunk("HR", "Home runs clear the fence..."),
        ]
        prompt = build_stat_query_prompt("what is slugging", docs)

        assert "AVG" in prompt or "H/AB" in prompt
        assert "home run" in prompt.lower() or "fence" in prompt
