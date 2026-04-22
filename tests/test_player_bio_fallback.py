"""Tests for player biography hybrid fallback.

When ChromaDB returns no corpus documents, the player_biography path should:
1. Look up the player's facts from DuckDB (name, birth info, teams, career stats)
2. Generate a bio on-the-fly using those structured facts

This is better than returning "No biography found" because the structured data
is always available for any player in the Lahman database.
"""

from unittest.mock import patch


class TestPlayerBioFallback:
    """Test that player_biography falls back to DuckDB when no corpus docs exist."""

    def test_no_chunks_falls_back_to_duckdb_facts(self):
        """If ChromaDB returns empty, answer() should still return player facts + generated bio.

        Instead of returning "No biography found", it should:
        1. Look up the player's data from DuckDB (people table)
        2. Call LLM to generate a narrative from those facts
        """
        with (
            patch("baseball_rag.cli.route") as mock_route,
            patch("baseball_rag.cli.retrieve") as mock_retrieve,
            patch("baseball_rag.db.init_db"),
            patch("baseball_rag.generation.llm.make_request") as mock_llm,
        ):
            from baseball_rag.routing import RouteResult

            # Player is known to the router
            mock_route.return_value = RouteResult(
                intent="player_biography",
                stat=None,
                time_period=None,
                position=None,
                player_name="Mickey Mantle",
                raw_question="who was Mickey Mantle",
            )
            # No corpus documents found
            mock_retrieve.return_value = []

            # LLM should still be called with DuckDB-derived facts (not just an error)
            bio = "Mickey Mantle was a Hall of Famer."
            mock_llm.return_value = type("Response", (), {"content": bio})()

            from baseball_rag.cli import answer

            result = answer("who was Mickey Mantle")

            # Should NOT return a "no biography found" error
            assert "No player biography found" not in result, (
                f"When corpus is empty but DuckDB has facts, should generate bio. Got: {result}"
            )
            # LLM should have been called to generate the bio (not just returning raw facts)
            assert mock_llm.called, (
                "LLM should be called to narrativize DuckDB facts when corpus is empty"
            )

    def test_no_chunks_includes_player_facts_from_duckdb(self):
        """When falling back, player name and basic info must appear in the prompt sent to LLM."""
        with (
            patch("baseball_rag.cli.route") as mock_route,
            patch("baseball_rag.cli.retrieve") as mock_retrieve,
            patch("baseball_rag.db.init_db"),
            patch("baseball_rag.generation.llm.make_request") as mock_llm,
        ):
            from baseball_rag.routing import RouteResult

            mock_route.return_value = RouteResult(
                intent="player_biography",
                stat=None,
                time_period=None,
                position=None,
                player_name="Ty Cobb",
                raw_question="tell me about Ty Cobb",
            )
            mock_retrieve.return_value = []

            def capture_prompt(prompt, **kwargs):
                # Capture the prompt so we can verify DuckDB facts are included
                captured_prompts.append(prompt)
                return type("Response", (), {"content": "Ty Cobb was a legendary hitter."})()

            captured_prompts = []
            mock_llm.side_effect = capture_prompt

            from baseball_rag.cli import answer

            answer("tell me about Ty Cobb")

            assert len(captured_prompts) == 1, "Expected exactly one LLM call, got " + str(
                len(captured_prompts)
            )
            # The prompt should mention the player name so LLM knows who to narrativize.
            # Support both string prompts and (system, user) tuple format.
            prompt_text = (
                captured_prompts[0]
                if isinstance(captured_prompts[0], str)
                else captured_prompts[0][1]  # user part of (system, user) tuple
            )
            assert "Cobb" in prompt_text or "Ty Cobb" in prompt_text, (
                f"Prompt sent to LLM should include player name. Got: {prompt_text[:500]}"
            )

    def test_no_chunks_player_not_in_db_returns_error(self):
        """If ChromaDB is empty AND the player isn't in DuckDB, return a clear error."""
        with (
            patch("baseball_rag.cli.route") as mock_route,
            patch("baseball_rag.cli.retrieve") as mock_retrieve,
            patch("baseball_rag.db.init_db"),
        ):
            from baseball_rag.routing import RouteResult

            mock_route.return_value = RouteResult(
                intent="player_biography",
                stat=None,
                time_period=None,
                position=None,
                player_name="Nonexistent Player XYZ123",
                raw_question="who was Nonexistent Player XYZ123",
            )
            mock_retrieve.return_value = []

            from baseball_rag.cli import answer

            result = answer("who was Nonexistent Player XYZ123")

            # Should indicate the player couldn't be found
            assert "not" in result.lower() or "no" in result.lower(), (
                f"Expected error/not-found message, got: {result}"
            )
