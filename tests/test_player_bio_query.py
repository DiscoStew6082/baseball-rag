"""Tests for player biography retrieval + generation path in CLI."""

from unittest.mock import patch

from baseball_rag.cli import answer
from baseball_rag.retrieval.chroma_store import RetrievedChunk


class TestPlayerBioQuery:
    """Test the player_biography intent handling in cli.answer()."""

    def test_player_biography_intent_routes_correctly(self):
        """A question routed as player_biography should use bio retrieval path."""
        # Mock the route to return player_biography
        mock_chunk = RetrievedChunk(
            text=(
                "Wally Pipp was a first baseman who played for "
                "the Chicago Cubs and New York Yankees."
            ),
            source="/path/to/pipp.md",
            title="Wally Pipp",
            score=0.95,
        )

        with (
            patch("baseball_rag.cli.route") as mock_route,
            patch("baseball_rag.cli.retrieve") as mock_retrieve,
            patch("baseball_rag.db.init_db"),
        ):
            from baseball_rag.routing import RouteResult

            mock_route.return_value = RouteResult(
                intent="player_biography",
                stat=None,
                year=None,
                position=None,
                player_name="Wally Pipp",
                raw_question="who was Wally Pipp",
            )
            mock_retrieve.return_value = [mock_chunk]

            answer("who was Wally Pipp")

            # Should have called retrieve with the player name
            mock_retrieve.assert_called()
            call_args = mock_retrieve.call_args[0]
            assert "Wally Pipp" in call_args[0] or call_args[0] == "Wally Pipp"

    def test_player_biography_uses_bio_prompt(self):
        """Player biography path should use build_player_bio_prompt, not explanation prompt."""
        mock_chunk = RetrievedChunk(
            text="Rogers Hornsby was a Hall of Fame second baseman.",
            source="/path/to/hornsby.md",
            title="Rogers Hornsby",
            score=0.9,
        )

        with (
            patch("baseball_rag.cli.route") as mock_route,
            patch("baseball_rag.cli.retrieve") as mock_retrieve,
            patch("baseball_rag.db.init_db"),
            patch("baseball_rag.generation.prompt.build_player_bio_prompt") as mock_prompt_builder,
        ):
            from baseball_rag.routing import RouteResult

            mock_route.return_value = RouteResult(
                intent="player_biography",
                stat=None,
                year=None,
                position=None,
                player_name="Rogers Hornsby",
                raw_question="tell me about Rogers Hornsby",
            )
            mock_retrieve.return_value = [mock_chunk]
            mock_prompt_builder.return_value = "fake prompt"

            answer("tell me about Rogers Hornsby")

            # The bio path should use build_player_bio_prompt
            assert mock_prompt_builder.called, (
                "player_biography intent should use build_player_bio_prompt"
            )

    def test_player_biography_connection_error_fallback(self):
        """If LM Studio is down during player biography query, show docs instead."""
        mock_chunk = RetrievedChunk(
            text="Mickey Mantle was a switch-hitting outfielder for the New York Yankees.",
            source="/path/to/mantle.md",
            title="Mickey Mantle",
            score=0.95,
        )

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
                year=None,
                position=None,
                player_name="Mickey Mantle",
                raw_question="who was Mickey Mantle",
            )
            mock_retrieve.return_value = [mock_chunk]
            # Simulate LM Studio being down
            mock_llm.side_effect = ConnectionError("LM Studio not running")

            result = answer("who was Mickey Mantle")

            assert (
                "LM Studio not running" in result or "showing relevant documents" in result.lower()
            )
            # Should show the chunk content as fallback
            assert "Mickey Mantle" in result

    def test_player_biography_no_chunks_returns_helpful_message(self):
        """If no bio chunks found, return a helpful message."""
        with (
            patch("baseball_rag.cli.route") as mock_route,
            patch("baseball_rag.cli.retrieve") as mock_retrieve,
            patch("baseball_rag.db.init_db"),
        ):
            from baseball_rag.routing import RouteResult

            mock_route.return_value = RouteResult(
                intent="player_biography",
                stat=None,
                year=None,
                position=None,
                player_name="Unknown Player",
                raw_question="who was Unknown Player",
            )
            mock_retrieve.return_value = []  # No results

            result = answer("who was Unknown Player")

            assert "No player biography found" in result or "not in the dataset" in result.lower()

    def test_player_biography_not_found_error_shows_ingest_message(self):
        """If ChromaDB raises NotFoundError, suggest running ingest."""
        with (
            patch("baseball_rag.cli.route") as mock_route,
            patch("baseball_rag.cli.retrieve") as mock_retrieve,
            patch("baseball_rag.db.init_db"),
        ):
            from baseball_rag.routing import RouteResult

            mock_route.return_value = RouteResult(
                intent="player_biography",
                stat=None,
                year=None,
                position=None,
                player_name="Some Player",
                raw_question="who was Some Player",
            )
            # Simulate ChromaDB not found error
            mock_retrieve.side_effect = Exception("NotFoundError: collection not found")

            result = answer("who was Some Player")

            assert "ingest" in result.lower() or "indexed" in result.lower()
