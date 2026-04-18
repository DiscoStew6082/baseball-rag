"""Tests for player bio generator."""

import pytest

from baseball_rag.corpus.player_bios import build_player_bio
from baseball_rag.db.duckdb_schema import get_duckdb


@pytest.fixture
def conn():
    """Get a DuckDB connection."""
    return get_duckdb()


class TestBuildPlayerBio:
    def test_build_player_bio_returns_string(self, conn):
        """Basic sanity: function returns a string."""
        # Use Dick Littlefield who played for many teams (pitcher)
        result = build_player_bio("littldi01", conn)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_player_bio_has_frontmatter(self, conn):
        """Bio should have YAML frontmatter with required fields."""
        result = build_player_bio("littldi01", conn)
        # Check frontmatter
        assert result.startswith("---")
        assert "title:" in result or "player_id:" in result
        assert "category: player_biography" in result

    def test_build_player_bio_has_player_name(self, conn):
        """Bio should contain the player's name."""
        result = build_player_bio("littldi01", conn)
        # Dick Littlefield's full name should appear
        assert "Dick Littlefield" in result or "Littlefield" in result

    def test_build_player_bio_has_career_summary(self, conn):
        """Bio should have a career summary section."""
        result = build_player_bio("littldi01", conn)
        assert "Career Summary" in result or "career" in result.lower()

    def test_build_player_bio_with_position_player(self, conn):
        """Test with George Kell, a position player (3B) who played for multiple teams."""
        # First verify he exists and has multiple teams
        row = conn.execute(
            "SELECT COUNT(DISTINCT teamID) FROM batting WHERE playerID = 'kellge01'"
        ).fetchone()
        assert row[0] > 1, "George Kell should have played for multiple teams"

        result = build_player_bio("kellge01", conn)
        assert isinstance(result, str)
        assert "Career Summary" in result
        # George Kell was a third baseman
        assert "3B" in result or "third" in result.lower() or "Third" in result

    def test_build_player_bio_season_list(self, conn):
        """Bio should list seasons chronologically."""
        result = build_player_bio("littldi01", conn)
        # Should have season-by-season section with years in range 1940-1970
        import re

        years_in_bio = [int(y) for y in re.findall(r"- (\d{4}):", result)]
        assert len(years_in_bio) > 0, (
            f"Should have year references in season list. Got: {result[:200]}"
        )
        # Verify they're roughly in order (allowing for team changes within a year)
        assert years_in_bio == sorted(years_in_bio), (
            f"Years should be chronological: {years_in_bio}"
        )
