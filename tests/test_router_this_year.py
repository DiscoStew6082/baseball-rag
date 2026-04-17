"""Tests for 'this year' handling in query routing (Bug 1)."""
import datetime

from baseball_rag.routing.query_router import _extract_year


class TestExtractYearThisYear:
    """Test that _extract_year handles 'this year' correctly."""

    def test_this_year_returns_current_year(self):
        """'_extract_year' should return the current year when text contains 'this year'."""
        current_year = datetime.datetime.now().year
        result = _extract_year(f"stats for this year {current_year}")
        assert result == current_year

    def test_this_year_standalone(self):
        """'_extract_year' with just 'this year' should return the current year."""
        current_year = datetime.datetime.now().year
        result = _extract_year("this year")
        assert result == current_year

    def test_this_year_in_question(self):
        """Query like 'who led MLB in RBI this year' should extract current year."""
        current_year = datetime.datetime.now().year
        result = _extract_year(f"most RBIs this year")
        assert result == current_year