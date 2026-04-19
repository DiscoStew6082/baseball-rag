"""Tests for query routing."""

from baseball_rag.routing import route


class TestRouter:
    def test_classify_stat_query(self):
        """'most RBIs in 1962' → stat=RBI, year=1962."""
        result = route("who had the most RBIs in 1962")
        assert result.intent == "stat_query"
        assert result.stat == "RBI"
        assert result.year == 1962

    def test_year_variants_four_digit(self):
        """Four-digit years parse correctly."""
        result = route("most home runs 1977")
        assert result.year == 1977

    def test_unknown_year_not_required(self):
        """Stat query without a year still routes correctly."""
        result = route("career home run leaders")
        assert result.intent == "stat_query"
        assert result.stat == "HR"

    def test_era_stat_detection(self):
        """'best ERA 1968' parses as stat=ERA, year=1968."""
        result = route("who had the best ERA in 1968")
        assert result.intent == "stat_query"
        assert result.stat == "ERA"
        assert result.year == 1968

    def test_original_question_preserved(self):
        """raw_question always contains original text."""
        q = "who led MLB in RBI in 1957"
        result = route(q)
        assert result.raw_question == q
