"""Tests for Triple Crown query correctness.

The Triple Crown requires leading in BOTH batting average AND home runs AND RBIs.
This was a regression: the freeform query system only included HR and RBI,
omitting AVG, causing incomplete results.
"""

import pytest


class TestTripleCrownQuery:
    """Verify Triple Crown queries include all three required stats."""

    def test_triple_crown_includes_avg_in_leader_stats(self):
        """When LLM returns 'Triple Crown' intent, leader_stats must include AVG.

        The bug: _INTENT_SYSTEM prompt only mentioned HR and RBI as examples,
        so the model omitted AVG. Triple Crown = leading in AVG + HR + RBI.
        """
        import json

        from baseball_rag.db.freeform import _parse_intent

        # Simulate LLM output for "who won the Triple Crown"
        llm_response = json.dumps(
            {
                "stat_tables": ["batting"],
                "team_name_pattern": None,
                "year_value": None,
                "leader_stats": ["HR", "RBI"],  # bug: missing AVG
            }
        )

        intent = _parse_intent(llm_response)
        assert len(intent.leader_stats) == 3, (
            f"Triple Crown requires AVG + HR + RBI, got {intent.leader_stats}"
        )
        assert "AVG" in intent.leader_stats

    def test_triple_crown_sql_includes_all_three_stat_conditions(self):
        """Generated SQL must have correlated subqueries for AVG, HR, and RBI."""
        from baseball_rag.db.freeform import QueryIntent, _assemble_sql

        # Correct intent with all three stats
        intent = QueryIntent(
            stat_tables=["batting"],
            team_name_pattern=None,
            year_value=None,
            leader_stats=["HR", "RBI", "AVG"],
        )

        sql = _assemble_sql(intent)
        upper_sql = sql.upper()

        # HR and RBI are column names
        assert "HR" in upper_sql, "SQL missing HR condition"
        assert "RBI" in upper_sql, "SQL missing RBI condition"

        # AVG is computed inline as H/AB — check for that computation pattern
        assert "NULLIF(AB, 0)" in sql or "H AS DOUBLE" in sql, (
            f"AVG must be computed as H/AB with NULLIF guard. Got: {sql}"
        )

    @pytest.mark.llm
    def test_triple_crown_returns_all_winners(self):
        """Full integration: the query must return all 6 Triple Crown winners.

        MLB Triple Crown winners (batting avg + HR + RBI leaders same season):
          1901 Nap Lajoie (AL)
          1909 Ty Cobb (AL)
          1933 Jimmie Foxx (AL)
          1934 Lou Gehrig (AL)
          1956 Mickey Mantle (AL)
          1967 Carl Yastrzemski (AL)

        The original bug returned only Paul Hines, Mickey Mantle, and Ty Cobb
        because it was using HR+RBI correlation without AVG.
        """
        from baseball_rag.db.freeform import query

        conn = __import__(
            "baseball_rag.db.duckdb_schema",
            fromlist=["get_duckdb"],
        ).get_duckdb()

        result = query("who won the Triple Crown and which years did they win it", conn)

        # Should return exactly 6 rows (one per winner)
        assert result.row_count == 6, (
            f"Expected 6 Triple Crown winners, got {result.row_count}. Rows: {result.rows}"
        )

        names = {row[0] for row in result.rows}  # nameFirst, nameLast
        expected_names = {
            "Nap Lajoie",
            "Ty Cobb",
            "Jimmie Foxx",
            "Lou Gehrig",
            "Mickey Mantle",
            "Carl Yastrzemski",
        }
        assert names == expected_names, f"Expected winners {expected_names}, got {names}"

    def test_format_result_shows_year_and_full_name(self):
        """Results must show year, full name (not separate first/last), and look pretty.

        The display was showing raw tuples like:
          ['nameFirst', 'nameLast']
          ('Miguel', 'Cabrera')
        instead of:
          1901 — Nap Lajoie
          1909 — Ty Cobb
        """
        from baseball_rag.db.freeform import format_result

        # Simulate a result with yearID + full name columns + team + league
        mock_result = __import__(
            "baseball_rag.db.freeform",
            fromlist=["FreeformResult"],
        ).FreeformResult(
            sql="SELECT DISTINCT ...",
            rows=[
                (1956, "AL", "New York Yankees", "Mickey", "Mantle"),
                (1933, "NL", "Philadelphia Phillies", "Jimmie", "Foxx"),
                (1901, "AL", "Philadelphia Athletics", "Nap", "Lajoie"),
            ],
            columns=["yearID", "lgID", "teamName", "nameFirst", "nameLast"],
            row_count=3,
            truncated=False,
        )

        output = format_result(mock_result, "who won the Triple Crown")

        # Must show year, league, team, full name
        assert "1956" in output or "1901" in output  # years present
        assert "Nap Lajoie" in output
        assert "Mickey Mantle" in output
        assert "AL" in output or "NL" in output  # leagues present
        assert "Yankees" in output or "Athletics" in output  # teams present
        assert "['nameFirst', 'nameLast']" not in output  # no ugly column names
