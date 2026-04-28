"""Tests for freeform query -- intent-based SQL generation (deterministic by design)."""

from unittest.mock import MagicMock, patch

import pytest


class TestAssembleSQL:
    """Unit tests for the deterministic SQL assembler.

    _assemble_sql(intent) must always produce the same SQL for the same intent,
    regardless of what the LLM might have returned. No LLM calls -- pure function.
    """

    def test_assembles_batting_only(self):
        from baseball_rag.db.freeform import QueryIntent, _assemble_sql

        intent = QueryIntent(
            stat_tables=["batting"],
            team_name_pattern="Braves",
            year_value=1936,
        )
        sql = _assemble_sql(intent)

        assert "people" in sql.sql.lower()
        assert "batting" in sql.sql.lower()
        assert "teams" in sql.sql.lower()
        assert "?" in sql.sql
        assert sql.params == ["%Braves%", 1936]

    def test_assembles_batting_and_pitching_union(self):
        from baseball_rag.db.freeform import QueryIntent, _assemble_sql

        intent = QueryIntent(
            stat_tables=["batting", "pitching"],
            team_name_pattern="Yankees",
            year_value=1950,
        )
        sql = _assemble_sql(intent)

        assert "union" in sql.sql.lower()
        assert "batting" in sql.sql.lower()
        assert "pitching" in sql.sql.lower()

    def test_assembles_without_year(self):
        from baseball_rag.db.freeform import QueryIntent, _assemble_sql

        intent = QueryIntent(stat_tables=["batting"], team_name_pattern="Cubs")
        sql = _assemble_sql(intent)

        assert sql.params == ["%Cubs%"]
        # No year filter means no 1936 etc.
        assert "yearid" not in sql.sql.lower() or "BETWEEN" not in sql.sql.upper()

    def test_always_distinct(self):
        from baseball_rag.db.freeform import QueryIntent, _assemble_sql

        intent = QueryIntent(
            stat_tables=["batting", "pitching"],
            team_name_pattern="Dodgers",
            year_value=1955,
        )
        sql = _assemble_sql(intent)

        # DISTINCT must be present to avoid duplicate players appearing twice
        assert "distinct" in sql.sql.lower()


class TestParseIntent:
    """Tests for the intent parser -- LLM output -> Intent dataclass."""

    def test_parses_valid_intent_json(self):
        from baseball_rag.db.freeform import _parse_intent

        raw = (
            '{"stat_tables": ["batting", "pitching"], '
            '"team_name_pattern": "Braves", "year_value": 1936}'
        )
        intent = _parse_intent(raw)

        assert intent.stat_tables == ["batting", "pitching"]
        assert intent.team_name_pattern == "Braves"
        assert intent.year_value == 1936

    def test_parses_minimal_intent(self):
        from baseball_rag.db.freeform import _parse_intent

        raw = '{"stat_tables": ["batting"]}'
        intent = _parse_intent(raw)

        assert intent.stat_tables == ["batting"]
        assert intent.team_name_pattern is None
        assert intent.year_value is None

    def test_strips_markdown_fences(self):
        from baseball_rag.db.freeform import _parse_intent

        raw = '```json\n{"stat_tables": ["fielding"], "team_name_pattern": "Giants"}\n```'
        intent = _parse_intent(raw)

        assert intent.stat_tables == ["fielding"]
        assert intent.team_name_pattern == "Giants"

    def test_raises_on_malformed_json(self):
        from baseball_rag.db.freeform import _parse_intent

        with pytest.raises(ValueError, match="Could not determine"):
            _parse_intent("not valid json at all")

    def test_raises_when_stat_tables_missing(self):
        from baseball_rag.db.freeform import _parse_intent

        with pytest.raises(ValueError, match="stat_tables"):
            _parse_intent('{"team_name_pattern": "Braves"}')


class TestDeterminismSmokeSuite:
    """Smoke tests verifying deterministic output across semantically identical inputs.

    The property under test: equivalent question phrasings must produce identical
    row counts (same players, same SQL). This is the core guarantee of our
    intent-decomposition design -- if this fails, model variance has crept back in.
    """

    def _run_query(self, question: str) -> tuple[int, list[tuple]]:
        from baseball_rag.db.freeform import query

        conn = __import__(
            "baseball_rag.db.duckdb_schema",
            fromlist=["get_duckdb"],
        ).get_duckdb()
        result = query(question, conn)
        return result.row_count, result.rows

    @pytest.mark.llm
    def test_braves_1936_variants(self):
        """All phrasings of 'Braves 1936' must return identical row counts."""
        questions = [
            "Who played for the Braves in 1936?",
            "Who were the Braves players in 1936?",
            "What players were on the Atlanta Braves in 1936?",
            "Braves roster nineteen thirty six",
        ]
        results = [self._run_query(q) for q in questions]
        counts = [r[0] for r in results]

        assert len(set(counts)) == 1, "Non-deterministic row counts across variants: " + ", ".join(
            f"{q}->{c}" for (q,), c in zip(questions, counts)
        )

    @pytest.mark.llm
    def test_braves_2022_variants(self):
        """All phrasings of 'Braves 2022' must return identical row counts."""
        questions = [
            "Who played for the Braves in 2022?",
            "What players were on the Atlanta Braves in 2022?",
            "Braves roster twenty twenty two",
        ]
        results = [self._run_query(q) for q in questions]
        counts = [r[0] for r in results]

        assert len(set(counts)) == 1, "Non-deterministic row counts across variants: " + ", ".join(
            f"{q}->{c}" for (q,), c in zip(questions, counts)
        )

    @pytest.mark.llm
    def test_yankees_1950_variants(self):
        """All phrasings of 'Yankees 1950' must return identical row counts."""
        questions = [
            "Who played for the Yankees in 1950?",
            "What players were on the New York Yankees in 1950?",
            "Yankees roster nineteen fifty",
        ]
        results = [self._run_query(q) for q in questions]
        counts = [r[0] for r in results]

        assert len(set(counts)) == 1, "Non-deterministic row counts across variants: " + ", ".join(
            f"{q}->{c}" for (q,), c in zip(questions, counts)
        )


class TestGenerateSQLDeterminism:
    """Integration-style tests verifying deterministic output for the same inputs."""

    def test_same_prompt_produces_same_sql_twice(self):
        """Identical calls with same intent should produce byte-for-byte identical SQL."""
        import json

        from baseball_rag.db.freeform import _generate_sql

        # Mock the LLM to return a known intent JSON
        raw_response = json.dumps(
            {
                "stat_tables": ["batting"],
                "team_name_pattern": "Braves",
                "year_value": 1936,
            }
        )

        mock_resp = MagicMock()
        mock_resp.content = raw_response

        with patch("baseball_rag.db.freeform.make_request", return_value=mock_resp):
            sql1 = _generate_sql("Who played for the Braves in 1936?", "schema")
            sql2 = _generate_sql("Who played for the Braves in 1936?", "schema")

        assert sql1 == sql2, f"Non-deterministic SQL: {sql1!r} != {sql2!r}"

    def test_generate_sql_calls_llm_once(self):
        """_generate_sql should make exactly one LLM call per invocation."""
        from baseball_rag.db.freeform import _generate_sql

        mock_resp = MagicMock()
        mock_resp.content = (
            '{"stat_tables": ["batting"], "team_name_pattern": "Braves", "year_value": 1936}'
        )

        with patch("baseball_rag.db.freeform.make_request", return_value=mock_resp) as mock_call:
            _generate_sql("Who played for the Braves in 1936?", "schema")
            assert mock_call.call_count == 1
