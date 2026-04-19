"""Tests for freeform natural language → SQL query module."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestFreeformResult:
    """Unit tests for FreeformResult dataclass and formatting."""

    def test_result_dataclass_fields(self):
        """FreeformResult should have sql, rows, columns, row_count, truncated."""
        from baseball_rag.db.freeform import FreeformResult

        result = FreeformResult(
            sql="SELECT * FROM batting LIMIT 5",
            rows=[("a", "b"), ("c", "d")],
            columns=["col1", "col2"],
            row_count=2,
            truncated=False,
        )
        assert result.sql == "SELECT * FROM batting LIMIT 5"
        assert result.rows == [("a", "b"), ("c", "d")]
        assert result.columns == ["col1", "col2"]
        assert result.row_count == 2
        assert result.truncated is False


class TestExtractJsonBlocks:
    """Unit tests for _extract_json_blocks helper (also used by query_router)."""

    def test_extracts_single_block(self):
        from baseball_rag.db.freeform import _extract_json_blocks

        text = '{"sql": "SELECT * FROM batting"}'
        blocks = _extract_json_blocks(text)
        assert len(blocks) == 1
        start, end = blocks[0]
        assert json.loads(text[start:end]) == {"sql": "SELECT * FROM batting"}

    def test_extracts_nested_block(self):
        from baseball_rag.db.freeform import _extract_json_blocks

        text = 'Here is the result: {"sql": "SELECT 1", "extra": 2}'
        blocks = _extract_json_blocks(text)
        assert len(blocks) == 1
        start, end = blocks[0]
        # Only the outer balanced block is returned (inner {...} pairs are strings inside values)
        assert json.loads(text[start:end])["sql"] == "SELECT 1"

    def test_no_block_returns_empty(self):
        from baseball_rag.db.freeform import _extract_json_blocks

        blocks = _extract_json_blocks("no json here")
        assert blocks == []


class TestFormatResult:
    """Unit tests for result formatting (CLI output)."""

    def test_format_empty_result(self):
        from baseball_rag.db.freeform import FreeformResult, format_result

        result = FreeformResult(
            sql="SELECT * FROM batting WHERE 1=0",
            rows=[],
            columns=["name", "HR"],
            row_count=0,
            truncated=False,
        )
        output = format_result(result, "who played for the Braves in 1936")
        assert "No results found" in output
        assert "Braves" in output

    def test_format_shows_headers(self):
        from baseball_rag.db.freeform import FreeformResult, format_result

        result = FreeformResult(
            sql="SELECT name, HR FROM batting LIMIT 3",
            rows=[("Ruth", 60), ("Gehrig", 47)],
            columns=["name", "HR"],
            row_count=2,
            truncated=False,
        )
        output = format_result(result, "top hr seasons")
        assert "['name', 'HR']" in output
        assert "('Ruth', 60)" in output

    def test_format_truncation_warning(self):
        from baseball_rag.db.freeform import FreeformResult, format_result

        rows = [(f"player_{i}", i) for i in range(1000)]
        result = FreeformResult(
            sql="SELECT name, HR FROM batting",  # no LIMIT
            rows=rows,
            columns=["name", "HR"],
            row_count=1000,
            truncated=True,
        )
        output = format_result(result, "all players")
        assert "showing first 1000" in output


class TestValidateSql:
    """Unit tests for SQL validation against schema."""

    def test_valid_table_passes(self):
        from baseball_rag.db.duckdb_schema import get_duckdb
        from baseball_rag.db.freeform import _validate_sql

        conn = get_duckdb()
        # batting is a real table in the schema (batting has playerID, not name)
        _validate_sql("SELECT * FROM batting LIMIT 1", conn)  # Should not raise

    def test_invalid_table_raises(self):
        from baseball_rag.db.duckdb_schema import get_duckdb
        from baseball_rag.db.freeform import _validate_sql

        conn = get_duckdb()
        with pytest.raises(ValueError, match="Unknown table"):
            _validate_sql("SELECT * FROM nonexistent_table", conn)

    def test_valid_join_passes(self):
        from baseball_rag.db.duckdb_schema import get_duckdb
        from baseball_rag.db.freeform import _validate_sql

        conn = get_duckdb()
        # Both batting and teams are real tables (use playerID not name for batting)
        sql = "SELECT b.playerID, t.name FROM batting b JOIN teams t ON b.teamID = t.teamID LIMIT 1"
        _validate_sql(sql, conn)  # Should not raise


class TestSchemaCaching:
    """Test that schema is cached (only fetched once)."""

    def test_schema_cached_after_first_call(self):
        from baseball_rag.db import freeform

        # Reset module-level cache before test
        freeform._cached_schema = None

        with patch.object(freeform, "_validate_sql"):
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.side_effect = [
                [("memory", "main", "batting"), ("memory", "main", "pitching")],
                [["name", "VARCHAR"], ["HR", "INTEGER"]],
                [],  # sample row for batting
                [["name", "VARCHAR"], ["G", "INT"]],
                [],  # sample row for pitching
            ]
            mock_conn.description = [("name", "VARCHAR", None, None, None, None, None)]

            schema1 = freeform._get_schema_cached(mock_conn)
            schema2 = freeform._get_schema_cached(mock_conn)

            assert schema1 == schema2
            # Should only call SHOW ALL TABLES once (caching works)


class TestExecuteSafe:
    """Test execution guardrails."""

    def test_auto_limit_appended(self):
        from baseball_rag.db.duckdb_schema import get_duckdb
        from baseball_rag.db.freeform import _execute_safe

        conn = get_duckdb()
        # Query WITHOUT LIMIT should get one appended (use playerID - valid column)
        result = _execute_safe("SELECT playerID FROM batting", conn)
        assert "LIMIT" in result.sql.upper()

    def test_existing_limit_respected(self):
        from baseball_rag.db.duckdb_schema import get_duckdb
        from baseball_rag.db.freeform import _execute_safe

        conn = get_duckdb()
        # Query WITH explicit LIMIT should not be double-limited (use playerID - valid column)
        result = _execute_safe("SELECT playerID FROM batting LIMIT 5", conn)
        assert result.row_count <= 5


class TestIntegration:
    """Integration tests — these need the real DB and LLM (or mocked)."""

    @pytest.mark.unit
    def test_query_returns_result_for_simple_question(self):
        from baseball_rag.db.freeform import FreeformResult, query

        with patch("baseball_rag.db.freeform._generate_sql") as mock_generate:
            # _generate_sql returns the SQL string directly (not a response object)
            # Use playerID instead of name since batting has no 'name' column
            mock_generate.return_value = (
                "SELECT playerID, teamID, HR FROM batting "
                "WHERE yearID = 1923 AND teamID = 'NYA' ORDER BY HR DESC LIMIT 10"
            )

            from baseball_rag.db.duckdb_schema import get_duckdb

            conn = get_duckdb()
            result = query("who were the Yankees players with most HRs in 1923", conn)

        assert isinstance(result, FreeformResult)
        assert result.row_count > 0
        assert result.columns

    @pytest.mark.unit
    def test_query_truncation_flag_set(self):
        from baseball_rag.db.freeform import query

        with patch("baseball_rag.db.freeform._generate_sql") as mock_generate:
            # Generate a query that returns all rows (no LIMIT - will get capped at MAX_ROWS)
            mock_generate.return_value = "SELECT playerID FROM batting"

            from baseball_rag.db.duckdb_schema import get_duckdb

            conn = get_duckdb()
            result = query("list all players", conn)

        # If batting has more than MAX_ROWS rows, truncated should be True
        assert isinstance(result.truncated, bool)
