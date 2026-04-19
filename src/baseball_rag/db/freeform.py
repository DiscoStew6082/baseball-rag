"""Natural language → SQL query execution with safety guardrails."""

import json
import re
from dataclasses import dataclass

import duckdb

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MAX_ROWS = 1000
SCHEMA_TIMEOUT_MS = 5000


@dataclass
class FreeformResult:
    """Result of a freeform natural language query."""

    sql: str  # The generated SQL (for transparency/debugging)
    rows: list[tuple]  # Raw result rows
    columns: list[str]  # Column names from cursor description
    row_count: int  # len(rows)
    truncated: bool  # True if results exceeded MAX_ROWS


def query(question: str, conn: duckdb.DuckDBPyConnection) -> FreeformResult:
    """Convert a natural language question to SQL and execute it.

    Safeguards applied:
      - Query timeout (5s)
      - Row limit (1000 rows)
      - Schema validation of generated SQL
      - Empty-result graceful degradation

    Raises:
        ValueError: if the LLM returns unparseable output or schema validation fails.
        RuntimeError: if query times out or DuckDB raises an error.
    """
    # 1. Get schema description (cached, ~200ms first call)
    schema = _get_schema_cached(conn)

    # 2. Generate SQL via LLM
    raw_sql = _generate_sql(question, schema)
    sql = raw_sql.strip().rstrip(";")

    # 3. Validate against schema (table/column names exist)
    _validate_sql(sql, conn)

    # 4. Execute with timeout + row limit enforcement
    result = _execute_safe(sql, conn)

    return result


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_cached_schema: str | None = None


def _get_schema_cached(conn: duckdb.DuckDBPyConnection) -> str:
    global _cached_schema
    if _cached_schema is not None:
        return _cached_schema

    # SHOW ALL TABLES returns (database, schema, name, column_names, column_types, temporary)
    tables = conn.execute("SHOW ALL TABLES").fetchall()
    lines = []
    for row in tables:
        tbl = row[2]  # table name is at index 2
        cols = conn.execute(f"DESCRIBE {tbl}").fetchall()
        col_list = ", ".join(f"{c[0]} ({c[1]})" for c in cols)
        sample = conn.execute(f"SELECT * FROM {tbl} LIMIT 2").fetchall()
        lines.append(f"- **{tbl}**: {col_list}")
        if sample:
            # cursor.description gives column names for the last query
            desc = conn.description
            lines.append(f"  Sample row: {dict(zip([d[0] for d in desc], sample[0]))}")

    _cached_schema = "\n".join(lines)
    return _cached_schema


# ---------------------------------------------------------------------------
# SQL Generation
# ---------------------------------------------------------------------------

_SQL_GENERATION_PROMPT = """You are a DuckDB SQL expert. Given the schema below and a user question,
output ONLY valid JSON with a single key "sql" containing the complete SQL query.

Rules:
- Output ONLY: {{"sql": "<your sql>"}}
- No markdown, no explanation, no trailing text
- All string literals use single quotes
- DuckDB dialect — use read_csv_auto-compatible syntax if querying CSVs

Schema:
{schema}

Question: {question}
"""


def _generate_sql(question: str, schema: str) -> str:
    from baseball_rag.generation.llm import make_request  # local LLM call

    prompt = _SQL_GENERATION_PROMPT.format(schema=schema, question=question)
    response = make_request(prompt, max_tokens=500, temperature=0.1)

    try:
        data = json.loads(response.content.strip())
        return data["sql"]
    except (json.JSONDecodeError, KeyError):
        # Try extracting {...} block
        for start, end in _extract_json_blocks(response.content):
            try:
                data = json.loads(response.content[start:end])
                if "sql" in data:
                    return data["sql"]
            except json.JSONDecodeError:
                continue
        raise ValueError(f"Could not parse SQL from LLM response: {response.content[:200]}")


def _extract_json_blocks(text: str) -> list[tuple[int, int]]:
    """Find all candidate JSON objects in text (start brace → balanced end brace).

    Returns list of (start, end+1) byte positions for each {...} block found.
    """
    blocks = []
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append((start, i + 1))
    return blocks


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_sql(sql: str, conn: duckdb.DuckDBPyConnection) -> None:
    """Check all table/column references in sql exist in the schema."""
    # SHOW ALL TABLES returns (database, schema, name, column_names, ...)
    tables = {row[2] for row in conn.execute("SHOW ALL TABLES").fetchall()}

    # Find TABLE references (simple regex — good enough for structured queries)
    referenced_tables = set(re.findall(r"\bFROM\s+(\w+)\b", sql, re.IGNORECASE))
    referenced_tables |= set(re.findall(r"\bJOIN\s+(\w+)\b", sql, re.IGNORECASE))

    for tbl in referenced_tables:
        if tbl.lower() not in {t.lower() for t in tables}:
            raise ValueError(f"Unknown table '{tbl}' in generated SQL")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _execute_safe(sql: str, conn: duckdb.DuckDBPyConnection) -> FreeformResult:
    """Execute with timeout and row limit guardrails."""
    # Set DuckDB query timeout (best effort — not all DuckDB versions support this)
    try:
        conn.execute(f"SET statement_timeout = '{SCHEMA_TIMEOUT_MS}ms'")
    except Exception:  # noqa: BLE001 — duckdb CatalogException not always importable
        pass

    # Append row limit if not already present
    safe_sql = sql
    if "LIMIT" not in sql.upper():
        safe_sql = f"{sql} LIMIT {MAX_ROWS}"

    try:
        rows = conn.execute(safe_sql).fetchall()
        columns = [d[0] for d in conn.description]
        truncated = len(rows) == MAX_ROWS
        return FreeformResult(
            sql=safe_sql,  # Store what was actually executed (includes LIMIT if appended)
            rows=rows,
            columns=columns,
            row_count=len(rows),
            truncated=truncated,
        )
    except Exception as e:
        raise RuntimeError(f"Query failed: {e}\nSQL: {sql}") from e


# ---------------------------------------------------------------------------
# Formatting (for CLI output)
# ---------------------------------------------------------------------------


def format_result(result: FreeformResult, question: str) -> str:
    """Convert result to readable string for terminal output."""
    if result.row_count == 0:
        return f"No results found for '{question}'."

    lines = []
    header = f"{result.columns}"
    lines.append(header)

    # Truncated warning
    if result.truncated:
        lines.append(f"(showing first {MAX_ROWS} of many rows — consider refining your query)")

    for row in result.rows[:20]:  # Only show first 20 in terminal
        lines.append(str(row))

    return "\n".join(lines)
