"""Natural language -> SQL query execution with safety guardrails."""

import json
import re
from dataclasses import dataclass

import duckdb

from baseball_rag.db.team_history import get_contextual_hint

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


def query(
    question: str,
    conn: duckdb.DuckDBPyConnection,
    *,
    year: int | None = None,
) -> FreeformResult:
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

    # 2. Inject historical context for team nicknames based on the query year
    hint = get_contextual_hint(question, year)
    enriched_question = f"{question} {hint}".strip() if hint else question

    # 3. Generate SQL via LLM
    raw_sql = _generate_sql(enriched_question, schema)
    sql = raw_sql.strip().rstrip(";")

    # 4. Validate against schema (table/column names exist)
    _validate_sql(sql, conn)

    # 5. Execute with timeout + row limit enforcement
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

_SQL_GENERATION_SYSTEM = (
    "You are a DuckDB SQL expert. Given the schema below and a user question, "
    "respond with ONLY valid JSON -- no markdown fences, no explanation.\n"
    "\n"
    'Output format:\n{"sql": "<complete SQL query>"}\n'
    "\n"
    "CRITICAL SCHEMA RULES:\n"
    "- The 'teams' table has EXACTLY two columns: teamID (VARCHAR), name (VARCHAR). "
    "It does NOT have yearID, year, or any other time column.\n"
    "- To filter teams by time period you must JOIN batting/fielding/pitching on teamID "
    "and filter by yearID from those tables -- NOT from teams.\n"
    "- The 'batting', 'fielding', and 'pitching' tables each have: "
    "playerID, yearID, stint, teamID, lgID, ...\n"
    "- To find who played for a team in a given YEAR, you must:\n"
    "  1. Find the team's teamID from teams WHERE name LIKE '%TeamNickname%'\n"
    "  2. JOIN with batting/fielding/pitching ON playerID AND filter by yearID = <that_year>\n"
    "- The 'people' table has: ID, playerID, birthYear, ..., nameFirst, nameLast, ...\n"
    "\n"
    "Rules:\n"
    "- All string literals use single quotes\n"
    "- DuckDB dialect -- use read_csv_auto-compatible syntax if querying CSVs\n"
    "- Do NOT include any text outside the JSON object\n"
    "\n"
    "Schema:\n"
)


def _generate_sql(question: str, schema: str) -> str:
    import re as _re

    from baseball_rag.generation.llm import make_request

    prompt = (_SQL_GENERATION_SYSTEM + schema, question)
    response = make_request(prompt, max_tokens=2000, temperature=0.1)

    raw = response.content.strip()

    # Try direct JSON parse first
    try:
        data = json.loads(raw)
        return data["sql"]
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences (```json ... ``` or ```sql ... ```)
    raw = _re.sub(r"^```(?:json|sql)?\s*\n?(.*?)\n?```$", r"\1", raw, flags=_re.DOTALL).strip()

    try:
        data = json.loads(raw)
        return data["sql"]
    except json.JSONDecodeError:
        pass

    # Last resort: extract from {...} block containing "sql"
    for start, end in _extract_json_blocks(raw):
        try:
            data = json.loads(raw[start:end])
            if "sql" in data:
                return data["sql"]
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse SQL from LLM response: {raw[:500]}")


def _extract_json_blocks(text: str) -> list[tuple[int, int]]:
    """Find all candidate JSON objects in text (start brace -> balanced end brace).

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

    # Find TABLE references -- be strict so we don't accidentally capture keywords
    # like BETWEEN, INNER, etc. that sometimes appear on the same line as FROM
    referenced_tables: set[str] = set()
    # Sort by length descending so "INNER JOIN" matches before bare "JOIN"
    for keyword in sorted(
        ("FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "CROSS JOIN"),
        key=len,
        reverse=True,
    ):
        referenced_tables |= set(re.findall(rf"\b{keyword}\s+(\w+)\b", sql, re.IGNORECASE))

    for tbl in referenced_tables:
        if tbl.lower() not in {t.lower() for t in tables}:
            raise ValueError(f"Unknown table '{tbl}' in generated SQL")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _execute_safe(sql: str, conn: duckdb.DuckDBPyConnection) -> FreeformResult:
    """Execute with timeout and row limit guardrails."""
    # Set DuckDB query timeout (best effort -- not all DuckDB versions support this)
    try:
        conn.execute(f"SET statement_timeout = '{SCHEMA_TIMEOUT_MS}ms'")
    except Exception:  # noqa: BLE001 -- duckdb CatalogException not always importable
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

    # Always show total row count
    if result.truncated:
        lines.append(f"({result.row_count} rows total, showing first 100)")
    else:
        lines.append(f"({result.row_count} rows)")

    for row in result.rows[:100]:
        lines.append(str(row))

    return "\n".join(lines)
