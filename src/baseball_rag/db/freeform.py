"""Natural language -> SQL query execution with safety guardrails."""

import json
import re
from dataclasses import dataclass, field

import duckdb

from baseball_rag.db.team_history import get_contextual_hint
from baseball_rag.generation.llm import make_request

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

    # 3. Generate SQL via structured intent extraction — Python assembles all SQL
    sql = _generate_sql(enriched_question, schema).strip().rstrip(";")

    # Validate table/column references before executing
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

    # Always append derived-stat guidance so the LLM never invents columns like BA, AVG
    lines.append(
        "\nComputed / derived stats (do NOT assume these exist as columns):\n"
        "  batting: AVG = CAST(H AS DOUBLE) / NULLIF(AB, 0)\n"
        "  pitching: ERA is pre-computed and exists as a column\n"
    )

    _cached_schema = "\n".join(lines)
    return _cached_schema


# ---------------------------------------------------------------------------
# SQL Generation
# ---------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Intent parsing -- deterministic by design
# -----------------------------------------------------------------------------


@dataclass
class QueryIntent:
    """Structured intent extracted from a natural language question.

    Same intent always produces the same SQL, regardless of LLM version or
    temperature. The model identifies WHAT data is relevant; _assemble_sql
    handles HOW to build the query.
    """

    stat_tables: list[str] = field(default_factory=list)
    # batting, pitching, fielding -- which stat tables contain the answer

    team_name_pattern: str | None = None
    # Team nickname to match via teams.name LIKE '%pattern%'

    year_value: int | None = None
    # Year to filter on (from batting/fielding/pitching.yearID)

    leader_stats: list[str] = field(default_factory=list)
    # Stats to find league-wide leaders for (e.g. ["HR", "RBI"]).
    # When non-empty, _assemble_sql builds correlated MAX() subqueries
    # across ALL teams — no teamID filter in the subquery.


_VALID_STAT_TABLES: frozenset[str] = frozenset({"batting", "pitching", "fielding"})


def _parse_intent(raw: str) -> QueryIntent:
    """Parse LLM JSON output into a QueryIntent.

    Tries direct parse, then strips markdown fences, then extracts from {...} blocks.
    Raises ValueError if stat_tables cannot be determined.
    """
    import re as _re

    def _from_data(data: dict) -> QueryIntent | None:
        tables = data.get("stat_tables")
        if not tables or any(t.lower() not in _VALID_STAT_TABLES for t in tables):
            return None  # signal caller to try next block
        return QueryIntent(
            stat_tables=[t.lower() for t in tables],
            team_name_pattern=data.get("team_name_pattern"),
            year_value=data.get("year_value"),
            leader_stats=[
                s.upper()
                for s in (data.get("leader_stats") or [])
                if s.upper() in _COMPUTED_STATS
                or s.upper()
                in {
                    "HR",
                    "RBI",
                    "W",
                    "L",
                    "G",
                    "GS",
                    "SV",
                    "SO",
                    "BB",
                    "ERA",
                    "AVG",
                }
            ],
        )

    candidates = [
        raw,
        _re.sub(r"^```[\w]*\s*\n?(.*?)\n?```$", r"\1", raw.strip(), flags=_re.DOTALL),
    ]

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            result = _from_data(data)
            if result is not None:
                return result
        except json.JSONDecodeError:
            pass

    # Last resort: extract from {...} blocks
    for start, end in _extract_json_blocks(raw):
        try:
            data = json.loads(raw[start:end])
            result = _from_data(data)
            if result is not None:
                return result
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not determine stat_tables from LLM response: {raw[:200]}")


_AVG_COLS = {"H", "AB"}  # columns needed to compute batting average

# Stats that must be computed inline (not stored as pre-existing columns)
_COMPUTED_STATS: set[str] = {"AVG"}


def _leader_condition(tbl: str, stat: str) -> str:
    """Return the WHERE clause fragment for a league-wide leader condition.

    For regular stats (HR, RBI):  tbl.stat = (SELECT MAX(stat) FROM ... WHERE yearID matches)
    For AVG:                     computed formula = (SELECT MAX(computed) FROM ...)
    Subquery never filters by teamID — it finds the max across ALL teams in that year.
    """
    if stat.upper() == "AVG":
        # batting average = H / AB, with guard against div-by-zero
        outer = f"CAST({tbl}.H AS DOUBLE) / NULLIF({tbl}.AB, 0)"
        inner = (
            f"SELECT MAX(CAST(H AS DOUBLE) / NULLIF(AB, 0)) FROM {tbl} b2 "
            f"WHERE b2.yearID = {tbl}.yearID AND b2.lgID = {tbl}.lgID "
            f"AND b2.AB > 100"
        )
    else:
        outer = f"{tbl}.{stat}"
        # Use COALESCE to handle years where no one has a value in this stat
        inner = (
            f"SELECT MAX({stat}) FROM {tbl} b2 "
            f"WHERE b2.yearID = {tbl}.yearID AND b2.lgID = {tbl}.lgID"
        )

    return f"{outer} = ({inner})"


def _assemble_sql(intent: QueryIntent) -> str:
    """Build SQL deterministically from a QueryIntent.

    Same intent always produces the same SQL, regardless of LLM version or
    temperature. The model identifies which tables and leader stats are needed;
    this function handles join logic, correlated subqueries, etc.
    """
    if not intent.stat_tables:
        raise ValueError("intent.stat_tables cannot be empty")

    union_parts: list[str] = []

    for tbl in intent.stat_tables:
        # Base join: people -> stat_table on playerID
        join_conditions = [f"p.playerID = {tbl}.playerID"]

        if intent.team_name_pattern is not None:
            from_part = (
                f"SELECT DISTINCT p.nameFirst, p.nameLast "
                f"FROM people p "
                f"JOIN {tbl} ON {' AND '.join(join_conditions)} "
                f"JOIN teams t ON {tbl}.teamID = t.teamID "
                f"AND t.name ILIKE '%{intent.team_name_pattern}%'"
            )
        else:
            from_part = (
                f"SELECT DISTINCT p.nameFirst, p.nameLast "
                f"FROM people p "
                f"JOIN {tbl} ON {' AND '.join(join_conditions)}"
            )

        where_parts: list[str] = []
        if intent.year_value is not None:
            where_parts.append(f"{tbl}.yearID = {intent.year_value}")

        # Build leader conditions deterministically — no raw SQL from LLM
        for stat in intent.leader_stats:
            where_parts.append(_leader_condition(tbl, stat))

        if where_parts:
            from_part += " WHERE " + " AND ".join(where_parts)

        union_parts.append(from_part)

    # Combine with UNION (deduplicates across stat tables)
    if len(union_parts) == 1:
        return union_parts[0]
    return "\nUNION\n".join(union_parts)


# -----------------------------------------------------------------------------
# SQL Generation
# -----------------------------------------------------------------------------

_INTENT_SYSTEM = (
    "You are a query planner. Given the user question, produce ONLY valid JSON "
    "-- no markdown fences, no explanation.\n"
    "\n"
    "Output format:\n"
    "{\n"
    '  "stat_tables": ["batting"],   -- list of: batting, pitching, fielding\n'
    '  "team_name_pattern": "Braves",  -- team nickname (omit if not about a team)\n'
    '  "year_value": 1936,           -- year ID filter (omit if no year mentioned)\n'
    '  "leader_stats": ["HR", "RBI"]  -- stats to find league-wide leaders for\n'
    "}\n"
    "\n"
    "Rules:\n"
    "- stat_tables: include ONLY the tables actually needed. "
    'Batting-only questions (HRs, RBIs, AVG) use ["batting"] only; '
    'pitching-only (wins, ERA) use ["pitching"] only.\n'
    "- team_name_pattern: extract nickname from question ('Yankees', 'Braves').\n"
    "- year_value: integer year when a specific year is mentioned.\n"
    '- leader_stats: stats to find league-wide leaders for — e.g. ["HR","RBI","AVG"] '
    "for Triple Crown. Use canonical names (HR, RBI, AVG, ERA, W, etc.).\n"
    "- Omit any field entirely if it does not apply -- do not guess.\n"
)


def _generate_sql(question: str, schema: str) -> str:
    """Convert question to SQL via structured intent extraction.

    The LLM provides JSON with stat_tables + leader_stats; Python builds
    all SQL deterministically — no raw SQL from the model.
    """
    prompt = _INTENT_SYSTEM + "\n\nSchema:\n" + schema, question
    response = make_request(prompt, max_tokens=1000, temperature=0.1)

    try:
        intent = _parse_intent(response.content.strip())
    except ValueError:
        # Retry once with explicit warning about valid tables and the error cause
        retry_prompt = (
            _INTENT_SYSTEM
            + "\n\nCRITICAL: Only use stat_tables values from {'batting', 'pitching', 'fielding'}. "
            + "Do NOT use 'people'. Schema:\n"
            + schema,
            question,
        )
        response = make_request(retry_prompt, max_tokens=1000, temperature=0.1)
        intent = _parse_intent(response.content.strip())

    return _assemble_sql(intent)


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
    tables = {row[2]: row[3] for row in conn.execute("SHOW ALL TABLES").fetchall()}

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

    # Validate that all aliased column references (tbl.col) actually exist.
    # We check against a union of all columns across all tables, because SQL
    # uses table-name aliases (p.people_col, t.team_col) and the validation
    # doesn't need to know which alias maps to which real table — only that
    # every referenced column is valid in *some* registered table.
    all_valid_cols: set[str] = set()
    for _tbl_name, cols in tables.items():
        all_valid_cols.update(c.lower() for c in cols)
    all_valid_cols.add("avg")  # allow computed AVG alias

    col_refs = re.findall(r"\b(\w+)\.(\w+)\b", sql, re.IGNORECASE)
    for _tbl_alias, col in col_refs:
        if col.lower() not in all_valid_cols:
            raise ValueError(
                f"Unknown column '{col}'. Valid columns: {', '.join(sorted(all_valid_cols))}"
            )


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
