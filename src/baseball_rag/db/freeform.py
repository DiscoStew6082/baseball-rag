"""Natural language -> SQL query execution with safety guardrails."""

import json
import re
from dataclasses import dataclass, field
from typing import cast

import duckdb

from baseball_rag.db.stat_registry import StatTable, get_stat, supported_stats, supported_tables
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
    params: list[object] = field(default_factory=list)


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
    # 1. Inject historical context for team nicknames based on the query year
    hint = get_contextual_hint(question, year)
    enriched_question = f"{question} {hint}".strip() if hint else question

    # 2. Prefer deterministic templates for common baseball-history questions.
    #    Unmatched questions fall back to LLM-backed typed intent extraction.
    assembled = _detect_template(enriched_question)
    if assembled is None:
        schema = _get_schema_cached(conn)
        spec = _generate_query_spec(enriched_question, schema)
        assembled = _assemble_sql(spec)
    sql = assembled.sql.strip().rstrip(";")

    # Validate table/column references before executing
    _validate_sql(sql, conn)

    # 4. Execute with timeout + row limit enforcement
    result = _execute_safe(sql, conn, assembled.params)

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


@dataclass(frozen=True)
class QuerySpec:
    """Structured intent extracted from a natural language question.

    Same intent always produces the same SQL, regardless of LLM version or
    temperature. The model identifies WHAT data is relevant; _assemble_sql
    handles HOW to build the query.
    """

    stat_tables: list[StatTable] = field(default_factory=list)
    # batting, pitching, fielding -- which stat tables contain the answer

    team_name_pattern: str | None = None
    # Team nickname to match via teams.name LIKE '%pattern%'

    year_value: int | None = None
    # Year to filter on (from batting/fielding/pitching.yearID)

    leader_stats: list[str] = field(default_factory=list)
    # Stats to find league-wide leaders for (e.g. ["HR", "RBI"]).
    # When non-empty, _assemble_sql builds correlated MAX() subqueries
    # across ALL teams — no teamID filter in the subquery.


QueryIntent = QuerySpec


@dataclass(frozen=True)
class AssembledSQL:
    sql: str
    params: list[object] = field(default_factory=list)


def _normalize_question(question: str) -> str:
    """Return a compact lowercase form for deterministic template matching."""
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _extract_threshold(text: str, *, default: int) -> int:
    match = re.search(r"\b(\d{2,4})\b", text)
    return int(match.group(1)) if match else default


def _extract_explicit_wins_threshold(text: str) -> int | None:
    match = re.search(
        r"\b(?:over|more than|at least|minimum|min|with|threshold|>=)\s+(\d{2,4})\s+wins?\b",
        text,
    )
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d{2,4})\s+wins?\s+(?:club|threshold)\b", text)
    return int(match.group(1)) if match else None


def _extract_year(text: str) -> int | None:
    match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", text)
    return int(match.group(1)) if match else None


def _extract_min_ipouts(text: str, *, default: int) -> int:
    match = re.search(r"\b(?:at least|minimum|min)?\s*(\d{2,4})\s+innings\b", text)
    return int(match.group(1)) * 3 if match else default


def _unsupported_sql(reason: str) -> AssembledSQL:
    return AssembledSQL(
        "SELECT ? AS unsupported_reason WHERE FALSE",
        [reason],
    )


def _detect_template(question: str) -> AssembledSQL | None:
    """Return deterministic SQL for high-value freeform baseball-history patterns."""
    q = _normalize_question(question)

    if "500 club" in q and "home run" not in q and "hr" not in q:
        return _unsupported_sql(
            "The question says 500 club but does not specify home runs or pitching wins."
        )

    if "triple crown" in q:
        return _triple_crown_sql()

    if re.search(r"\b30\s*30\b", q) or "30 30 club" in q or "thirty thirty" in q:
        return _thirty_thirty_sql()

    if (
        ("home run" in q or "homer" in q or re.search(r"\bhrs?\b", q))
        and ("500" in q or "club" in q or "career" in q)
        and not _looks_like_single_season(q)
    ):
        return _career_home_run_sql(_extract_threshold(q, default=500))

    if (
        ("wins" in q or re.search(r"\bw\b", q))
        and ("pitcher" in q or "pitching" in q or "career" in q or "500" in q)
        and not _looks_like_single_season(q)
    ):
        return _career_pitching_wins_sql(_extract_explicit_wins_threshold(q))

    if "era" in q and "career" in q:
        if not _has_era_qualification_guard(q):
            return _unsupported_sql(
                "Career ERA leader questions need an explicit qualification guard."
            )
        return _career_era_sql(_extract_min_ipouts(q, default=3000))

    if "era" in q and ("lowest" in q or "best" in q or "leader" in q or "leaders" in q):
        year = _extract_year(q)
        if year is None:
            return _unsupported_sql(
                "Season ERA leader questions need a specific year and innings qualification."
            )
        if not _has_era_qualification_guard(q):
            return _unsupported_sql(
                "Season ERA leader questions need an innings qualification guard."
            )
        return _qualified_season_era_sql(year, _extract_min_ipouts(q, default=300))

    return None


def _has_era_qualification_guard(q: str) -> bool:
    return "qualified" in q or "qualifying" in q or "enough innings" in q or " innings" in q


def _looks_like_single_season(q: str) -> bool:
    return _extract_year(q) is not None and "career" not in q and "club" not in q


def _triple_crown_sql() -> AssembledSQL:
    return AssembledSQL(
        """
        WITH season_batting AS (
            SELECT
                b.playerID,
                b.yearID,
                b.lgID,
                p.nameFirst,
                p.nameLast,
                SUM(b.HR) AS HR,
                SUM(b.RBI) AS RBI,
                SUM(b.H) AS H,
                SUM(b.AB) AS AB,
                CAST(SUM(b.H) AS DOUBLE) / NULLIF(SUM(b.AB), 0) AS AVG
            FROM batting b
            JOIN people p ON p.playerID = b.playerID
            WHERE b.lgID IN ('AL', 'NL')
            GROUP BY b.playerID, b.yearID, b.lgID, p.nameFirst, p.nameLast
            HAVING SUM(b.AB) >= ?
        ),
        league_leaders AS (
            SELECT
                yearID,
                lgID,
                MAX(HR) AS HR,
                MAX(RBI) AS RBI,
                MAX(AVG) AS AVG
            FROM season_batting
            GROUP BY yearID, lgID
        )
        SELECT
            s.nameFirst,
            s.nameLast,
            s.yearID,
            s.lgID,
            s.HR,
            s.RBI,
            ROUND(s.AVG, 3) AS AVG
        FROM season_batting s
        JOIN league_leaders l
            ON l.yearID = s.yearID
            AND l.lgID = s.lgID
            AND l.HR = s.HR
            AND l.RBI = s.RBI
            AND l.AVG = s.AVG
        ORDER BY s.yearID, s.lgID, s.nameLast, s.nameFirst
        """,
        [300],
    )


def _thirty_thirty_sql() -> AssembledSQL:
    return AssembledSQL(
        """
        SELECT
            p.nameFirst,
            p.nameLast,
            b.yearID,
            SUM(b.HR) AS HR,
            SUM(b.SB) AS SB
        FROM batting b
        JOIN people p ON p.playerID = b.playerID
        GROUP BY b.playerID, p.nameFirst, p.nameLast, b.yearID
        HAVING SUM(b.HR) >= ? AND SUM(b.SB) >= ?
        ORDER BY b.yearID, p.nameLast, p.nameFirst
        """,
        [30, 30],
    )


def _career_home_run_sql(threshold: int) -> AssembledSQL:
    return AssembledSQL(
        """
        SELECT
            p.nameFirst,
            p.nameLast,
            SUM(b.HR) AS career_HR
        FROM batting b
        JOIN people p ON p.playerID = b.playerID
        GROUP BY b.playerID, p.nameFirst, p.nameLast
        HAVING SUM(b.HR) >= ?
        ORDER BY career_HR DESC, p.nameLast, p.nameFirst
        """,
        [threshold],
    )


def _career_pitching_wins_sql(threshold: int | None) -> AssembledSQL:
    having = "HAVING SUM(pi.W) >= ?" if threshold is not None else ""
    limit = "" if threshold is not None else "LIMIT ?"
    params: list[object] = [threshold] if threshold is not None else [25]
    return AssembledSQL(
        """
        SELECT
            p.nameFirst,
            p.nameLast,
            SUM(pi.W) AS career_W
        FROM pitching pi
        JOIN people p ON p.playerID = pi.playerID
        GROUP BY pi.playerID, p.nameFirst, p.nameLast
        {having}
        ORDER BY career_W DESC, p.nameLast, p.nameFirst
        {limit}
        """.format(having=having, limit=limit),
        params,
    )


def _career_era_sql(min_ipouts: int) -> AssembledSQL:
    return AssembledSQL(
        """
        SELECT
            p.nameFirst,
            p.nameLast,
            ROUND(27.0 * SUM(pi.ER) / NULLIF(SUM(pi.IPouts), 0), 2) AS career_ERA,
            SUM(pi.IPouts) AS IPouts
        FROM pitching pi
        JOIN people p ON p.playerID = pi.playerID
        GROUP BY pi.playerID, p.nameFirst, p.nameLast
        HAVING SUM(pi.IPouts) >= ?
        ORDER BY career_ERA ASC, IPouts DESC, p.nameLast, p.nameFirst
        """,
        [min_ipouts],
    )


def _qualified_season_era_sql(year: int, min_ipouts: int) -> AssembledSQL:
    return AssembledSQL(
        """
        SELECT
            p.nameFirst,
            p.nameLast,
            pi.yearID,
            pi.lgID,
            pi.ERA,
            pi.IPouts
        FROM pitching pi
        JOIN people p ON p.playerID = pi.playerID
        WHERE pi.yearID = ?
            AND pi.IPouts >= ?
            AND pi.ERA IS NOT NULL
            AND pi.ERA = (
                SELECT MIN(p2.ERA)
                FROM pitching p2
                WHERE p2.yearID = pi.yearID
                    AND p2.lgID = pi.lgID
                    AND p2.IPouts >= ?
                    AND p2.ERA IS NOT NULL
            )
        ORDER BY pi.yearID, pi.lgID, pi.ERA, p.nameLast, p.nameFirst
        """,
        [year, min_ipouts, min_ipouts],
    )


def _parse_intent(raw: str) -> QuerySpec:
    """Parse LLM JSON output into a typed QuerySpec.

    Tries direct parse, then strips markdown fences, then extracts from {...} blocks.
    Raises ValueError if stat_tables cannot be determined.
    """
    import re as _re

    def _from_data(data: dict) -> QuerySpec | None:
        tables = data.get("stat_tables")
        if (
            not isinstance(tables, list)
            or not tables
            or any(str(t).lower() not in supported_tables() for t in tables)
        ):
            return None  # signal caller to try next block
        typed_tables = cast(list[StatTable], [str(t).lower() for t in tables])

        team_name_pattern = data.get("team_name_pattern")
        if team_name_pattern is not None and not isinstance(team_name_pattern, str):
            team_name_pattern = None

        year_value = data.get("year_value")
        if not isinstance(year_value, int):
            year_value = None

        leader_stats: list[str] = []
        for raw_stat in data.get("leader_stats") or []:
            if not isinstance(raw_stat, str):
                continue
            try:
                stat_def = get_stat(raw_stat)
            except ValueError:
                continue
            if stat_def.table in typed_tables:
                leader_stats.append(stat_def.canonical)

        return QuerySpec(
            stat_tables=typed_tables,
            team_name_pattern=team_name_pattern,
            year_value=year_value,
            leader_stats=leader_stats,
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


def _leader_condition(tbl: StatTable, stat: str) -> str:
    """Return the WHERE clause fragment for a league-wide leader condition.

    For regular stats (HR, RBI):  tbl.stat = (SELECT MAX(stat) FROM ... WHERE yearID matches)
    For AVG:                     computed formula = (SELECT MAX(computed) FROM ...)
    Subquery never filters by teamID — it finds the max across ALL teams in that year.
    """
    stat_def = get_stat(stat, table=tbl)
    outer = stat_def.expression(tbl)
    inner_expr = stat_def.expression("b2")
    aggregate = "MAX" if stat_def.higher_is_better else "MIN"
    inner = (
        f"SELECT {aggregate}({inner_expr}) FROM {tbl} b2 "
        f"WHERE b2.yearID = {tbl}.yearID AND b2.lgID = {tbl}.lgID"
    )
    if stat_def.min_sample_clause:
        inner += f" AND {stat_def.min_sample_clause.format(alias='b2')}"

    return f"{outer} = ({inner})"


def _assemble_sql(intent: QuerySpec) -> AssembledSQL:
    """Build parameterized SQL deterministically from a QuerySpec.

    Same intent always produces the same SQL, regardless of LLM version or
    temperature. The model identifies which tables and leader stats are needed;
    this function handles join logic, correlated subqueries, etc.
    """
    if not intent.stat_tables:
        raise ValueError("intent.stat_tables cannot be empty")

    union_parts: list[str] = []
    params: list[object] = []

    for tbl in intent.stat_tables:
        if tbl not in supported_tables():
            raise ValueError(f"Unsupported stat table '{tbl}'")
        # Base join: people -> stat_table on playerID
        join_conditions = [f"p.playerID = {tbl}.playerID"]

        if intent.team_name_pattern is not None:
            from_part = (
                f"SELECT DISTINCT p.nameFirst, p.nameLast "
                f"FROM people p "
                f"JOIN {tbl} ON {' AND '.join(join_conditions)} "
                f"JOIN teams t ON {tbl}.teamID = t.teamID "
                f"AND t.name ILIKE ?"
            )
            params.append(f"%{intent.team_name_pattern}%")
        else:
            from_part = (
                f"SELECT DISTINCT p.nameFirst, p.nameLast "
                f"FROM people p "
                f"JOIN {tbl} ON {' AND '.join(join_conditions)}"
            )

        where_parts: list[str] = []
        if intent.year_value is not None:
            where_parts.append(f"{tbl}.yearID = ?")
            params.append(intent.year_value)

        # Build leader conditions deterministically — no raw SQL from LLM
        for stat in intent.leader_stats:
            where_parts.append(_leader_condition(tbl, stat))

        if where_parts:
            from_part += " WHERE " + " AND ".join(where_parts)

        union_parts.append(from_part)

    # Combine with UNION (deduplicates across stat tables)
    if len(union_parts) == 1:
        return AssembledSQL(union_parts[0], params)
    return AssembledSQL("\nUNION\n".join(union_parts), params)


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
    return _assemble_sql(_generate_query_spec(question, schema)).sql


def _generate_query_spec(question: str, schema: str) -> QuerySpec:
    """Convert question to a typed query spec; SQL assembly happens separately."""
    prompt = (
        _INTENT_SYSTEM
        + "\n\nSupported stats:\n"
        + ", ".join(supported_stats())
        + "\n\nSchema:\n"
        + schema,
        question,
    )
    response = make_request(prompt, max_tokens=1000, temperature=0.1)

    try:
        intent = _parse_intent(response.content.strip())
    except ValueError:
        # Retry once with explicit warning about valid tables and the error cause
        retry_prompt = (
            _INTENT_SYSTEM
            + "\n\nCRITICAL: Only use stat_tables values from {'batting', 'pitching', 'fielding'}. "
            + "Do NOT use 'people'. Supported stats:\n"
            + ", ".join(supported_stats())
            + "\nSchema:\n"
            + schema,
            question,
        )
        response = make_request(retry_prompt, max_tokens=1000, temperature=0.1)
        intent = _parse_intent(response.content.strip())

    return intent


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
    cte_names = {
        match.group(1).lower()
        for match in re.finditer(
            r"(?:WITH|,)\s+(\w+)\s+AS\s*\(",
            sql,
            re.IGNORECASE,
        )
    }

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
        if tbl.lower() in cte_names:
            continue
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
    for tbl_alias, col in col_refs:
        if tbl_alias.isdigit():
            continue
        if col.lower() not in all_valid_cols:
            raise ValueError(
                f"Unknown column '{col}'. Valid columns: {', '.join(sorted(all_valid_cols))}"
            )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _execute_safe(
    sql: str, conn: duckdb.DuckDBPyConnection, params: list[object] | None = None
) -> FreeformResult:
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
        safe_params = params or []
        rows = conn.execute(safe_sql, safe_params).fetchall()
        columns = [d[0] for d in conn.description]
        truncated = len(rows) == MAX_ROWS
        return FreeformResult(
            sql=safe_sql,  # Store what was actually executed (includes LIMIT if appended)
            rows=rows,
            columns=columns,
            row_count=len(rows),
            truncated=truncated,
            params=safe_params,
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
