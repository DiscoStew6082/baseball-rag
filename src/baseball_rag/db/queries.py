"""SQL query helpers for baseball statistics."""

import duckdb

from baseball_rag.arch.tracing import traced
from baseball_rag.db.duckdb_schema import TEAM_MAP, get_duckdb


def _team_name(team_id: str) -> str:
    """Map a team ID to its full name via TEAM_MAP."""
    return TEAM_MAP.get(team_id, "Unknown")


@traced(component_id="duckdb", label="DB Query")
def get_stat_leaders(stat: str, year: int) -> list[dict]:
    """Get top 10 batting stat leaders for a given year.

    Args:
        stat: The statistic to rank by (HR, RBI, H, AB, R, 2B, 3B)
        year: The season year

    Returns:
        List of dicts with keys: name, team, stat_value
    """
    # Map user-facing stat names to DB column names (quote 2B/3B)
    col_map = {
        "HR": "HR",
        "RBI": "RBI",
        "H": "H",
        "AB": "AB",
        "R": "R",
        "2B": '"2B"',
        "3B": '"3B"',
        "SB": "SB",
        "BB": "BB",
        "SO": "SO",
    }
    col = col_map.get(stat, stat)

    query = f"""
    SELECT
        p.nameLast || ', ' || p.nameFirst AS name,
        b.teamID,
        b.{col} AS stat_value
    FROM batting b
    JOIN people p ON b.playerID = p.playerID
    WHERE b.yearID = ?
      AND b.{col} IS NOT NULL
    ORDER BY b.{col} DESC
    LIMIT 10
    """

    conn = get_duckdb()
    result = conn.execute(query, [year]).fetchall()

    return [{"name": r[0], "team": _team_name(r[1]), "stat_value": r[2]} for r in result]


@traced(component_id="duckdb", label="DB Query")
def get_career_stat_leaders(stat: str, limit: int = 10) -> list[dict]:
    """Get career batting stat leaders.

    Args:
        stat: The statistic to rank by (HR, RBI, H, etc.)
        limit: Number of results to return

    Returns:
        List of dicts with keys: name, team, stat_value
    """
    col_map = {
        "HR": "HR",
        "RBI": "RBI",
        "H": "H",
        "AB": "AB",
        "R": "R",
        "2B": '"2B"',
        "3B": '"3B"',
    }
    col = col_map.get(stat, stat)

    query = f"""
    SELECT
        p.nameLast || ', ' || p.nameFirst AS name,
        SUM(b.{col}) AS stat_value
    FROM batting b
    JOIN people p ON b.playerID = p.playerID
    WHERE b.{col} IS NOT NULL
    GROUP BY p.nameLast, p.nameFirst
    HAVING SUM(b.{col}) > 0
    ORDER BY stat_value DESC
    LIMIT ?
    """

    conn = get_duckdb()
    result = conn.execute(query, [limit]).fetchall()

    return [{"name": r[0], "team": "Career", "stat_value": r[1]} for r in result]


def get_fielding_leaders(year: int, position: str) -> list[dict]:
    """Get fielding putouts leaders for a given year and position.

    Args:
        year: The season year
        position: 'OF' for all outfield, or specific 'LF'/'CF'/'RF'

    Returns:
        List of dicts with keys: player (name), stat_value (putouts)
    """
    if position.upper() == "OF":
        pos_clause = "AND f.POS IN (?, ?, ?)"
        params: list = [year, "LF", "CF", "RF"]
    else:
        pos_clause = "AND f.POS = ?"
        params = [year, position.upper()]

    query = f"""
    SELECT
        p.nameLast || ', ' || p.nameFirst AS player,
        SUM(f.PO) AS stat_value
    FROM fielding f
    JOIN people p ON f.playerID = p.playerID
    WHERE f.yearID = ?
      {pos_clause}
    GROUP BY p.nameLast, p.nameFirst
    ORDER BY stat_value DESC
    LIMIT 20
    """

    conn = get_duckdb()
    result = conn.execute(query, params).fetchall()

    return [{"player": r[0], "stat_value": r[1]} for r in result]


def _normalize(s: str) -> str:
    """ASCII-fold a string for fuzzy matching.

    Uses NFD normalization so that composed accented characters like "ñ" (U+00F1)
    decompose into base letter + combining mark, then unidecode strips the combining
    mark — yielding the same result as DuckDB's strip_accents(LOWER(...)).

    Example: "Acuña" → NFD → "Acun~a" (combining tilde) → unidecode → "acuna"
             matching DB: strip_accents(LOWER('Acuña')) = 'acuna'
    """
    import re
    import unicodedata

    from unidecode import unidecode

    return re.sub(r"[^a-z]", "", unidecode(unicodedata.normalize("NFD", s)).lower())


def get_player_stat(conn: duckdb.DuckDBPyConnection, player_name: str, stat: str) -> dict | None:
    """Get a single player's stat for their most recent season.

    Args:
        conn: Active DuckDB connection.
        player_name: Full name e.g. "Ronald Acuna" or "Matt Olson".
        stat: The statistic to fetch (HR, RBI, etc.).

    Returns:
        Dict with keys: name, year, team, stat_value, or None if not found.
    """
    col_map = {
        "HR": "HR",
        "RBI": "RBI",
        "H": "H",
        "AB": "AB",
        "R": "R",
        "2B": '"2B"',
        "3B": '"3B"',
        "SB": "SB",
        "BB": "BB",
        "SO": "SO",
    }
    col = col_map.get(stat, stat)

    # Split player name into first/last
    parts = player_name.strip().split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
    else:
        last = parts[0]
        first = None
    norm_first = _normalize(first) if first else None
    norm_last = _normalize(last)

    # Both Python and DB use ASCII-folded last names for comparison.
    # DuckDB's strip_accents(LOWER(...)) normalizes accents, matching _normalize().
    if first:
        where_clause = (
            "strip_accents(LOWER(p.nameFirst)) = ? AND strip_accents(LOWER(p.nameLast)) = ?"
        )
        params: list = [norm_first, norm_last]
    else:
        where_clause = "strip_accents(LOWER(p.nameLast)) = ?"
        params = [norm_last]

    query = f"""
    SELECT
        p.nameLast || ', ' || p.nameFirst AS name,
        b.yearID,
        b.teamID,
        b.{col} AS stat_value
    FROM batting b
    JOIN people p ON b.playerID = p.playerID
    WHERE {where_clause}
      AND b.{col} IS NOT NULL
    ORDER BY b.yearID DESC
    LIMIT 1
    """
    result = conn.execute(query, params).fetchone()
    if not result:
        return None
    return {
        "name": result[0],
        "year": result[1],
        "team": _team_name(result[2]),
        "stat_value": result[3],
    }
