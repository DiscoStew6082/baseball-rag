"""SQL query helpers for baseball statistics."""

from baseball_rag.db.duckdb_schema import TEAM_MAP, get_duckdb


def _team_name(team_id: str) -> str:
    """Map a team ID to its full name via TEAM_MAP."""
    return TEAM_MAP.get(team_id, "Unknown")


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
    conn.close()

    return [
        {"name": r[0], "team": _team_name(r[1]), "stat_value": r[2]} for r in result
    ]


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
    conn.close()

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
        pos_clause = "AND f.POS IN ('LF', 'CF', 'RF')"
        params: list = [year]
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
    conn.close()

    return [{"player": r[0], "stat_value": r[1]} for r in result]
