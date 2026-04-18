"""Generate markdown biography documents for baseball players from DuckDB data."""

from typing import Optional

import duckdb


def build_player_bio(player_id: str, conn: duckdb.DuckDBPyConnection) -> str:
    """Generate a markdown biography document for a player from DuckDB data.

    Args:
        player_id: The player's ID (e.g., 'ruthb001bab-01' or 'olso001mat-01').
            You MUST look up the playerID by name — don't assume you know it.
        conn: A DuckDB connection from get_duckdb()

    Returns a markdown string with YAML frontmatter:
    ---
    title: Player Name
    player_id: xxx
    category: player_biography
    ---
    # Player Name

    **Full name:** ...
    **Bats/Throws:** ...
    **Primary position:** ...

    ## Career Summary
    Played for TEAM (YEARS), TEAM (YEARS)...

    ## Season-by-Season Teams
    - YEAR: Team
    - YEAR: Team
    """
    from baseball_rag.db.duckdb_schema import TEAM_MAP

    # 1. Get player info from people table
    person = conn.execute(
        "SELECT nameFirst, nameLast, birthCity, birthState, bats, throws, debut, finalGame "
        "FROM people WHERE playerID = ?",
        [player_id],
    ).fetchone()

    if not person:
        raise ValueError(f"Player {player_id} not found in database")

    name_first, name_last, birth_city, birth_state, bats, throws = (
        person[0],
        person[1],
        person[2],
        person[3],
        person[4],
        person[5],
    )
    debut = str(person[6]) if person[6] else "Unknown"
    final_game = str(person[7]) if person[7] else "Unknown"

    full_name = f"{name_first} {name_last}"
    bats_throws = f"{bats}/{throws}" if bats and throws else "Unknown"

    # 2. Get all batting records to build season list and teams summary
    batting_records = conn.execute(
        "SELECT yearID, teamID FROM batting WHERE playerID = ? ORDER BY yearID",
        [player_id],
    ).fetchall()

    if not batting_records:
        raise ValueError(f"No batting records found for {player_id}")

    # Build season list and group by (year, team) pairs
    seasons: list[tuple[int, str]] = []
    seen_teams: dict[str, tuple[int, int | None]] = {}  # teamID -> (first_year, last_year)

    for year_id, team_id in batting_records:
        if team_id is not None:
            seasons.append((int(year_id), team_id))
            if team_id not in seen_teams:
                seen_teams[team_id] = (int(year_id), int(year_id))
            else:
                old_first, _ = seen_teams[team_id]
                seen_teams[team_id] = (old_first, int(year_id))

    # 3. Get primary position from fielding (most games at which position)
    pos_row = conn.execute(
        "SELECT POS FROM fielding WHERE playerID = ? GROUP BY POS ORDER BY SUM(G) DESC LIMIT 1",
        [player_id],
    ).fetchone()
    primary_position = pos_row[0] if pos_row else "Unknown"

    # 4. Build career summary (team + year ranges)
    team_summary_parts: list[str] = []
    for team_id, (first_year, last_year) in sorted(seen_teams.items(), key=lambda x: x[1][0]):
        team_name = TEAM_MAP.get(team_id, team_id)
        if first_year == last_year:
            team_summary_parts.append(f"{team_name} ({first_year})")
        else:
            team_summary_parts.append(f"{team_name} ({first_year}-{last_year})")

    career_summary = ", ".join(team_summary_parts)

    # 5. Format season-by-season list
    season_lines: list[str] = []
    current_year: Optional[int] = None
    for year_id, team_id in seasons:
        team_name = TEAM_MAP.get(team_id, team_id)
        if year_id != current_year:
            season_lines.append(f"- {year_id}: {team_name}")
            current_year = year_id
        else:
            # Same year, multiple teams (stints) - append on same line
            season_lines[-1] += f", {team_name}"

    # 6. Build markdown with YAML frontmatter
    lines: list[str] = []
    lines.append("---")
    lines.append(f"title: {full_name}")
    lines.append(f"player_id: {player_id}")
    lines.append("category: player_biography")
    lines.append("---")
    lines.append("")
    lines.append(f"# {full_name}")
    lines.append("")
    if birth_city or birth_state:
        location = ", ".join(filter(None, [birth_city, birth_state]))
        lines.append(f"**Born:** {location}  ")
    lines.append(f"**Bats/Throws:** {bats_throws}  ")
    lines.append(f"**Primary position:** {primary_position}")
    lines.append(f"**Debut:** {debut}  ")
    if final_game != debut:
        lines.append(f"**Final Game:** {final_game}  ")
    lines.append("")
    lines.append("## Career Summary")
    lines.append(career_summary)
    lines.append("")
    lines.append("## Season-by-Season Teams")
    for line in season_lines:
        lines.append(line)

    return "\n".join(lines)


def get_player_id_by_name(name: str, conn: duckdb.DuckDBPyConnection) -> str | None:
    """Look up a player's ID by their name.

    Args:
        name: Full name or partial name to search for
        conn: A DuckDB connection

    Returns:
        The playerID if found, or None
    """
    # Try exact match first
    row = conn.execute(
        "SELECT playerID FROM people WHERE nameFirst || ' ' || nameLast = ?",
        [name],
    ).fetchone()
    if row:
        return row[0]

    # Try partial match (LIKE)
    pattern = f"%{name}%"
    row = conn.execute(
        "SELECT playerID FROM people WHERE nameFirst LIKE ? OR nameLast LIKE ? LIMIT 1",
        [pattern, pattern],
    ).fetchone()
    if row:
        return row[0]

    return None
