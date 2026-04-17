"""DuckDB CSV schema setup — zero-ingestion queries over NeuML/baseballdata CSVs."""
import duckdb

from pathlib import Path

# Project root: go up 4 levels — lahman.py -> db/ -> baseball_rag/ -> src/ -> repo/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

# Static team ID → name map (covers seed data + common teams)
TEAM_MAP = {
    "NYA": "New York Yankees",
    "SFN": "San Francisco Giants",
    "ATL": "Atlanta Braves",
    "BOS": "Boston Red Sox",
    "WS1": "Washington Senators",
}


def get_duckdb() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection with CSV data registered as tables."""
    conn = duckdb.connect(database=":memory:", read_only=False)

    # Register CSVs via read_csv_auto
    batting_path = str(DATA_DIR / "Batting.csv")
    fielding_path = str(DATA_DIR / "Fielding.csv")
    people_path = str(DATA_DIR / "People.csv")
    pitching_path = str(DATA_DIR / "Pitching.csv")

    conn.execute(f"CREATE TABLE batting AS SELECT * FROM read_csv_auto('{batting_path}')")
    conn.execute(f"CREATE TABLE fielding AS SELECT * FROM read_csv_auto('{fielding_path}')")
    conn.execute(f"CREATE TABLE people AS SELECT * FROM read_csv_auto('{people_path}')")
    conn.execute(f"CREATE TABLE pitching AS SELECT * FROM read_csv_auto('{pitching_path}')")

    # Create a teams table from TEAM_MAP so queries can JOIN on it
    teams_rows = ", ".join(f"('{k}', '{v}')" for k, v in TEAM_MAP.items())
    conn.execute(f"CREATE TABLE teams (teamID TEXT, name TEXT)")
    if teams_rows:
        conn.execute(f"INSERT INTO teams VALUES {teams_rows}")

    return conn


def init_db() -> None:
    """No-op — kept for backward compatibility with code that calls it."""
    pass