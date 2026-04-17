"""Create a minimal seed Lahman-style SQLite database for testing."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "lahman.sqlite"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS people (
    playerID TEXT PRIMARY KEY,
    nameFirst TEXT,
    nameLast TEXT,
    birthYear INTEGER,
    bats TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    teamID TEXT PRIMARY KEY,
    name TEXT,
    lgID TEXT,
    yearID INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS batting (
    playerID TEXT,
    yearID INTEGER,
    teamID TEXT,
    G INTEGER, AB INTEGER, R INTEGER, H INTEGER,
    "2B" INTEGER, "3B" INTEGER, HR INTEGER, RBI INTEGER,
    SB INTEGER, CS INTEGER, BB INTEGER, SO INTEGER,
    IBB INTEGER, HBP INTEGER, SH INTEGER, SF INTEGER,
    PRIMARY KEY (playerID, yearID, teamID)
);

CREATE TABLE IF NOT EXISTS pitching (
    playerID TEXT,
    yearID INTEGER,
    teamID TEXT,
    W INTEGER, L INTEGER, G INTEGER, GS INTEGER, CG INTEGER, SHO INTEGER,
    SV INTEGER, IPouts INTEGER, H INTEGER, ER INTEGER, HR INTEGER,
    BB INTEGER, SO INTEGER, BAVp REAL,
    PRIMARY KEY (playerID, yearID, teamID)
);

CREATE TABLE IF NOT EXISTS fielding (
    playerID TEXT,
    yearID INTEGER,
    teamID TEXT,
    POS TEXT,
    G INTEGER, PO INTEGER, A INTEGER, E INTEGER, DP INTEGER,
    PRIMARY KEY (playerID, yearID, teamID, POS)
);
"""

SEED_SQL = """
-- People
INSERT OR IGNORE INTO people VALUES ('mantlmi01', 'Mickey',  'Mantle',   1931, 'B');
INSERT OR IGNORE INTO people VALUES ('ruthb01',   'Babe',    'Ruth',     1895, 'L');
INSERT OR IGNORE INTO people VALUES ('aaronha01', 'Hank',    'Aaron',    1934, 'R');
INSERT OR IGNORE INTO people VALUES ('mayswi01',  'Willie',  'Mays',     1931, 'R');
INSERT OR IGNORE INTO people VALUES ('willite01', 'Ted',     'Williams', 1918, 'L');
INSERT OR IGNORE INTO people VALUES ('johnswa01', 'Walter',  'Johnson',  1887, 'R');

-- Teams
INSERT OR IGNORE INTO teams VALUES ('NYA', 'New York Yankees',     'AL', 1962);
INSERT OR IGNORE INTO teams VALUES ('SFN', 'San Francisco Giants', 'NL', 1983);
INSERT OR IGNORE INTO teams VALUES ('ATL', 'Atlanta Braves',       'NL', 1974);
INSERT OR IGNORE INTO teams VALUES ('BOS', 'Boston Red Sox',       'AL', 1957);
INSERT OR IGNORE INTO teams VALUES ('WS1', 'Washington Senators',  'AL', 1913);

-- Batting - Mickey Mantle 1962: #1 RBI (123) in the AL
INSERT OR IGNORE INTO batting VALUES
  ('mantlmi01', 1962, 'NYA', 123, 377, 99, 175, 28, 6, 56, 123,
   1, 3, 73, 65, NULL, 5, NULL, NULL);

-- Batting - Mickey Mantle 1961 (also RBI-heavy)
INSERT OR IGNORE INTO batting VALUES
  ('mantlmi01', 1961, 'NYA', 153, 514, 131, 180, 34, 10, 54, 128,
   2, 3, 108, 63, NULL, 4, NULL, NULL);

-- Batting — Babe Ruth 1920–1925 (career HR totals)
INSERT OR IGNORE INTO batting VALUES
  ('ruthb01', 1920, 'NYA', 142, 458, 158, 205, 34, 11, 54, 152, 9, 5, 80, 39, NULL, 3, NULL, NULL);
INSERT OR IGNORE INTO batting VALUES
  ('ruthb01', 1921, 'NYA', 152, 550, 177, 204, 44, 16, 59, 171, 2, 2, 145, 54, NULL, 4, NULL, NULL);
INSERT OR IGNORE INTO batting VALUES
  ('ruthb01', 1923, 'NYA', 159, 538, 151, 205, 45, 13, 41,
   131, 17, 8, 110, 44, NULL, 4, NULL, NULL);
INSERT OR IGNORE INTO batting VALUES
  ('ruthb01', 1924, 'NYA', 153, 529, 143, 200, 39, 7, 46, 121, 4, 4, 232, 81, NULL, 3, NULL, NULL);
INSERT OR IGNORE INTO batting VALUES
  ('ruthb01', 1925, 'NYA', 98, 359, 61, 99, 13, 2, 25, 67, 2, 1, 59, 44, NULL, 3, NULL, NULL);

-- Batting — Hank Aaron (career HR leader)
INSERT OR IGNORE INTO batting VALUES
  ('aaronha01', 1959, 'ATL', 154, 629, 116, 223, 40, 11,
   39, 123, 8, 3, 51, 54, NULL, 2, NULL, NULL);
INSERT OR IGNORE INTO batting VALUES
  ('aaronha01', 1974, 'ATL', 142, 382, 47, 91, 20, 1, 20, 69, 0, 1, 22, 29, NULL, 2, NULL, NULL);

-- Batting — Willie Mays
INSERT OR IGNORE INTO batting VALUES
  ('mayswi01', 1965, 'SFN', 159, 558, 118, 170, 33, 10, 52, 112, 9, 7, 69, 83, NULL, 4, NULL, NULL);

-- Batting — Ted Williams (1941 .406 season)
INSERT OR IGNORE INTO batting VALUES
  ('willite01', 1941, 'BOS', 143, 456, 135, 185, 28,
   3, 37, 120, 2, 6, 147, 27, NULL, 3, NULL, NULL);

-- Pitching — Walter Johnson 1913
INSERT OR IGNORE INTO pitching VALUES
  ('johnswa01', 1913, 'WS1', 29, 25, 35, 35, 30, 4, 0, 346, 243, 37, 8, 48, 175, NULL);

-- Fielding — Willie Mays CF 1983 (outfield putouts)
INSERT OR IGNORE INTO fielding VALUES
  ('mayswi01', 1983, 'SFN', 'CF', 120, 252, 4, 1, 2);
"""


def create_seed_db(db_path: Path) -> None:
    """Create a seed Lahman-style SQLite DB with known test data."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    cur.executescript(SEED_SQL)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_seed_db(DB_PATH)
    print(f"Seed DB created at {DB_PATH}")



