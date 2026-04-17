"""Tests for database initialization and seed creation."""
import sqlite3

from baseball_rag.db import DATA_DIR, create_seed_db


class TestSeedDbCreation:
    """Test suite for the seed database creation."""

    def test_lahman_file_exists(self, tmp_path):
        """Test that lahman.sqlite is created in the target directory."""
        db_path = tmp_path / "lahman.sqlite"
        create_seed_db(db_path)

        assert db_path.exists(), f"Expected {db_path} to exist"
        assert db_path.name == "lahman.sqlite"

    def test_tables_exist(self, tmp_path):
        """Test that the seed database contains all required Lahman tables."""
        db_path = tmp_path / "lahman.sqlite"
        create_seed_db(db_path)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = {row[0] for row in cur.fetchall()}
        conn.close()

        required_tables = {"batting", "pitching", "fielding", "people", "teams"}
        missing = required_tables - table_names
        assert not missing, f"Missing tables: {missing}"

    def test_global_db_contains_seed_data(self):
        """Test that the global DATA_DIR DB has been seeded with test data."""
        db_path = DATA_DIR / "lahman.sqlite"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM people")
        count = cur.fetchone()[0]
        conn.close()

        assert count >= 5, f"Expected at least 5 players in DB, got {count}"
