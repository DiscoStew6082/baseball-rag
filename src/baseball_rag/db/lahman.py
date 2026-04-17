"""Database initialization and configuration."""
from pathlib import Path

# Project root is src/baseball_rag/db/../../  -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def init_db() -> None:
    """Ensure the seed database exists; create it if missing."""
    db_path = DATA_DIR / "lahman.sqlite"
    if not db_path.exists():
        from baseball_rag.db.create_seed import create_seed_db

        create_seed_db(db_path)
