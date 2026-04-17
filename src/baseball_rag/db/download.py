"""Download utilities for Lahman baseball database."""

from pathlib import Path

import requests

LAHMAN_SQLITE_URL = "https://github.com/chasewilliam/lahman-sqlite-2016/raw/master/lahman2016.sqlite"


def download_lahman(db_dir: Path) -> Path:
    """Download the Lahman SQLite database to the specified directory.

    Args:
        db_dir: Directory where the lahman.sqlite file will be saved.

    Returns:
        Path to the downloaded lahman.sqlite file.

    Raises:
        RuntimeError: If the download fails or returns a non-200 status code.
    """
    db_dir = Path(db_dir)
    db_path = db_dir / "lahman.sqlite"

    # Create parent directories if they don't exist
    db_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(LAHMAN_SQLITE_URL, stream=True, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to download Lahman database: HTTP {response.status_code}")

    with open(db_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return db_path
