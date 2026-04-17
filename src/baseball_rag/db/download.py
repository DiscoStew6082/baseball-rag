"""Download utilities for NeuML/baseballdata CSVs."""

from pathlib import Path

import requests

DATA_FILES = ["Batting.csv", "Fielding.csv", "People.csv", "Pitching.csv"]
BASE_URL = "https://huggingface.co/datasets/NeuML/baseballdata/resolve/main"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def download_csv(filename: str, db_dir: Path) -> Path:
    """Download a single CSV from the NeuML/baseballdata HuggingFace repo.

    Args:
        filename: Name of the CSV file (e.g. 'Batting.csv').
        db_dir: Directory to save the file in.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If the HTTP response is not 200.
    """
    url = f"{BASE_URL}/{filename}"
    dest = db_dir / filename

    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download {url}: HTTP {response.status_code}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return dest


def download_all(db_dir: Path | None = None) -> list[Path]:
    """Download all four NeuML/baseballdata CSVs.

    Args:
        db_dir: Target directory. Defaults to project data/ dir.

    Returns:
        List of paths to the downloaded files.
    """
    if db_dir is None:
        db_dir = DATA_DIR

    return [download_csv(fname, Path(db_dir)) for fname in DATA_FILES]