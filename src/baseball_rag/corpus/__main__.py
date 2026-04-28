import argparse
from pathlib import Path

from baseball_rag.corpus.ingest import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the local Chroma corpus index.")
    parser.add_argument("--persist-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Index only checked-in Markdown corpus docs, not generated player bios.",
    )
    args = parser.parse_args()
    build_index(args.persist_dir, include_players=not args.static_only)
