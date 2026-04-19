from pathlib import Path

from baseball_rag.corpus.ingest import build_index

if __name__ == "__main__":
    build_index(Path("data"))
