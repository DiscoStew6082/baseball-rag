"""CLI entrypoint for corpus ingestion and diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baseball_rag.corpus.diagnostics import diagnostics_json
from baseball_rag.corpus.ingest import main as ingest_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"diagnostics", "diag"}:
        parser = argparse.ArgumentParser(description="Print corpus diagnostics as JSON.")
        parser.add_argument("command", choices=["diagnostics", "diag"])
        parser.add_argument("--persist-dir", type=Path, default=None)
        parsed = parser.parse_args(args)
        print(diagnostics_json(parsed.persist_dir))
        return 0
    return ingest_main(args)


raise SystemExit(main())
