"""Tests for corpus diagnostics."""

import json
import subprocess
import sys

from baseball_rag.corpus.diagnostics import corpus_diagnostics


def test_corpus_diagnostics_tolerates_missing_index(tmp_path):
    diagnostics = corpus_diagnostics(tmp_path / "missing")

    assert diagnostics["persist_dir"] == str(tmp_path / "missing")
    assert diagnostics["chroma_collection"]["persist_dir_exists"] is False
    assert diagnostics["chroma_collection"]["exists"] is False
    assert diagnostics["chroma_collection"]["indexed_count"] == 0
    assert diagnostics["manifest"]["exists"] is False
    assert diagnostics["corpus_files"]["stat_definition_count"] == 10
    assert diagnostics["corpus_files"]["hof_bio_count"] == 5


def test_corpus_diagnostics_counts_static_index_and_manifest(tmp_path):
    from baseball_rag.corpus.ingest import build_index

    persist_dir = tmp_path / "chroma"
    build_index(persist_dir, include_players=False)

    diagnostics = corpus_diagnostics(persist_dir)

    assert diagnostics["persist_dir"] == str(persist_dir)
    assert diagnostics["chroma_collection"]["persist_dir_exists"] is True
    assert diagnostics["chroma_collection"]["exists"] is True
    assert diagnostics["chroma_collection"]["indexed_count"] == 15
    assert diagnostics["chroma_collection"]["category_counts"] == {
        "hof_bio": 5,
        "stat_definition": 10,
    }
    assert diagnostics["chroma_collection"]["doc_kind_counts"] == {"missing": 15}
    assert diagnostics["manifest"]["exists"] is True
    assert diagnostics["manifest"]["static_document_count"] == 15
    assert diagnostics["manifest"]["generated_player_profile_count"] == 0
    assert diagnostics["manifest"]["document_count"] == 15


def test_corpus_diagnostics_cli_prints_json(tmp_path):
    from baseball_rag.corpus.ingest import build_index

    persist_dir = tmp_path / "chroma"
    build_index(persist_dir, include_players=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "baseball_rag.corpus",
            "diagnostics",
            "--persist-dir",
            str(persist_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["persist_dir"] == str(persist_dir)
    assert payload["chroma_collection"]["indexed_count"] == 15
