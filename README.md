# Grounded Baseball Analytics Assistant

[![CI](https://github.com/DiscoStew6082/baseball-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/DiscoStew6082/baseball-rag/actions/workflows/ci.yml)

Baseball RAG is a local-first question answering system for MLB history. It routes natural language questions to grounded sources, answers from DuckDB or retrieved corpus documents, and returns provenance with the rows, SQL, checksums, and dataset license metadata used to support the answer.

The project is designed as an AI engineering portfolio piece: the interesting part is not that an LLM can produce baseball prose, but that the system constrains where facts come from and shows its work.

## Problem

Baseball questions mix structured analytics and fuzzy language:

- "who had the most RBIs in 1962"
- "how many HRs did Ronald Acuna Jr. have in 2023"
- "who played for the Braves in 1936"
- "what is OPS"
- "who was Babe Ruth"

A pure chatbot will often answer confidently without exposing the data. A pure SQL interface is brittle for nontechnical users. This project sits between them: language in, typed routing and whitelisted SQL in the middle, grounded answer plus evidence out.

## Architecture

```text
Question
  |
  v
Router
  |-- stat_query ------> DuckDB over NeuML/baseballdata CSVs
  |-- freeform_query --> typed query spec -> parameterized SQL -> DuckDB
  |-- player_bio ------> ChromaDB corpus retrieval -> grounded generation
  |-- explanation -----> ChromaDB corpus retrieval -> grounded generation
  |
  v
StructuredAnswer(answer, intent, sources, warnings, unsupported)
```

Key choices:

- DuckDB is the source of truth for structured stats.
- `src/baseball_rag/db/stat_registry.py` is the only stat whitelist used by SQL builders.
- Freeform SQL keeps the intent-to-SQL idea, but the model only returns a typed query spec. Python assembles parameterized SQL.
- Every DuckDB source includes `data/manifest.json` provenance: source URL, files, row counts, year coverage, checksums, download time, and license notes.
- ChromaDB is used for small curated corpus docs: stat definitions and Hall of Fame bios.

## API Example

Start the server:

```bash
uv run uvicorn baseball_rag.api.server:app --reload
```

Ask a question:

```bash
curl -s http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"who had the most RBIs in 1962"}'
```

Response shape:

```json
{
  "answer": "Top RBI leaders (1962-1962):\n  1. Davis, Tommy: 153 RBI\n  ...",
  "intent": "stat_query",
  "sources": [
    {
      "type": "duckdb",
      "label": "RBI leaderboard for 1962-1962",
      "rows": [{ "name": "Davis, Tommy", "team": "Range", "stat_value": 153 }],
      "sql": null,
      "data_manifest": {
        "dataset": { "name": "NeuML/baseballdata", "license": "CC BY-SA 3.0" },
        "coverage": { "structured_stat_years": { "min": 1871, "max": 2025 } }
      }
    }
  ],
  "warnings": [],
  "unsupported": false
}
```

Dataset provenance is also available directly:

```bash
curl -s http://127.0.0.1:8000/sources
```

## CLI And UI

CLI:

```bash
uv run baseball-rag "career home run leaders"
uv run baseball-rag "who was Babe Ruth"
```

Gradio UI:

```bash
uv run python -m baseball_rag.web_app
```

The UI shows the answer, evidence table, source JSON, and SQL for query paths that generate SQL.

## Try These Questions

These make a compact demo script for the CLI, API, or Gradio UI:

- "who had the most RBIs in 1962" - deterministic stat query from DuckDB.
- "who won the Triple Crown and which years" - deterministic freeform template with a visible provenance badge.
- "who played for the Braves in 1936" - LLM-backed typed freeform fallback using parameterized SQL.
- "who played for the Dodgers in 1947" - historical roster query with old team names handled through the database.
- "who was Babe Ruth" - player biography answer from retrieved corpus documents.
- "how many home runs did Williams have in 1941" - ambiguity should fail closed instead of guessing Ted Williams.
- "who played for the Yankees in 1950" - inspect the returned SQL, rows, source manifest, and checksums.
- "what is the Yankees score right now" - unsupported because this is historical data, not a live scoreboard.

## Data Provenance

The structured dataset is [`NeuML/baseballdata`](https://huggingface.co/datasets/NeuML/baseballdata), a copy of the Lahman Baseball Database. The local manifest records:

- CSV files: `Batting.csv`, `Fielding.csv`, `People.csv`, `Pitching.csv`
- Row counts: 128,598 batting, 174,332 fielding, 24,270 people, 57,630 pitching
- Year coverage: 1871-2025 for structured stat tables
- SHA-256 checksums for reproducibility
- Download time: `2026-04-20T13:29:00-04:00`
- License: CC BY-SA 3.0 per Hugging Face metadata

See [data/manifest.json](data/manifest.json).

Generated data policy:

- `data/manifest.json` is tracked because it documents the data contract.
- `data/*.csv` is downloaded on demand.
- `data/*.duckdb`, `data/chroma.sqlite3`, and Chroma UUID directories are generated local state and are not tracked.
- The checked-in corpus source is the Markdown under `src/baseball_rag/corpus/`; the Chroma index is just a rebuildable search index over that source.

Populate structured data and regenerate the manifest:

```bash
uv run python -m baseball_rag.db.download
```

Rebuild the current small Chroma index from checked-in Markdown only:

```bash
uv run python -m baseball_rag.corpus --static-only
```

Build the larger experimental index with generated player bios from DuckDB:

```bash
uv run python -m baseball_rag.corpus
```

Inspect the corpus/index state. This is safe before ingest, after ingest, or
after a partial failed build:

```bash
uv run python -m baseball_rag.corpus diagnostics --persist-dir data
```

## Evaluation

The golden eval set lives in [evals/questions.yaml](evals/questions.yaml). It covers:

- known stat answers and row-count expectations
- freeform typed-query cases
- SQL visibility and parameterization checks
- unsupported live/future/non-baseball questions
- ambiguous player names
- minimum sample size cases for AVG and ERA
- source manifest requirements

Current automated tests:

```bash
uv run pytest -q
```

Retrieval-only strategy comparison for Chroma-backed eval cases:

```bash
uv run python -m evals.questions --all-strategies --retrieval-only
```

The eval file is intentionally human-readable so it can drive a later test runner, CI report, or model-routing regression harness.

## Why Not Just Ask ChatGPT?

The point is deterministic execution, not smarter-sounding baseball prose. The LLM classifies intent and narrates grounded results; DuckDB executes the structured stat work. Freeform database questions become typed specs, then Python builds parameterized/template SQL instead of trusting model-written SQL. Responses expose SQL where available, result rows, retrieved source docs, and the data manifest with checksums and license metadata. Ambiguous or unsupported questions fail closed, and the eval set protects common baseball-history claims, SQL visibility, source provenance, minimum-sample rules, and live-data limitations from drifting.

## Why This Is Grounded

Grounding is enforced in several places:

- The router returns structured intent instead of prose.
- Stat SQL only accepts registered stats; unsupported stats raise instead of falling through to raw column names.
- Freeform SQL uses typed specs and bound parameters for model-supplied values.
- DuckDB sources include rows and dataset manifest metadata.
- Chroma answers are generated only after retrieving local corpus chunks; missing corpus context returns an unsupported response instead of an ungrounded answer.

## Limitations

- This is historical data, not a live MLB scoreboard, injury feed, betting model, salary database, or Statcast warehouse.
- The corpus is intentionally small: useful for demonstrating retrieval and citation mechanics, not comprehensive baseball knowledge.
- Some freeform database questions still depend on the local intent model producing a supported typed spec.
- Lahman-style historical data can encode old team IDs and historical naming conventions that require careful interpretation.

## Development

```bash
uv sync
uv run python -m baseball_rag.db.download
uv run python -m baseball_rag.corpus --static-only
uv run pytest -q
```

Useful docs:

- [API reference](docs/api.md)
- [Architecture notes](docs/architecture.md)
- [CLI notes](docs/cli.md)
