# Development

## Setup

```bash
git clone <repo-url>
cd baseball-rag
uv sync
```

### Data Dependencies

The project requires MLB data from Sean Lahman's database (hosted by NeuGrid/baseballdata). Download once:

```bash
uv run python -m baseball_rag.db.download
```

This fetches CSV files into `data/` — required before running queries or tests.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://localhost:1234/v1/chat/completions` | LLM endpoint for generation |

## Running Locally

```bash
# CLI (stat query — DuckDB)
uv run python -m baseball_rag.cli "who had the most RBIs in 1962"

# API server (port 8000)
uv run uvicorn baseball_rag.api.server:app --reload

# Web UI (port 7860)
uv run python -m baseball_rag.web_app
```

## Code Quality

### Lint

```bash
uv run ruff check src/ tests/
```

### Type Check

```bash
uv run mypy src/
```

### Tests

```bash
uv run pytest tests/ -v
```

### Coverage Report

```bash
uv run pytest --cov=baseball_rag --cov-report=term-missing
```

Coverage report is also generated as `coverage.xml` and `coverage.html` (see `.coverage` and `htmlcov/` after runs).

## CI Pipeline

`.github/workflows/ci.yml` runs three jobs in sequence:

| Job | Depends On | What it does |
|-----|------------|--------------|
| `lint` | — | `ruff check src/ tests/` |
| `typecheck` | — | `mypy src/` + type stubs |
| `test` | lint, typecheck | Full pytest suite with coverage upload to Codecov |

Python version: **3.11** (ubuntu-latest). All dependencies installed via pip (not uv) in CI to avoid PATH issues.

## Project Conventions

- Package location: `src/baseball_rag/` (explicit package discovery via `[tool.hatch.build.targets.wheel]` in pyproject.toml)
- Tests live in `tests/`, mirror source layout
- ChromaDB and DuckDB are initialized lazily on first use (no manual setup step beyond `db download`)
- Corpus index is wiped and rebuilt on every `ingest` call for reproducibility

## Adding a New Stat or Player

1. Create the markdown file in `corpus/stat_definitions/` or `corpus/hof/`
2. Rebuild: `uv run python -m baseball_rag.corpus.ingest`
3. No other changes needed — routing, retrieval, and prompts all derive from frontmatter automatically
