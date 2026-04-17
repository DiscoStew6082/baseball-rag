# Baseball RAG Query Engine

[![CI](https://github.com/DiscoStew6082/baseball-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/DiscoStew6082/baseball-rag/actions/workflows/ci.yml)

A retrieval-augmented generation (RAG) system for MLB history, powered by a local LLM.

## What It Does

Ask questions about baseball history in natural language:

- **Stat queries:** "who had the most RBIs in 1962"
- **Player bios:** "who was Babe Ruth"
- **Stat definitions:** "what is OPS"

The system retrieves relevant corpus documents, then generates a grounded answer using a local LLM (Gemma 4 via LM Studio).

## Architecture

```
User Question
     │
     ▼
┌─────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│   Router    │───▶│ ChromaDB Store   │    │  DuckDB / Lahman DB  │
│(query路由)   │    │ (corpus向量检索)       │  (结构化棒球数据查询)      │
└─────────────┘    └──────────────────┘    └──────────────────────┘
                           │                         │
                           ▼                         ▼
                    ┌─────────────────────────────────────────┐
                    │         Generation (LLM + Prompts)      │
                    │   Grounds response in retrieved context  │
                    └─────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed breakdown.

## Quick Start

### Prerequisites

- Python 3.12+
- [LM Studio](https://lmstudio.ai/) running with Gemma 4 (or another compatible model)
- Dependencies: `uv sync`

### Setup

```bash
# Download the Lahman baseball database
uv run python -m baseball_rag.db.download

# Build the vector index from corpus documents
uv run python -m baseball_rag.corpus.ingest

# Start the API server
uv run uvicorn baseball_rag.api.server:app --reload

# In another terminal, start the web UI
uv run python -m baseball_rag.web_app
```

Or use the CLI:

```bash
uv run python -m baseball_rag.cli ask "who was Babe Ruth"
```

### Docker (HuggingFace Space)

The `space-app/` directory is configured for [HuggingFace Spaces](https://huggingface.co/spaces). See `space-app/README.md` for details.

## Project Structure

```
src/baseball_rag/
├── api/           # FastAPI server endpoints
├── cli.py         # Command-line interface
├── corpus/        # Markdown documents + ingestion logic
│   ├── stat_definitions/  # Baseball stat definitions (HR, RBI, OPS...)
│   └── hof/              # Hall of Fame player biographies
├── db/            # DuckDB schema and queries (Lahman data)
├── generation/    # LLM prompting and answer synthesis
├── retrieval/     # ChromaDB vector store operations
└── routing/       # Query classification (stat vs. general)
```

## Testing

```bash
uv run pytest -v
```