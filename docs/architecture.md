# Architecture

## System Overview

Baseball RAG is organized as a clean architecture with four distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                        API / CLI                            │
├─────────────────────────────────────────────────────────────┤
│                      Generation Layer                       │
│         (LLM prompting, answer synthesis)                   │
├─────────────────────────────────────────────────────────────┤
│  Retrieval Layer          │       Routing Layer             │
│   (ChromaDB vector        │    (Query classification)       │
│    semantic search)       │                                 │
├─────────────────────────────────────────────────────────────┤
│                     Data Layer                              │
│   ┌──────────────┐    ┌──────────────────────────────────┐  │
│   │ Corpus Docs  │    │     DuckDB / Lahman SQLite       │  │
│   │ (Markdown)   │    │     (Structured baseball data)   │  │
│   └──────────────┘    └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## The RAG Pipeline

When a user asks a question, the system:

1. **Classify** — Router determines if this is a `stat_query` or `general_explanation`
2. **Retrieve** — ChromaDB semantic search finds relevant corpus documents
3. **Generate** — LLM produces an answer grounded in the retrieved context

## Corpus Integration

The corpus is the knowledge foundation for the RAG system. It lives in `src/baseball_rag/corpus/`.

### Document Format

Each document is a Markdown file with YAML frontmatter:

```markdown
---
title: Home Runs (HR)
category: stat_definition
tags:
  - hitting
  - power
---

A home run occurs when a batter hits the ball over the outfield fence...
```

The `frontmatter.py` parser (`parse_frontmatter()`) extracts metadata and body separately. During ingestion:

1. **`title`** becomes the document's display name (used in citations like `[Source: HR.md]`)
2. **`category`** is stored as ChromaDB metadata for filtering/query routing
3. **`body`** is combined with title for embedding: `"{title}\n\n{body}"`

### Ingestion (`corpus/ingest.py`)

```python
def build_index(persist_dir: Path) -> None:
    # 1. Wipes existing "baseball_corpus" collection (reproducibility)
    # 2. Creates new collection with description metadata
    # 3. Reads all stat_definitions/*.md and hof/*.md files
    # 4. Parses frontmatter, formats text for embedding
    # 5. Batch-adds to ChromaDB with id=filename stem (e.g., "Babe_Ruth")
```

The resulting ChromaDB collection holds **15 documents**:
- 10 stat definitions: AVG, BB, 2B, ERA, HR, OPS, PO, RBI, SB, WHIP
- 5 HOF biographies: Babe Ruth, Willie Mays, Hank Aaron, Mickey Mantle, Ted Williams

### Retrieval (`retrieval/chroma_store.py`)

At query time, the user's question is embedded and cosine similarity search finds the top-k most relevant documents. These are passed to the prompt layer.

### Prompt Grounding (`generation/prompt.py`)

Two templates control how context is used:

| Template | Used For | Key Instruction |
|----------|----------|-----------------|
| `STAT_QUERY_TEMPLATE` | Stat queries (e.g., "most RBIs in 1962") | State player, team, value; explain the stat; cite sources |
| `GENERAL_EXPLANATION_TEMPLATE` | Player bios / general questions | Give thorough explanation; ground every claim with a citation |

Both templates instruct the LLM to cite documents explicitly: `[Source: HR.md]`.

## Data Layer

### DuckDB + Lahman Schema (`db/`)

Structured MLB statistics live in a DuckDB database built from Sean Lahman's baseball databank. Key tables:

- `batting` — per-season player stats (HR, RBI, AVG, OPS, etc.)
- `people` — player names, birth dates, Hall of Fame status

Queries join these with corpus knowledge to answer questions like "who had the most RBIs in 1962".

### Corpus Documents (`corpus/`)

Pure domain knowledge for RAG grounding. Not used for structured queries.

## Routing (`routing/query_router.py`)

A lightweight classifier (LLM-based or heuristic) splits queries:

- **stat_query** → DuckDB lookup + corpus stat definition context
- **general_explanation** → ChromaDB retrieval + HOF bio context

This determines which prompt template and data sources are used.

## Extending the Corpus

To add a new stat definition:
1. Create `src/baseball_rag/corpus/stat_definitions/{STAT_NAME}.md`
2. Include frontmatter with `title`, `category: stat_definition`, and `tags`
3. Rebuild the index: `uv run python -m baseball_rag.corpus.ingest`

To add a player bio:
1. Create `src/baseball_rag/corpus/hof/{Player_Name}.md`
2. Include frontmatter with `title` and `category: hof_bio`
3. Rebuild the index as above
