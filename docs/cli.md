# CLI Reference

## `baseball-rag`

Single command for ad-hoc queries against the grounded answer pipeline.

### Usage

```bash
uv run python -m baseball_rag.cli "your question here"
```

Or via the installed entry point (if configured):

```bash
baseball-rag "who was Babe Ruth"
```

### Arguments

All positional arguments after the script name are joined into a single question string. There are no flags — everything is expressed as natural language.

### Examples

```bash
# Stat query with year filter → DuckDB lookup
uv run python -m baseball_rag.cli "who had the most RBIs in 1962"

# Career stat leaders → DuckDB lookup (no year)
uv run python -m baseball_rag.cli "career home run leaders"

# Player bio / general question → ChromaDB retrieval + LLM generation
uv run python -m baseball_rag.cli "who was Babe Ruth"
uv run python -m baseball_rag.cli "what is OPS"
```

### How It Works

```
answer(question)
  │
  ├─▶ service.answer(question)        # Shared structured answer service
  │
  ├─▶ route(question)                 # Classify query intent
  │
  ├─▶ [stat_query]
  │     init_db()
  │     get_stat_leaders_range(...)    # DuckDB query -> top 10 leaders
  │       — or —
  │     get_career_stat_leaders(...)   # All-time career leaders
  │
  └─▶ [general_explanation]
        retrieve(question, top_k=3)    # ChromaDB semantic search
        build_explanation_prompt()      # Format with context docs
        make_request(prompt)            # LLM (Gemma via LM Studio)
        render_text(...)                # CLI-only text rendering
```

### Error Handling

| Condition | Behavior |
|-----------|----------|
| No year in query | Returns career leaders instead of season leaders |
| Player-specific stat not found | Returns a no-result message instead of switching to league leaders |
| Corpus not indexed | Returns: `No corpus indexed yet - run: uv run python -m baseball_rag.corpus.ingest` |
| Corpus index embedding mismatch | Returns a rebuild-index message |
| LM Studio offline | Falls back to printing retrieved document text (no LLM call) |
| DuckDB uninitialized | Auto-initializes on first stat query via `init_db()` |

The CLI intentionally does not ask the model to answer from pretraining when no
grounding evidence is available.

### Help Text

```bash
uv run python -m baseball_rag.cli --help
```

Prints:

```
Baseball RAG Query Engine
Usage: baseball-rag 'your question'

Examples:
  baseball-rag 'who had the most RBIs in 1962'
  baseball-rag 'career home run leaders'
```
