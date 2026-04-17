# API Reference

FastAPI server exposing the RAG pipeline over HTTP.

## Start the Server

```bash
uv run uvicorn baseball_rag.api.server:app --reload
```

Default port: **8000** (--reload enables auto-reload on code changes).

## Endpoints

### `GET /health`

Health check. No authentication required.

**Response**
```json
{ "status": "ok" }
```

---

### `POST /query`

Ask a baseball question and get an answer.

**Request**

```json
{
  "question": "who had the most RBIs in 1962"
}
```

**Response**

```json
{
  "answer": "Top RBI leaders for 1962:\n  1. Mickey Mantle (New York Yankees): 123 RBI\n  2. ...",
  "sources": []
}
```

| Field | Type | Description |
|-------|------|-------------|
| `question` | string | Natural language baseball question |

**Response fields**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Full answer text (formatted list for stat queries, prose for general questions) |
| `sources` | array | Placeholder — reserved for future citation metadata |

## Error Responses

| Status | Condition |
|--------|-----------|
| 422 Unprocessable Entity | Missing or invalid request body |
| 500 Internal Server Error | LM Studio offline, corpus not indexed, DuckDB error |

## Architecture Note

The `/query` endpoint delegates directly to `cli.answer()`, which runs the full pipeline:

1. **Stat query** → DuckDB lookup (same as CLI)
2. **General question** → ChromaDB retrieval + LLM generation

This means API and CLI behave identically for the same input.

## Development

The server is intentionally minimal — it reuses the CLI logic rather than duplicating it, keeping the API and command-line interface consistent.
