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

Ask a baseball question and get a grounded answer with provenance metadata.

**Request**

```json
{
  "question": "who had the most RBIs in 1962"
}
```

**Response**

```json
{
  "answer": "Top RBI leaders (1962-1962):\n  1. Tommy Davis: 153 RBI\n  2. ...",
  "intent": "stat_query",
  "sources": [
    {
      "type": "duckdb",
      "label": "RBI leaderboard for 1962-1962",
      "detail": "Tables: batting, people. Dataset: local Hugging Face NeuML/baseballdata CSVs.",
      "sql": null,
      "rows": [
        { "name": "Davis, Tommy", "team": "Range", "stat_value": 153 }
      ],
      "columns": [],
      "score": null
    }
  ],
  "warnings": [],
  "unsupported": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `question` | string | Natural language baseball question |

**Response fields**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Full answer text (formatted list for stat queries, prose for general questions) |
| `intent` | string | Router intent used to answer the question |
| `sources` | array | DuckDB/Chroma evidence records used to ground the answer |
| `warnings` | array | Non-fatal caveats, such as missing indexes or truncated results |
| `unsupported` | boolean | True when the system could not answer from grounded evidence |

## Error Responses

| Status | Condition |
|--------|-----------|
| 422 Unprocessable Entity | Missing or invalid request body |
| 500 Internal Server Error | Unexpected DuckDB, ChromaDB, or server error |

## Architecture Note

The `/query` endpoint calls the shared answer service. The CLI renders the same
structured answer as text, while the API returns the full JSON payload:

1. **Stat query** → DuckDB lookup (same as CLI)
2. **General question** → ChromaDB retrieval + LLM generation

This means API and CLI behave identically for the same input.

## Development

The server is intentionally minimal — it reuses the CLI logic rather than duplicating it, keeping the API and command-line interface consistent.
