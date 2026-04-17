"""Embedder client using LM Studio's embedding endpoint."""
import os

import requests

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b"
EMBEDDING_ENDPOINT = "/embeddings"

def embed(
    text: str,
    base_url: str | None = None,
    model: str | None = None,
) -> list[float]:
    """Embed a single text string using LM Studio's embedding endpoint."""
    base_url = base_url or os.environ.get("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL)
    model = model or os.environ.get("LMSTUDIO_MODEL", DEFAULT_MODEL)

    payload = {"model": model, "input": text}

    try:
        resp = requests.post(f"{base_url}{EMBEDDING_ENDPOINT}", json=payload, timeout=30)
        resp.raise_for_status()
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Could not connect to LM Studio at {base_url}. "
            "Is the server running? (LM Studio → Server tab → Start server)"
        ) from exc

    data = resp.json()
    # LM Studio may return ints mixed with floats; normalize to float
    return [float(x) for x in data["data"][0]["embedding"]]
