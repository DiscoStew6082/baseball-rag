"""LLM integration — calls local Gemma via LM Studio."""

import os
from dataclasses import dataclass
from typing import Iterator

import requests

from baseball_rag.arch.tracing import traced


@dataclass
class LLMResponse:
    content: str
    model: str
    done: bool


DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b"


@traced(component_id="llm", label="Generate Answer")
def make_request(
    prompt: str,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> LLMResponse:
    """Send a chat-style prompt to LM Studio and return the response.

    Args:
        prompt: The full rendered prompt (system + user content).
        base_url: LM Studio server URL. Defaults to localhost:1234.
        model: Model name to specify in request.
        max_tokens: Max new tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        LLMResponse with the generated text.

    Raises:
        ConnectionError: If LM Studio is not running at the given URL.
    """
    base_url = base_url or os.environ.get("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL)
    model = model or os.environ.get("LMSTUDIO_MODEL", DEFAULT_MODEL)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    try:
        resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Could not connect to LM Studio at {base_url}. "
            "Is the server running? (Start LM Studio → Server tab → Start server)"
        ) from exc

    data = resp.json()
    choice = data["choices"][0]["message"]
    # Gemma 4 may put response text in reasoning_content instead of content
    return LLMResponse(
        content=choice.get("content") or choice.get("reasoning_content", ""),
        model=data.get("model", model),
        done=True,
    )


def make_request_stream(
    prompt: str,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> Iterator[str]:
    """Streaming version — yields content tokens as they arrive."""
    base_url = base_url or os.environ.get("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL)
    model = model or os.environ.get("LMSTUDIO_MODEL", DEFAULT_MODEL)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    try:
        resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.ConnectionError as exc:
        raise ConnectionError(f"Could not connect to LM Studio at {base_url}.") from exc

    for line in resp.iter_lines(decode_unicode=True):
        if not line or line == "data: [DONE]":
            break
        if line.startswith("data: "):
            import json as _json

            chunk = _json.loads(line[6:])
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            # Gemma 4 puts response in reasoning_content, not content
            token = delta.get("content") or delta.get("reasoning_content", "")
            if token:
                yield token
