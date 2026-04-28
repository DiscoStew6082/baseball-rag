"""Tests for LM Studio embedding client configuration."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from baseball_rag.embedder import DEFAULT_EMBEDDING_MODEL, embed


def test_embedder_prefers_embedding_model_env_var(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_MODEL", "chat-model")
    monkeypatch.setenv("LMSTUDIO_EMBEDDING_MODEL", "embedding-model")
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"embedding": [1, 2.5]}]}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = embed("Babe Ruth")

    assert result == [1.0, 2.5]
    assert mock_post.call_args.kwargs["json"]["model"] == "embedding-model"


def test_embedder_default_model_is_embedding_specific(monkeypatch):
    monkeypatch.delenv("LMSTUDIO_MODEL", raising=False)
    monkeypatch.delenv("LMSTUDIO_EMBEDDING_MODEL", raising=False)
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"embedding": [0.25]}]}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        embed("OPS")

    assert mock_post.call_args.kwargs["json"]["model"] == DEFAULT_EMBEDDING_MODEL


def test_embedder_http_error_names_embedding_model(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_EMBEDDING_MODEL", "chat-only-model")
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("400 Bad Request")

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="LMSTUDIO_EMBEDDING_MODEL"):
            embed("OPS")
