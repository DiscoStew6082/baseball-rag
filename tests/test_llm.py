"""Tests for LLM client — mocked since LM Studio may not be running."""
from unittest.mock import MagicMock, patch

import pytest

from baseball_rag.generation.llm import make_request, make_request_stream


class TestLLMClient:
    def test_generate_returns_text(self):
        """make_request returns an LLMResponse with non-empty content."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Mickey Mantle had 123 RBI in 1962."}}],
            "model": "gemma-4-26b",
        }

        with patch("requests.post", return_value=mock_resp):
            result = make_request("who had most RBIs in 1962")

        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert result.model == "gemma-4-26b"

    def test_connection_error_raises(self):
        """ConnectionError is raised when LM Studio is not running."""
        import requests

        with patch("requests.post", side_effect=requests.ConnectionError("connection refused")):
            with pytest.raises(ConnectionError, match="Could not connect"):
                make_request("test query")

    def test_stream_yields_chunks(self):
        """make_request_stream yields tokens as they arrive."""
        lines = [
            'data: {"choices":[{"delta":{"content":"Mickey"}}]}',
            'data: {"choices":[{"delta":{"content":" Mantle"}}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(lines)

        with patch("requests.post", return_value=mock_resp):
            tokens = list(make_request_stream("who had most RBIs"))

        assert "Mickey" in tokens
        # Note: token may include leading space from SSE delta
        assert any("Mantle" in t for t in tokens)
