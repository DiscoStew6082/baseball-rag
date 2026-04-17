"""Tests for FastAPI server — Phases 7.1 and 7.2."""
from fastapi.testclient import TestClient

from baseball_rag.api.server import app

client = TestClient(app)

class TestApi:
    def test_health_endpoint(self):
        """GET /health returns 200 with {"status": "ok"}."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_query_endpoint_returns_answer(self):
        """POST /query with JSON body returns {answer: str, sources: list}."""
        # Note: This will call the real cli.answer(). If ChromaDB isn't indexed,
        # it returns a fallback message — that's fine.
        response = client.post("/query", json={"question": "who had most RBIs in 1962"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert "sources" in data
        assert isinstance(data["sources"], list)

