"""Tests for Gradio web app — Phase 7.4."""
from baseball_rag import web_app

class TestGradio:
    def test_gradio_interface_loads(self):
        """gradio ChatInterface demo exists and is correct type."""
        assert hasattr(web_app, "demo")
        from gradio import ChatInterface
        assert isinstance(web_app.demo, ChatInterface)