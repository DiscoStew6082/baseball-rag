"""Tests for Gradio web app — Phase 4 dashboard integration."""

from baseball_rag import web_app


class TestGradio:
    def test_demo_is_blocks(self):
        """gr.Blocks dashboard exists and has arch_diagram attached."""
        assert hasattr(web_app, "demo")
        from gradio import Blocks

        assert isinstance(web_app.demo, Blocks)
        assert hasattr(web_app.demo, "arch_diagram")

    def test_build_dashboard_returns_blocks(self):
        """build_dashboard() returns a gr.Blocks with an arch_diagram."""
        demo = web_app.build_dashboard()
        from gradio import Blocks

        assert isinstance(demo, Blocks)
        assert hasattr(demo, "arch_diagram")

    def test_respond_without_diagram(self):
        """respond() works without a diagram (passthrough to answer())."""
        result = web_app.respond("who had the most HRs in 1970", [], diagram=None)
        assert isinstance(result, str)
        assert len(result) > 0
