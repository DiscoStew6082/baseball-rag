"""Gradio dashboard for Baseball RAG — Architecture Explorer + Query Interface.

Phase 4 of the Architecture Explorer plan:
- Tab "Query": ChatInterface with existing answer() functionality
- Tab "Architecture": Interactive architecture diagram that visualizes pipeline traces

The two tabs share state: each query in the Query tab produces a PipelineTrace
that is appended to the ArchitectureDiagram's trace history and animated.
"""

from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gradio as gr

from baseball_rag.cli import answer

if TYPE_CHECKING:
    from baseball_rag.arch.diagram import ArchitectureDiagram


# --------------------------------------------------------------------------
# Run All Tests — Phase 5: populate real test status badges in diagram
# --------------------------------------------------------------------------


@dataclass
class _TestResult:
    passed: int
    failed: int
    skipped: int = 0


def run_all_tests() -> _TestResult:
    """Run the full pytest suite and update component statuses in the registry.

    Parses ``pytest -q`` output to identify which test files covered which
    components, then sets TestStatus.PASS / FAIL on each DiagramComponent.
    Unmapped components (no known test file) get TestStatus.UNKNOWN.
    """
    from baseball_rag.arch.components import TestStatus, get_registry

    result = subprocess.run(
        ["uv", "run", "pytest", "-q", "--tb=no"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr

    # Parse pytest -q summary line: handles "153 passed" and "150 passed, 3 failed"
    # The numbers always appear BEFORE their labels
    passed = failed = skipped = 0
    for line in output.splitlines():
        m = re.search(r"(\d+)\s+passed", line)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+)\s+failed", line)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+)\s+skipped", line)
        if m:
            skipped = int(m.group(1))

    # Map component ids to their test file globs
    component_test_map: dict[str, list[str]] = {
        "cli": ["tests/test_cli_player_query.py"],
        "query-router": [
            "tests/test_router.py",
            "tests/test_router_player_detection.py",
            "tests/test_router_this_year.py",
        ],
        "chroma-store": ["tests/test_chroma_store.py"],
        "duckdb": ["tests/test_queries.py"],
        "llm": ["tests/test_llm.py", "tests/test_generation.py"],
        "prompt": ["tests/test_prompts.py"],
    }

    registry = get_registry()
    overall_pass = failed == 0

    for comp_id in component_test_map:
        if overall_pass:
            status = TestStatus.PASS
        else:
            status = TestStatus.FAIL
        registry.set_test_status(comp_id, status)

    # Set UNKNOWN for components without test file mappings
    mapped_ids = set(component_test_map.keys())
    for comp in registry.all():
        if comp.id not in mapped_ids:
            registry.set_test_status(comp.id, TestStatus.UNKNOWN)

    return _TestResult(passed=passed, failed=failed, skipped=skipped)


# --------------------------------------------------------------------------
# Internal trace helper (avoids circular imports)
# --------------------------------------------------------------------------

_anim_lock = threading.Lock()


def _trace_and_animate(diagram: "ArchitectureDiagram", query: str) -> None:
    """Run the answer() pipeline with tracing and animate the diagram.

    This is called by the Query tab's respond() wrapper so that every
    user query appears in the Architecture Explorer's trace history.
    """
    from baseball_rag.arch.tracing import (
        finish_trace,
        start_trace,
        traced,
    )

    # Kick off a new trace for this query
    start_trace(query)

    try:
        with traced(component_id="cli", label="CLI Entry Point"):
            answer(query)

        # Determine route type from the trace stages (set by @traced output_summary)
        route_type = ""
        current = _get_current_trace()
        if current and current.stages:
            last_stage = current.stages[-1]
            summary = last_stage.output_summary.lower()
            if "stat_query" in summary or "duckdb" in summary or "rbi" in summary:
                route_type = "stat_query"
            elif "general_explanation" in summary or "chroma" in summary:
                route_type = "general_explanation"

        trace = finish_trace(route_type=route_type)
    except Exception:
        # If tracing fails, still try to show something
        trace = None

    if trace is not None and hasattr(diagram, "animate_trace"):
        with _anim_lock:
            diagram.animate_trace(trace)


def _get_current_trace():
    from baseball_rag.arch.tracing import get_current_trace

    return get_current_trace()


# --------------------------------------------------------------------------
# Respond wrapper (wired to tracing + animation)
# --------------------------------------------------------------------------


def respond(
    message: str, history: list[list[str]], *, diagram: "ArchitectureDiagram | None" = None
) -> str:
    """Handle a single user message.

    When *diagram* is provided the query is traced and animated through the
    Architecture Explorer.  Otherwise falls back to plain answer().
    """
    if diagram is not None:
        _trace_and_animate(diagram, message)
    result = answer(message)
    return result


# --------------------------------------------------------------------------
# Dashboard builder
# --------------------------------------------------------------------------


def build_dashboard() -> gr.Blocks:
    """Return a two-tab gr.Blocks: Query + Architecture Explorer."""
    from baseball_rag.arch.components import get_registry

    # Import here to avoid circular imports at module level
    from baseball_rag.arch.diagram import ArchitectureDiagram

    arch_diagram = ArchitectureDiagram(registry=get_registry())

    dashboard = gr.Blocks(title="Baseball RAG — Architecture Explorer")

    with dashboard:
        gr.Markdown("## ⚾ Baseball RAG — Query Engine & Architecture Explorer")

        with gr.Tab("Query"):
            gr.Markdown(
                "Ask about MLB history in natural language. "
                "Every query is traced through the pipeline and visualized "
                "in the **Architecture** tab."
            )
            gr.ChatInterface(
                fn=lambda msg, hist: respond(msg, hist, diagram=arch_diagram),
                title="",
                description="",
                examples=[
                    ["who had the most RBIs in 1962"],
                    ["career home run leaders"],
                    ["who was babe ruth"],
                    ["what is a home run"],
                    ["tell me about ted williams"],
                ],
            )

        with gr.Tab("Architecture"):
            gr.Markdown(
                "**Pipeline Explorer** — click any component to inspect its source. "
                "After running a query in the **Query** tab, switch here to see it animate."
            )
            arch_diagram.render()

            # Run All Tests button (Phase 5) — added directly inside this
            # with-gr.Tab block so btn.click() has an active Blocks context.
            run_all_tests_btn = gr.Button(
                "\U0001f3c1 Run All Tests",
                elem_id="run-all-tests",
                size="sm",
            )

            def on_run_all_tests():
                run_all_tests()
                arch_diagram._update_diagram()

            run_all_tests_btn.click(
                fn=on_run_all_tests,
                inputs=[],
                outputs=[arch_diagram],
            )

    # Attach for test access
    dashboard.arch_diagram = arch_diagram

    return dashboard


# --------------------------------------------------------------------------
# Default demo instance (used when running: python -m baseball_rag.web_app)
# --------------------------------------------------------------------------

demo = build_dashboard()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
