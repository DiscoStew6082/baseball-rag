"""Tests for pipeline tracing — Phase 2."""

from datetime import datetime

import pytest

from baseball_rag.arch.tracing import (
    PipelineStage,
    PipelineTrace,
    finish_trace,
    get_current_trace,
    start_trace,
    traced,
)


class TestPipelineStage:
    def test_stage_records_timing(self):
        """PipelineStage records component_id, label, started_at, elapsed_ms."""
        before = datetime.now()
        stage = PipelineStage(
            component_id="query-router",
            label="Query Router",
            started_at=before,
            elapsed_ms=12.5,
        )
        assert stage.component_id == "query-router"
        assert stage.label == "Query Router"
        assert stage.started_at == before
        assert stage.elapsed_ms == 12.5

    def test_stage_output_summary(self):
        """Stage has an output_summary field."""
        stage = PipelineStage(
            component_id="chroma-store",
            label="ChromaDB Query",
            started_at=datetime.now(),
            elapsed_ms=8.0,
            output_summary="retrieved 3 chunks about Babe Ruth",
        )
        assert stage.output_summary == "retrieved 3 chunks about Babe Ruth"

    def test_stage_error_state(self):
        """Stage can represent error state."""
        stage = PipelineStage(
            component_id="llm",
            label="LLM Generation",
            started_at=datetime.now(),
            elapsed_ms=0.0,
            error="ConnectionError: LM Studio unreachable",
        )
        assert stage.is_success is False
        assert "ConnectionError" in stage.error

    def test_stage_success_state(self):
        """Stage with no error is considered successful."""
        stage = PipelineStage(
            component_id="duckdb",
            label="DuckDB Query",
            started_at=datetime.now(),
            elapsed_ms=3.2,
        )
        assert stage.is_success is True


class TestPipelineTrace:
    def test_trace_stores_query(self):
        """PipelineTrace.query stores the original query string."""
        trace = PipelineTrace(query="who had the most RBIs in 1962")
        assert trace.query == "who had the most RBIs in 1962"

    def test_trace_assembles_stages_in_order(self):
        """stages list is ordered by insertion (execution order)."""
        t = PipelineTrace(query="test")
        s1 = PipelineStage("r1", "Stage 1", datetime.now(), 1.0)
        s2 = PipelineStage("r2", "Stage 2", datetime.now(), 2.0)
        t.add_stage(s1)
        t.add_stage(s2)
        assert [s.component_id for s in t.stages] == ["r1", "r2"]

    def test_trace_total_ms_from_stages(self):
        """total_ms equals sum of stage elapsed_ms values."""
        trace = PipelineTrace(query="test")
        trace.add_stage(PipelineStage("a", "A", datetime.now(), 10.0))
        trace.add_stage(PipelineStage("b", "B", datetime.now(), 5.5))
        assert trace.total_ms == 15.5

    def test_trace_is_complete_when_stages_exist(self):
        """is_complete is True once at least one stage exists."""
        t = PipelineTrace(query="test")
        assert t.is_complete is False
        t.add_stage(PipelineStage("x", "X", datetime.now(), 1.0))
        assert t.is_complete is True


class TestRouteType:
    def test_route_type_recorded(self):
        """finish_trace records the route_type on the trace."""
        start_trace("who was babe ruth")
        # simulate a stage
        with traced(component_id="router", label="Router"):
            pass
        trace = finish_trace(route_type="general_explanation")
        assert trace.route_type == "general_explanation"


class TestTracedContextManager:
    def test_traced_records_stage(self):
        """@traced context manager records a stage on the current trace."""
        start_trace("test query")
        with traced(component_id="chroma-store", label="ChromaDB Query"):
            pass
        trace = finish_trace()
        assert len(trace.stages) == 1
        assert trace.stages[0].component_id == "chroma-store"
        assert trace.stages[0].label == "ChromaDB Query"

    def test_traced_records_elapsed_ms(self):
        """Elapsed time is recorded accurately."""
        start_trace("test")
        with traced(component_id="duckdb", label="DuckDB"):
            pass
        trace = finish_trace()
        assert trace.stages[0].elapsed_ms > 0

    def test_traced_captures_output_summary(self):
        """output_summary passed to traced appears in the stage."""
        start_trace("test")
        with traced(component_id="router", label="Router", output_summary="routed to stat_query"):
            pass
        trace = finish_trace()
        assert "stat_query" in trace.stages[0].output_summary

    def test_traced_captures_error_on_exception(self):
        """Exception is recorded as stage error and re-raised."""
        start_trace("test")
        with pytest.raises(ValueError, match="boom"):
            with traced(component_id="llm", label="LLM"):
                raise ValueError("boom")
        trace = finish_trace()
        assert "ValueError" in trace.stages[0].error
        assert "boom" in trace.stages[0].error

    def test_nested_traces_work(self):
        """Nested traced blocks produce multiple stages."""
        start_trace("outer query")
        with traced(component_id="outer", label="Outer"):
            with traced(component_id="inner", label="Inner"):
                pass
        trace = finish_trace()
        assert len(trace.stages) == 2
        assert [s.component_id for s in trace.stages] == ["outer", "inner"]

    def test_disable_tracing_env_var(self, monkeypatch):
        """DISABLE_TRACING=1 skips tracing."""
        monkeypatch.setenv("DISABLE_TRACING", "1")
        # Re-import to pick up env var (module caches it at import time)
        import importlib

        from baseball_rag.arch import tracing as tracing_mod

        importlib.reload(tracing_mod)

        tracing_mod.start_trace("test")
        with tracing_mod.traced(component_id="should-not-appear", label="Skipped"):
            pass
        trace = tracing_mod.finish_trace()
        assert len(trace.stages) == 0

        monkeypatch.delenv("DISABLE_TRACING", raising=False)
        importlib.reload(tracing_mod)


class TestTraceLifecycle:
    def test_start_and_finish_trace(self):
        """start_trace creates a new current trace; finish_trace returns and clears it."""
        start_trace("who was babe ruth")
        assert get_current_trace() is not None
        finished = finish_trace(route_type="general_explanation")
        assert finished.query == "who was babe ruth"
        assert finished.route_type == "general_explanation"
        assert get_current_trace() is None

    def test_finish_with_no_active_trace_returns_none(self):
        """finish_trace returns None when no trace is active."""
        result = finish_trace()
        assert result is None
