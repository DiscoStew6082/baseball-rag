"""Shared grounded answer service used by CLI and API."""

from __future__ import annotations

import logging
from typing import Any

from baseball_rag.db import (
    get_career_stat_leaders,
    get_player_stat,
    get_stat_leaders_range,
    init_db,
)
from baseball_rag.db.duckdb_schema import get_duckdb
from baseball_rag.provenance import SourceRecord, StructuredAnswer, compact_data_manifest
from baseball_rag.retrieval.chroma_store import RetrievedChunk, retrieve
from baseball_rag.routing import route
from baseball_rag.routing.query_router import TimePeriod, TimePeriodType

logger = logging.getLogger(__name__)


def answer(question: str) -> StructuredAnswer:
    """Answer a question with explicit grounding metadata."""
    init_db()
    decision = route(question)

    if decision.intent == "stat_query":
        return _answer_stat_query(question, decision)
    if decision.intent == "player_biography":
        return _answer_player_biography(question, decision)
    if decision.intent == "freeform_query":
        return _answer_freeform(question, decision)
    return _answer_general(question, decision)


def render_text(result: StructuredAnswer) -> str:
    """Render a structured answer for terminal/chat use."""
    lines = [result.answer]
    if result.warnings:
        lines.append("")
        lines.extend(f"Warning: {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _answer_stat_query(question: str, decision: Any) -> StructuredAnswer:
    stat = decision.stat or "HR"
    tp = _resolve_time_period(decision.time_period)

    if decision.player_name:
        conn = get_duckdb()
        result = get_player_stat(conn, decision.player_name, stat, year=decision.year)
        if not result:
            qualifier = f" in {decision.year}" if decision.year else ""
            return StructuredAnswer(
                answer=(
                    f"No {stat} result found for {decision.player_name}{qualifier} "
                    "in the local Lahman-derived batting data."
                ),
                intent=decision.intent,
                sources=[_duckdb_source("Player stat lookup", tables=["batting", "people"])],
                warnings=[
                    "No fallback leaderboard was returned because the question named a player."
                ],
                unsupported=True,
            )

        team_str = f" ({result['team']})" if result["team"] else ""
        return StructuredAnswer(
            answer=f"{result['name']}{team_str} ({result['year']}): {result['stat_value']} {stat}",
            intent=decision.intent,
            sources=[
                _duckdb_source(
                    "Player stat lookup",
                    tables=["batting", "people"],
                    rows=[result],
                )
            ],
        )

    if tp is not None:
        start_year, end_year = tp
        rows = get_stat_leaders_range(stat, start_year, end_year)
        lines = [f"Top {stat} leaders ({start_year}-{end_year}):"]
        for i, row in enumerate(rows[:10], 1):
            lines.append(f"  {i}. {row['name']}: {row['stat_value']} {stat}")
        return StructuredAnswer(
            answer="\n".join(lines),
            intent=decision.intent,
            sources=[
                _duckdb_source(
                    f"{stat} leaderboard for {start_year}-{end_year}",
                    tables=["batting", "people"],
                    rows=rows,
                )
            ],
        )

    rows = get_career_stat_leaders(stat)
    lines = [f"All-time career {stat} leaders:"]
    for i, row in enumerate(rows[:10], 1):
        lines.append(f"  {i}. {row['name']}: {row['stat_value']} {stat}")
    return StructuredAnswer(
        answer="\n".join(lines),
        intent=decision.intent,
        sources=[
            _duckdb_source(
                f"Career {stat} leaderboard",
                tables=["batting", "people"],
                rows=rows,
            )
        ],
    )


def _answer_player_biography(question: str, decision: Any) -> StructuredAnswer:
    try:
        chunks = retrieve(decision.player_name or question, top_k=3)
    except Exception as e:  # noqa: BLE001 - Chroma errors vary by installed version
        if "NotFoundError" in type(e).__name__ or "not found" in str(e).lower():
            return StructuredAnswer(
                answer="No corpus indexed yet - run: uv run python -m baseball_rag.corpus.ingest",
                intent=decision.intent,
                warnings=["Chroma collection was not available."],
                unsupported=True,
            )
        if _is_recoverable_chroma_index_error(e):
            return StructuredAnswer(
                answer=(
                    "The indexed corpus could not be queried. Rebuild it with: "
                    "uv run python -m baseball_rag.corpus.ingest"
                ),
                intent=decision.intent,
                warnings=[str(e)],
                unsupported=True,
            )
        logger.exception("ChromaDB retrieval failed for player biography query %r", question)
        raise

    if not chunks:
        return StructuredAnswer(
            answer=(
                f"No player biography found for '{decision.player_name}'. "
                "The player may not be in the corpus or the corpus may need re-indexing."
            ),
            intent=decision.intent,
            warnings=["No LLM fallback was used because no grounding context was retrieved."],
            unsupported=True,
        )

    from baseball_rag.generation.prompt import build_player_bio_prompt

    prompt = build_player_bio_prompt(decision.raw_question, chunks)
    sources = [_chroma_source(chunk) for chunk in chunks]
    try:
        from baseball_rag.generation.llm import make_request

        response = make_request(prompt, max_tokens=1500)
        return StructuredAnswer(answer=response.content, intent=decision.intent, sources=sources)
    except ConnectionError:
        lines = ["(LM Studio not running - showing relevant documents instead):\n"]
        for chunk in chunks[:3]:
            lines.append(f"[{chunk.title}]\n{chunk.text}\n")
        return StructuredAnswer(
            answer="\n".join(lines),
            intent=decision.intent,
            sources=sources,
            warnings=["LM Studio was unavailable, so retrieved context was shown directly."],
        )


def _answer_freeform(question: str, decision: Any) -> StructuredAnswer:
    from baseball_rag.db.freeform import format_result, query

    conn = get_duckdb()
    query_result = query(decision.raw_question, conn)
    source = SourceRecord(
        type="duckdb",
        label="Freeform SQL query",
        detail="Natural-language intent converted to a constrained SQL query.",
        sql=query_result.sql,
        columns=query_result.columns,
        rows=_rows_to_dicts(query_result.columns, query_result.rows[:100]),
        data_manifest=compact_data_manifest(),
    )

    if query_result.row_count == 0:
        return StructuredAnswer(
            answer=(
                f"No results found for '{decision.raw_question}'.\n"
                "Try rephrasing with a specific team, player, stat, or year."
            ),
            intent=decision.intent,
            sources=[source],
            unsupported=True,
        )

    warnings = []
    if query_result.truncated:
        warnings.append("Results were truncated at the configured row limit.")
    return StructuredAnswer(
        answer=format_result(query_result, decision.raw_question),
        intent=decision.intent,
        sources=[source],
        warnings=warnings,
    )


def _answer_general(question: str, decision: Any) -> StructuredAnswer:
    try:
        chunks = retrieve(question, top_k=3)
    except Exception as e:  # noqa: BLE001 - Chroma errors vary by installed version
        if "NotFoundError" in type(e).__name__ or "not found" in str(e).lower():
            return StructuredAnswer(
                answer="No corpus indexed yet - run: uv run python -m baseball_rag.corpus.ingest",
                intent=decision.intent,
                warnings=["Chroma collection was not available."],
                unsupported=True,
            )
        if _is_recoverable_chroma_index_error(e):
            return StructuredAnswer(
                answer=(
                    "The indexed corpus could not be queried. Rebuild it with: "
                    "uv run python -m baseball_rag.corpus.ingest"
                ),
                intent=decision.intent,
                warnings=[str(e)],
                unsupported=True,
            )
        logger.exception("ChromaDB retrieval failed for query %r", question)
        raise

    if not chunks:
        return StructuredAnswer(
            answer=(
                "No relevant grounded documents were found for that query. "
                "Try asking about an indexed stat definition, indexed player biography, "
                "or a database-backed statistic."
            ),
            intent=decision.intent,
            warnings=["No LLM fallback was used because no grounding context was retrieved."],
            unsupported=True,
        )

    from baseball_rag.generation.prompt import build_explanation_prompt

    prompt = build_explanation_prompt(question, chunks)
    sources = [_chroma_source(chunk) for chunk in chunks]
    try:
        from baseball_rag.generation.llm import make_request

        response = make_request(prompt, max_tokens=1500)
        return StructuredAnswer(answer=response.content, intent=decision.intent, sources=sources)
    except ConnectionError:
        lines = ["(LM Studio not running - showing relevant documents instead):\n"]
        for chunk in chunks[:3]:
            lines.append(f"[{chunk.title}]\n{chunk.text}\n")
        return StructuredAnswer(
            answer="\n".join(lines),
            intent=decision.intent,
            sources=sources,
            warnings=["LM Studio was unavailable, so retrieved context was shown directly."],
        )


def _resolve_time_period(tp: TimePeriod | None) -> tuple[int, int] | None:
    if tp is None:
        return None
    if tp.type == TimePeriodType.DECADE and isinstance(tp.value, int):
        start_year = 1900 + tp.value
        return start_year, start_year + 9
    if tp.type == TimePeriodType.RANGE and isinstance(tp.value, list) and len(tp.value) >= 2:
        return int(tp.value[0]), int(tp.value[-1])
    if tp.type == TimePeriodType.SINGLE and isinstance(tp.value, int):
        return tp.value, tp.value
    return None


def _duckdb_source(
    label: str,
    *,
    tables: list[str],
    rows: list[dict[str, Any]] | None = None,
) -> SourceRecord:
    return SourceRecord(
        type="duckdb",
        label=label,
        detail=f"Tables: {', '.join(tables)}. Dataset: local Hugging Face NeuML/baseballdata CSVs.",
        rows=rows or [],
        data_manifest=compact_data_manifest(),
    )


def _chroma_source(chunk: RetrievedChunk) -> SourceRecord:
    return SourceRecord(
        type="chroma",
        label=chunk.title,
        detail=chunk.source,
        rows=[{"text": chunk.text}],
        score=chunk.score,
    )


def _rows_to_dicts(columns: list[str], rows: list[tuple]) -> list[dict[str, Any]]:
    return [dict(zip(columns, row)) for row in rows]


def _is_recoverable_chroma_index_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "dimension" in message or "embedding" in message
