"""Baseball RAG query engine — CLI entry point."""

import logging
import sys

from baseball_rag.db import (
    get_career_stat_leaders,
    get_player_stat,
    get_stat_leaders_range,
    init_db,
)
from baseball_rag.db.duckdb_schema import get_duckdb
from baseball_rag.generation.prompt import build_explanation_prompt, build_open_prompt
from baseball_rag.retrieval.chroma_store import retrieve
from baseball_rag.routing import route
from baseball_rag.routing.query_router import TimePeriodType

logger = logging.getLogger(__name__)


def answer(question: str) -> str:
    """Answer a single question using the full RAG pipeline.

    1. Route (keyword-based classifier)
    2a. For stat queries → execute SQL directly
    2b. For general questions → ChromaDB retrieval + LLM generation

    Args:
        question: Natural language baseball question.

    Returns:
        Answer string.
    """
    init_db()
    decision = route(question)

    if decision.intent == "stat_query":
        stat = decision.stat or "HR"
        tp = decision.time_period

        # ---- Resolve time_period to concrete [start_year, end_year] -----------------
        # The LLM only gives us the raw value; cli.py resolves it to a range so the
        # DB layer gets clean integers. This keeps calendar math out of the prompt.
        if tp is not None:
            if tp.type == TimePeriodType.DECADE and isinstance(tp.value, int):
                decade = tp.value  # e.g. 70 → 1970-1979
                start_year = 1900 + decade
                end_year = start_year + 9
                tp.resolved_start = start_year
                tp.resolved_end = end_year
            elif tp.type == TimePeriodType.RANGE and isinstance(tp.value, list):
                start_year, end_year = tp.value[0], tp.value[-1]
                tp.resolved_start = start_year
                tp.resolved_end = end_year
            elif tp.type == TimePeriodType.SINGLE and isinstance(tp.value, int):
                start_year = end_year = tp.value
                tp.resolved_start = start_year
                tp.resolved_end = end_year
            else:
                # relative or unparseable — degrade to career
                tp = None

        # ---- Player-specific query -------------------------------------------------
        if decision.player_name:
            conn = get_duckdb()
            result = get_player_stat(conn, decision.player_name, stat, year=decision.year)
            if result:
                team_str = f" ({result['team']})" if result["team"] else ""
                return (
                    f"{result['name']}{team_str} ({result['year']}): {result['stat_value']} {stat}"
                )
            # Player found but no stat for that year — fall through to leaders

        # ---- League-wide leaders ---------------------------------------------------
        if tp is not None and tp.resolved_start is not None:
            rows = get_stat_leaders_range(stat, start_year, end_year)
            lines = [f"Top {stat} leaders ({start_year}-{end_year}):"]
            for i, row in enumerate(rows[:10], 1):
                lines.append(f"  {i}. {row['name']}: {row['stat_value']} {stat}")
        else:
            rows = get_career_stat_leaders(stat)
            lines = [f"All-time career {stat} leaders:"]
            for i, row in enumerate(rows[:10], 1):
                lines.append(f"  {i}. {row['name']}: {row['stat_value']} {stat}")

        return "\n".join(lines)

    elif decision.intent == "player_biography":
        # Player biography: retrieve bio doc from ChromaDB + generate with bio-specific prompt
        try:
            chunks = retrieve(decision.player_name or question, top_k=3)
        except Exception as e:  # noqa: BLE001 — chromadb.errors.NotFoundError not always importable
            if "NotFoundError" in type(e).__name__ or "not found" in str(e).lower():
                return "(No corpus indexed yet — run: uv run python -m baseball_rag.corpus.ingest)"
            logger.exception("ChromaDB retrieval failed for player biography query %r", question)
            raise

        if not chunks:
            return (
                f"No player biography found for '{decision.player_name}'. "
                f"The player may not be in the dataset or the corpus hasn't been indexed yet."
            )

        from baseball_rag.generation.prompt import build_player_bio_prompt

        prompt = build_player_bio_prompt(decision.raw_question, chunks)

        try:
            from baseball_rag.generation.llm import make_request

            response = make_request(prompt, max_tokens=1500)
            return response.content
        except ConnectionError:
            # LM Studio not running — fall back to showing retrieved docs
            lines = ["(LM Studio not running - showing relevant documents instead):\n"]
            for chunk in chunks[:3]:
                lines.append(f"[{chunk.title}]\n{chunk.text}\n")
            return "\n".join(lines)

    elif decision.intent == "freeform_query":
        from baseball_rag.db.freeform import format_result, query

        conn = get_duckdb()
        query_result = query(decision.raw_question, conn)

        # If empty, try to be helpful before giving up entirely
        if query_result.row_count == 0:
            return (
                f"No results found for '{decision.raw_question}'.\n"
                "Try rephrasing — e.g., 'who played for the Boston Braves in 1936'\n"
                "(the Braves were in Boston until 1965, then Atlanta from 1966)."
            )

        return format_result(query_result, decision.raw_question)

    else:
        # General explanation: RAG retrieval + generation (fallback when LLM unavailable)
        try:
            chunks = retrieve(question, top_k=3)
        except Exception as e:  # noqa: BLE001 — chromadb.errors.NotFoundError not always importable
            if "NotFoundError" in type(e).__name__ or "not found" in str(e).lower():
                return "(No corpus indexed yet — run: uv run python -m baseball_rag.corpus.ingest)"
            logger.exception("ChromaDB retrieval failed for query %r", question)
            raise

        if not chunks:
            from baseball_rag.retrieval.chroma_store import corpus_diagnostics

            diag = corpus_diagnostics()

            stat_count = len(diag["corpus_files"]["stat_definitions"])
            hof_count = len(diag["corpus_files"]["hof_bios"])
            stat_list = ", ".join(sorted(diag["corpus_files"]["stat_definitions"]))
            hof_list = ", ".join(sorted(diag["corpus_files"]["hof_bios"]))

            lines = [
                "No relevant documents found in the corpus for that query.",
                "",
                "Available sources:",
                f"  - Stat definitions ({stat_count}): {stat_list}",
                f"  - Hall of Fame biographies ({hof_count}): {hof_list}",
            ]

            if diag["chroma_collection"]["indexed_count"] == 0:
                lines.append("")
                lines.append(
                    "Note: ChromaDB index is empty. "
                    "Run: uv run python -m baseball_rag.corpus.ingest"
                )

            # Try LLM without retrieved context — it may know this from training
            try:
                from baseball_rag.generation.llm import make_request

                prompt = build_open_prompt(decision.raw_question)
                response = make_request(prompt, max_tokens=1500)
                return response.content
            except ConnectionError:
                pass  # LM Studio not running; fall through to return the "no docs" message

            lines.extend(
                [
                    "",
                    "For player stats, try asking directly:",
                    '  baseball-rag "how many HRs did Aaron Judge have in 2024"',
                    '  baseball-rag "who led MLB in RBIs in 2022"',
                ]
            )
            return "\n".join(lines)

        prompt = build_explanation_prompt(question, chunks)

        try:
            from baseball_rag.generation.llm import make_request

            response = make_request(prompt, max_tokens=1500)
            return response.content
        except ConnectionError:
            # LM Studio not running — fall back to showing retrieved docs
            lines = ["(LM Studio not running - showing relevant documents instead):\n"]
            for chunk in chunks[:3]:
                lines.append(f"[{chunk.title}]\n{chunk.text}\n")
            return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print(
            "Baseball RAG Query Engine\n"
            "Usage: baseball-rag 'your question'\n\n"
            "Examples:\n"
            "  baseball-rag 'who had the most RBIs in 1962'\n"
            "  baseball-rag 'career home run leaders'\n"
        )
        sys.exit(0)

    question = " ".join(sys.argv[1:])
    result = answer(question)
    print(result)


if __name__ == "__main__":
    main()
