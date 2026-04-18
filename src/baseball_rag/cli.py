"""Baseball RAG query engine — CLI entry point."""

import logging
import sys

from baseball_rag.db import get_career_stat_leaders, get_player_stat, get_stat_leaders, init_db
from baseball_rag.generation.prompt import build_explanation_prompt
from baseball_rag.retrieval.chroma_store import retrieve
from baseball_rag.routing import route

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
        year = decision.year

        # If player name detected but no explicit year → get their latest-year stats
        if decision.player_name and not year:
            from baseball_rag.db.duckdb_schema import get_duckdb

            conn = get_duckdb()
            result = get_player_stat(conn, decision.player_name, stat)
            if result:
                team_str = f" ({result['team']})" if result["team"] else ""
                return (
                    f"{result['name']}{team_str} ({result['year']}): {result['stat_value']} {stat}"
                )

        if year:
            rows = get_stat_leaders(stat, year)
            if not rows and decision.player_name:
                # Requested year had no results for this player — show their latest
                from baseball_rag.db.duckdb_schema import get_duckdb

                conn = get_duckdb()
                result = get_player_stat(conn, decision.player_name, stat)
                if result:
                    team_str = f" ({result['team']})" if result["team"] else ""
                    return (
                        f"{result['name']}{team_str} "
                        f"({result['year']}): {result['stat_value']} {stat}"
                    )
            lines = [f"Top {stat} leaders for {year}:"]
            for i, row in enumerate(rows[:10], 1):
                team_str = f" ({row['team']})" if row["team"] else ""
                lines.append(f"  {i}. {row['name']}{team_str}: {row['stat_value']} {stat}")
        else:
            rows = get_career_stat_leaders(stat)
            lines = [f"All-time career {stat} leaders:"]
            for i, row in enumerate(rows[:10], 1):
                lines.append(f"  {i}. {row['name']}: {row['stat_value']} {stat}")

        return "\n".join(lines)

    else:
        # RAG path: retrieve + generate
        try:
            chunks = retrieve(question, top_k=3)
        except Exception as e:  # noqa: BLE001 — chromadb.errors.NotFoundError not always importable
            if "NotFoundError" in type(e).__name__ or "not found" in str(e).lower():
                return "(No corpus indexed yet — run: python -m baseball_rag.corpus.ingest)"
            logger.exception("ChromaDB retrieval failed for query %r", question)
            raise

        if not chunks:
            return "I don't have any relevant information to answer that question."

        prompt = build_explanation_prompt(question, chunks)

        try:
            from baseball_rag.generation.llm import make_request

            response = make_request(prompt)
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
