"""Generation layer — LLM + prompts."""
from baseball_rag.generation.answer import answer
from baseball_rag.generation.llm import make_request, make_request_stream
from baseball_rag.generation.prompt import (
    build_explanation_prompt,
    build_stat_query_prompt,
)

__all__ = [
    "answer",
    "make_request",
    "make_request_stream",
    "build_stat_query_prompt",
    "build_explanation_prompt",
]
