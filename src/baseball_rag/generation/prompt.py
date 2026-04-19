"""Prompt templates for RAG-grounded answer generation."""

from dataclasses import dataclass


@dataclass
class PromptBundle:
    system: str
    user: str

    def render(self) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) tuple for separate message fields."""
        return self.system, self.user


_BASE_ANSWER_TEMPLATE = PromptBundle(
    system=(
        "You are a knowledgeable baseball historian. Answer the user's question "
        "using ONLY the provided context documents.\n"
        "Do not include planning notes, internal monologue, task lists, or any structured "
        "reasoning markup (no lines starting with *, -, or `) in your response. "
        "Answer directly and concisely.\n"
        "Cite each piece of information by referencing the source document title in brackets."
    ),
    user=("Use the following documents to answer:\n\n{context}\n\n---\n\nQuestion: {question}"),
)


def build_stat_query_prompt(question: str, context_docs: list) -> tuple[str, str]:
    """Render a stat query prompt with retrieved document context."""
    ctx = "\n\n".join(f"[Source: {d.title}]\n{d.text}" for d in context_docs)
    return _BASE_ANSWER_TEMPLATE.render()[0], _BASE_ANSWER_TEMPLATE.user.format(
        context=ctx, question=question
    )


def build_explanation_prompt(question: str, context_docs: list) -> tuple[str, str]:
    """Render a general explanation prompt with retrieved document context."""
    ctx = "\n\n".join(f"[Source: {d.title}]\n{d.text}" for d in context_docs)
    return _BASE_ANSWER_TEMPLATE.render()[0], _BASE_ANSWER_TEMPLATE.user.format(
        context=ctx, question=question
    )


def build_player_bio_prompt(question: str, context_docs: list) -> tuple[str, str]:
    """Render a player biography prompt with retrieved document context."""
    ctx = "\n\n".join(f"[Source: {d.title}]\n{d.text}" for d in context_docs)
    return _BASE_ANSWER_TEMPLATE.render()[0], _BASE_ANSWER_TEMPLATE.user.format(
        context=ctx, question=question
    )
