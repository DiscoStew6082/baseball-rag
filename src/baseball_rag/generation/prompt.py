"""Prompt templates for RAG-grounded answer generation."""

from dataclasses import dataclass


@dataclass
class PromptBundle:
    system: str
    user: str

    def render(self) -> str:
        return f"<|system|>\n{self.system}\n<|user|>\n{self.user}"


STAT_QUERY_TEMPLATE = PromptBundle(
    system=(
        "You are a knowledgeable baseball historian. Answer the user's question "
        "using ONLY the provided context documents.\n"
        "For stat queries, your answer must:\n"
        "1. State the player name, team (if applicable), and stat value clearly\n"
        "2. Briefly explain how that stat is defined or why it matters\n"
        "3. Cite each piece of information by referencing the source document title in brackets\n"
        "If the context does not contain enough info to fully answer, say what you can.\n\n"
        "Example: Mickey Mantle led MLB with 123 RBI for the New York Yankees "
        "in 1962 [Source: RBI.md]. RBI stands for Runs Batted In..."
    ),
    user=("Use the following documents to answer:\n\n{context}\n\n---\n\nQuestion: {question}"),
)


GENERAL_EXPLANATION_TEMPLATE = PromptBundle(
    system=(
        "You are a knowledgeable baseball historian. Use the provided context "
        "documents to give a thorough, engaging explanation.\n"
        "Ground every factual claim in at least one document and cite it in "
        "brackets like [Source: DocumentName.md].\n"
        "If you do not have enough information from the documents to fully "
        "answer, say so honestly."
    ),
    user=("Use the following documents to answer:\n\n{context}\n\n---\n\nQuestion: {question}"),
)


def build_stat_query_prompt(question: str, context_docs: list) -> str:
    """Render a stat query prompt with retrieved document context."""
    ctx = "\n\n".join(f"[Source: {d.title}]\n{d.text}" for d in context_docs)
    return STAT_QUERY_TEMPLATE.render().format(context=ctx, question=question)


def build_explanation_prompt(question: str, context_docs: list) -> str:
    """Render a general explanation prompt with retrieved document context."""
    ctx = "\n\n".join(f"[Source: {d.title}]\n{d.text}" for d in context_docs)
    return GENERAL_EXPLANATION_TEMPLATE.render().format(context=ctx, question=question)


PLAYER_BIO_TEMPLATE = PromptBundle(
    system=(
        "You are a knowledgeable baseball historian. Use the provided player biography "
        "to answer the user's question about this player.\n"
        "Include relevant details from the bio: teams played for, years active, position, etc.\n"
        "Cite the source using [Source: playerID] format."
    ),
    user=("Use the following documents to answer:\n\n{context}\n\n---\n\nQuestion: {question}"),
)


def build_player_bio_prompt(question: str, context_docs: list) -> str:
    """Render a player biography prompt with retrieved document context."""
    ctx = "\n\n".join(f"[Source: {d.title}]\n{d.text}" for d in context_docs)
    return PLAYER_BIO_TEMPLATE.render().format(context=ctx, question=question)
