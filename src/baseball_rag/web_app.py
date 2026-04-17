"""Gradio web interface for Baseball RAG query engine."""
import gradio as gr

from baseball_rag.cli import answer


def respond(message: str, history: list[list[str]]) -> str:
    """Handle a single user message and return assistant response."""
    result = answer(message)
    return result


demo = gr.ChatInterface(
    fn=respond,
    title="Baseball RAG Query Engine",
    description=(
        "Ask about MLB history in natural language. "
        "Stat queries (e.g. *'most RBIs in 1962'*) run SQL directly against a seed database. "
        "General questions (*'who was Babe Ruth?'*) use "
        "retrieval-augmented generation with Gemma 4."
    ),
    examples=[
        ["who had the most RBIs in 1962"],
        ["career home run leaders"],
        ["who was babe ruth"],
        ["what is a home run"],
        ["tell me about ted williams"],
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
