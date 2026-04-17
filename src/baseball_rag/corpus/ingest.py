"""Build a ChromaDB vector index from corpus documents."""

from pathlib import Path

import chromadb

from baseball_rag.corpus import get_hof_bios, get_stat_defs
from baseball_rag.corpus.frontmatter import parse_frontmatter
from baseball_rag.retrieval.chroma_store import LMStudioEmbeddingFunction


def build_index(persist_dir: Path) -> None:
    """Ingest all corpus documents into a ChromaDB collection.

    Creates a "baseball_corpus" collection with one chunk per document,
    storing text + source filename as metadata.
    """
    persist_dir = Path(persist_dir)

    client = chromadb.PersistentClient(path=str(persist_dir))

    # Wipe and rebuild each time for reproducibility
    try:
        client.delete_collection("baseball_corpus")
    except Exception:
        pass

    collection = client.create_collection(
        name="baseball_corpus",
        embedding_function=LMStudioEmbeddingFunction(),  # type: ignore[arg-type]
        metadata={"description": "Baseball stat definitions and Hall of Fame biographies"},
    )

    docs_to_add = []
    for path in [*get_stat_defs(), *get_hof_bios()]:
        result = parse_frontmatter(path.read_text())
        text = f"{result['metadata']['title']}\n\n{result['body'].strip()}"
        docs_to_add.append({
            "id": path.stem,
            "text": text,
            "metadata": {
                "source": str(path.name),
                "category": result["metadata"].get("category", ""),
                "title": result["metadata"].get("title", ""),
            },
        })

    texts = [d["text"] for d in docs_to_add]
    ids = [d["id"] for d in docs_to_add]
    metas = [d["metadata"] for d in docs_to_add]

    collection.add(documents=texts, ids=ids, metadatas=metas)  # type: ignore[arg-type]
    print(f"Indexed {len(docs_to_add)} documents into baseball_corpus at {persist_dir}")
