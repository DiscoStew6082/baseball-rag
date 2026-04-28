"""Build a ChromaDB vector index from corpus documents."""

from pathlib import Path

import chromadb

from baseball_rag.corpus import get_hof_bios, get_stat_defs
from baseball_rag.corpus.frontmatter import parse_frontmatter
from baseball_rag.corpus.player_bios import build_player_bio
from baseball_rag.db.duckdb_schema import get_duckdb
from baseball_rag.retrieval.chroma_store import LMStudioEmbeddingFunction

# Batch size for ChromaDB inserts when indexing players
PLAYER_BATCH_SIZE = 500


def build_index(persist_dir: Path, *, include_players: bool = True) -> None:
    """Ingest all corpus documents into a ChromaDB collection.

    Creates a "baseball_corpus" collection with one chunk per document,
    storing text + source filename as metadata. By default, also indexes player
    bios from DuckDB for ~24k players.
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
        metadata={
            "description": (
                "Baseball stat definitions, Hall of Fame biographies, and player biographies"
            )
        },
    )

    total_docs = 0

    # Index static corpus docs (stat_defs and hof_bios)
    static_texts = []
    static_ids = []
    static_metas = []

    for path in [*get_stat_defs(), *get_hof_bios()]:
        result = parse_frontmatter(path.read_text())
        text = f"{result['metadata']['title']}\n\n{result['body'].strip()}"
        static_texts.append(text)
        static_ids.append(path.stem)
        static_metas.append(
            {
                "source": str(path.name),
                "category": result["metadata"].get("category", ""),
                "title": result["metadata"].get("title", ""),
            }
        )

    if static_texts:
        collection.add(documents=static_texts, ids=static_ids, metadatas=static_metas)  # type: ignore[arg-type]
        total_docs += len(static_texts)

    if not include_players:
        print(f"Indexed {total_docs} documents into baseball_corpus at {persist_dir}")
        return

    # Index player bios from DuckDB
    conn = get_duckdb()
    player_ids_rows = conn.execute("SELECT DISTINCT playerID FROM batting").fetchall()
    player_ids = [row[0] for row in player_ids_rows]
    print(f"Found {len(player_ids)} distinct players to index")

    # Process in batches
    batch_texts = []
    batch_ids = []
    batch_metas = []

    for idx, player_id in enumerate(player_ids):
        try:
            bio_text = build_player_bio(str(player_id), conn)
            batch_texts.append(bio_text)
            batch_ids.append(f"player:{player_id}")
            batch_metas.append(
                {
                    "source": f"{player_id}.md",
                    "category": "player_biography",
                    "title": player_id,
                }
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build bio for {player_id}: {e}") from e

        # Print progress every 1000 players
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(player_ids)} players...")

        # Flush batch when full
        if len(batch_texts) >= PLAYER_BATCH_SIZE:
            collection.add(documents=batch_texts, ids=batch_ids, metadatas=batch_metas)  # type: ignore[arg-type]
            total_docs += len(batch_texts)
            batch_texts = []
            batch_ids = []
            batch_metas = []

    # Flush any remaining players
    if batch_texts:
        collection.add(documents=batch_texts, ids=batch_ids, metadatas=batch_metas)  # type: ignore[arg-type]
        total_docs += len(batch_texts)

    print(f"Indexed {total_docs} documents into baseball_corpus at {persist_dir}")
