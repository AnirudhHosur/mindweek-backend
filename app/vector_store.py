import logging
import chromadb
from typing import List

# Persist to disk so you see chroma_data/ and embeddings survive restarts
CHROMA_PATH = "./chroma_data"
client = chromadb.PersistentClient(path=CHROMA_PATH)

task_collection = client.get_or_create_collection(
    name="tasks"
)

logger = logging.getLogger("mindweek-backend.vector_store")


def add_task_embedding(
    task_id: str,
    user_id: str,
    title: str,
    embedding: List[float],
    category: str = None,
):
    """Store a task's title embedding in Chroma (vector DB) for semantic search."""
    metadata = {
        "task_id": task_id,
        "user_id": user_id,
    }
    if category:
        metadata["category"] = category.lower()
    
    task_collection.add(
        ids=[task_id],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[title],
    )
    title_preview = (title[:50] + "…") if len(title) > 50 else title
    logger.info("Vector DB: added task embedding (task_id=%s, title=%s, category=%s)", task_id[:8], title_preview, category)


def query_task_ids(
    user_id: str,
    query_embedding: List[float],
    k: int = 10,
    category: str = None,
) -> List[str]:

    logger.info(
        "Querying vector DB: requesting %d results for user_id=%s, category=%s",
        k, user_id, category
    )

    if category:
        where_clause = {
            "$and": [
                {"user_id": user_id},
                {"category": category.lower()}
            ]
        }
    else:
        where_clause = {"user_id": user_id}

    results = task_collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_clause,
    )

    ids = results.get("ids", [[]])[0]

    logger.info(
        "Vector DB returned %d results for user_id=%s (requested %d)",
        len(ids), user_id, k
    )

    return ids
