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
):
    """Store a task's title embedding in Chroma (vector DB) for semantic search."""
    task_collection.add(
        ids=[task_id],
        embeddings=[embedding],
        metadatas=[{
            "task_id": task_id,
            "user_id": user_id,
        }],
        documents=[title],
    )
    title_preview = (title[:50] + "…") if len(title) > 50 else title
    logger.info("Vector DB: added task embedding (task_id=%s, title=%s)", task_id[:8], title_preview)


def query_task_ids(
    user_id: str,
    query_embedding: List[float],
    k: int = 10,
) -> List[str]:
    """
    Query Chroma for task IDs by semantic similarity.
    
    Note: We query MORE than k (k*3) to account for user_id filtering,
    since Chroma returns top-k by similarity BEFORE filtering.
    """
    # Query more results to ensure we get k after filtering by user_id
    query_k = max(k * 3, 50)  # Query at least 50, or 3x requested
    
    logger.info("Querying vector DB: requesting %d results (will filter to %d for user_id=%s)", 
                query_k, k, user_id)
    
    results = task_collection.query(
        query_embeddings=[query_embedding],
        n_results=query_k,
    )

    ids = results.get("ids", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    
    logger.debug("Vector DB returned %d total results before filtering", len(ids))

    # Filter by user
    task_ids = [
        meta["task_id"]
        for meta in metas
        if meta.get("user_id") == user_id
    ]
    
    logger.info("After filtering by user_id=%s: %d task IDs found (requested %d)", 
                user_id, len(task_ids), k)

    # Deduplicate and limit to k
    unique_task_ids = list(dict.fromkeys(task_ids))[:k]
    
    if len(unique_task_ids) < k:
        logger.warning("Only found %d tasks for user_id=%s (requested %d). "
                     "Consider querying all tasks or improving query embedding.", 
                     len(unique_task_ids), user_id, k)
    
    return unique_task_ids