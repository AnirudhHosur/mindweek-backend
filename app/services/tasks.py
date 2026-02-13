import json
from typing import List, Dict, Any

import logging
from google.genai import types

from app.llm import client, GEMINI_MODEL, GEMINI_EMBED_MODEL


logger = logging.getLogger("mindweek-backend.tasks")


# ---------- Task extraction schema & helpers ----------

TASK_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "category": {
                "type": "string",
                "enum": ["work", "health", "finance", "personal", "admin", "fun"],
            },
            "priority": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "estimated_minutes": {"type": "integer"},
        },
        "required": ["title", "category", "priority", "estimated_minutes"],
    },
}


def extract_tasks(text: str) -> List[dict]:
    """Use Gemini JSON mode to extract structured tasks from a brain dump."""
    prompt = f"""
Extract actionable tasks from the text below.

Rules:
- Each task must be actionable
- Estimate time in minutes
- Use only these categories: work, health, finance, personal, admin, fun
- Use only these priorities: high, medium, low

Text:
{text}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=TASK_SCHEMA,
        ),
    )

    raw = response.text or ""
    logger.info("Raw extract_tasks structured response: %s", raw[:300])

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse tasks JSON: %s", e)
        logger.error("Offending text: %s", raw)
        return []


def embed_task_titles(titles: List[str]) -> List[List[float]]:
    """Batch embedding helper specifically for task titles."""
    if not titles:
        return []

    response = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=titles,
    )

    return [list(e.values) for e in response.embeddings]

