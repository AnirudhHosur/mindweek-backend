import json
from typing import List, Dict, Any

import logging
from google.genai import types

from app.llm import client, GEMINI_MODEL, GEMINI_EMBED_MODEL


logger = logging.getLogger("mindweek-backend.planning")


# Central place for planning constraints
PLANNING_CONSTRAINTS: Dict[str, Any] = {
    "max_minutes_per_day": 8 * 60,
    "allowed_days": [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday",
    ],
    "task_unique": True,
    "no_new_tasks": True,
}


def embed_planning_query(text: str) -> List[float]:
    """Single-text embedding for planning queries."""
    response = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=text,
    )
    return list(response.embeddings[0].values)


def generate_plan_text(prompt: str) -> str:
    """LLM call for planning. Forces JSON-only output."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return response.text


def build_planner_prompt(task_payload: List[Dict[str, Any]], focus_category: Optional[str] = None) -> str:
    """Build the planner prompt from a normalized task payload.

    If focus_category is provided (e.g. "finance"), the planner is asked to
    prioritize tasks with that category, but it is still allowed to schedule
    other tasks if there is time. This is a *soft* guide, not a hard filter.
    """
    focus_rules = ""
    if focus_category:
        focus_rules = f"- Prioritize tasks with category: {focus_category}\n"

    return f"""
You are a weekly planning assistant.

Rules:
- Each task may appear AT MOST once
- Do NOT invent new tasks
- High priority tasks earlier in the week
- Respect estimated_minutes
- Max 8 hours of work per day
{focus_rules}

Tasks (JSON):
{json.dumps(task_payload, indent=2)}

Return JSON only in this format:
{{
  "Monday": [{{"title": "...", "minutes": 60}}],
  "Tuesday": [],
  "Wednesday": [],
  "Thursday": [],
  "Friday": [],
  "Saturday": [],
  "Sunday": []
}}
"""


def validate_plan(plan: Dict[str, Any], tasks: List[Dict[str, Any]]) -> List[str]:
    """
    Deterministic critic for weekly plans.

    Uses PLANNING_CONSTRAINTS to validate:
    - Total minutes per day do not exceed max_minutes_per_day
    - No task appears more than once (if task_unique is True)
    - No invented tasks (if no_new_tasks is True)
    - Only allowed days are used
    """
    errors: List[str] = []

    task_titles = {t["title"] for t in tasks}
    seen_tasks = set()

    for day, items in plan.items():
        # Check if day is in allowed_days
        if day not in PLANNING_CONSTRAINTS["allowed_days"]:
            errors.append(f"Invalid day: {day}")

        total_minutes = 0

        for item in items:
            title = item.get("title")
            minutes = item.get("minutes", 0)

            total_minutes += minutes

            # Check for invented tasks (if no_new_tasks constraint is enabled)
            if PLANNING_CONSTRAINTS.get("no_new_tasks", True):
                if title not in task_titles:
                    errors.append(f"Invented task: {title}")

            # Check for duplicate tasks (if task_unique constraint is enabled)
            if PLANNING_CONSTRAINTS.get("task_unique", True):
                if title in seen_tasks:
                    errors.append(f"Duplicate task: {title}")

            seen_tasks.add(title)

        # Check daily time limit
        if total_minutes > PLANNING_CONSTRAINTS["max_minutes_per_day"]:
            errors.append(
                f"{day} exceeds daily limit: {total_minutes} minutes "
                f"(max: {PLANNING_CONSTRAINTS['max_minutes_per_day']} minutes)"
            )

    return errors

