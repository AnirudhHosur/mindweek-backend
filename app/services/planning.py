import json
from typing import List, Dict, Any, Optional

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


def build_planner_prompt(task_payload: list[dict], focus_category: str | None = None) -> str:
    focus_line = (
        f"- Prefer tasks in the '{focus_category}' category when choosing what to schedule.\n"
        if focus_category
        else ""
    )

    return f"""
You are an AI weekly planning assistant.

You receive a list of tasks as JSON. Each task has:
- title (string)
- category (string)
- priority (\"high\" | \"medium\" | \"low\")
- estimated_minutes (integer, duration of the task)

Your goal is to assign these tasks to days in a 7-day week.

Hard rules:
- Each task from the input may appear AT MOST once in the plan.
- Do NOT invent new tasks or new fields.
- Do NOT invent extra days (only Monday–Sunday).
- Respect estimated_minutes for each task.
- Try to schedule higher-priority tasks earlier in the week.
- Total scheduled work per day should be at most ~8 hours (480 minutes).
- It's okay if some tasks remain unscheduled if time runs out.

Soft rules:
- Spread tasks reasonably across the week.
- Mix different categories where it makes sense (not all tasks on one day).
{focus_line}\
Input tasks (JSON array):
{json.dumps(task_payload, indent=2)}

Output format:
Return a single JSON object with EXACTLY these 7 keys:
- "Monday"
- "Tuesday"
- "Wednesday"
- "Thursday"
- "Friday"
- "Saturday"
- "Sunday"

Each value MUST be an array (list) of task objects for that day.
Each task object MUST have exactly:
- "title": string  (must match one of the input task titles)
- "minutes": integer  (the scheduled duration for that task on that day)

Example of the structure (this is just a structural example, NOT the actual plan):
{{
  "Monday": [
    {{ "title": "Apply for 3 jobs", "minutes": 90 }},
    {{ "title": "Go to the gym", "minutes": 60 }}
  ],
  "Tuesday": [],
  "Wednesday": [
    {{ "title": "Deep clean apartment", "minutes": 120 }}
  ],
  "Thursday": [],
  "Friday": [],
  "Saturday": [],
  "Sunday": []
}}

Important:
- Follow this structure for ALL days (Monday through Sunday).
- Every day must be present in the output JSON, even if the array is empty.
- Do not wrap the JSON in markdown fences or add any explanation.
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

