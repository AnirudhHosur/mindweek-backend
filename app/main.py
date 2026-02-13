import os
import uuid
import json
from datetime import datetime
from typing import List

import logging
from fastapi import FastAPI, Depends
from sqlmodel import Session, select

from .db import init_db, get_session
from .models import (
    BrainDump,
    WeeklyPlan,
    Task,
    TaskPriority,
    BrainDumpCreate,
    PlanRequest,
    WeeklyPlanRead,
    PlanMode,
)
from .vector_store import add_task_embedding, query_task_ids
from .services.tasks import extract_tasks, embed_task_titles
from .services.planning import (
    embed_planning_query,
    build_planner_prompt,
    generate_plan_text,
    validate_plan,
)

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mindweek-backend")


app = FastAPI(title="MindWeek Backend v2", version="0.2.0")


@app.on_event("startup")
def on_startup():
    init_db()


# ---------- ROUTES ----------

@app.post("/brain-dump", response_model=BrainDump)
def create_brain_dump(
    payload: BrainDumpCreate,
    session: Session = Depends(get_session),
):
    """Store brain dump, extract tasks, store tasks in SQL + vector DB."""
    dump_id = str(uuid.uuid4())
    now = datetime.utcnow()

    dump = BrainDump(
        id=dump_id,
        user_id=payload.user_id,
        content=payload.content,
        created_at=now,
    )
    session.add(dump)
    session.commit()
    session.refresh(dump)

    # 1) Extract structured tasks from LLM
    tasks = extract_tasks(payload.content)
    logger.info("Extracted %d tasks from brain dump (dump_id=%s)", len(tasks), dump_id)

    if not tasks:
        logger.info("No tasks extracted; brain dump saved only (no SQL tasks or vector store writes)")
        return dump

    # 2) Batch-embed task titles (one API call for all)
    task_ids = [str(uuid.uuid4()) for _ in tasks]
    titles = [t["title"] for t in tasks]
    embeddings = embed_task_titles(titles)
    logger.info("Batch embedded %d task titles; storing in SQL and Chroma", len(embeddings))

    # 3) Save each task in SQL + add its embedding to Chroma (vector DB)
    for task_id, t, emb in zip(task_ids, tasks, embeddings):
        task_row = Task(
            id=task_id,
            user_id=payload.user_id,
            title=t["title"],
            category=t["category"],
            priority=TaskPriority(t["priority"]),
            estimated_minutes=t["estimated_minutes"],
            source_dump_id=dump_id,
            created_at=now,
        )
        session.add(task_row)
        add_task_embedding(
            task_id=task_id,
            user_id=payload.user_id,
            title=t["title"],
            embedding=emb,
        )

    session.commit()
    logger.info("Stored %d tasks in SQL and Chroma for user_id=%s (dump_id=%s)", len(tasks), payload.user_id, dump_id)

    return dump


@app.post("/generate-weekly-plan", response_model=WeeklyPlanRead)
def generate_weekly_plan(
    req: PlanRequest,
    session: Session = Depends(get_session),
):
    """Task-aware weekly planner using SQL + vector retrieval.

    Modes:
    - mode=\"all\":           plan ALL tasks for this user (no semantic top-k)
    - mode=\"semantic_top_k\": plan only the top-k most relevant tasks
    """
    logger.info(
        "Generating weekly plan for user_id=%s (mode=%s, k=%s, focus_category=%s)",
        req.user_id,
        req.mode.value,
        req.k,
        req.category,
    )

    # 1) Select candidate tasks
    if req.mode == PlanMode.all:
        # Plan ALL tasks for this user (optionally soft-focus in the prompt via category)
        query = select(Task).where(Task.user_id == req.user_id)
        tasks = session.exec(query).all()
        logger.info("Mode=all → selected %d tasks for planning", len(tasks))

    else:  # PlanMode.semantic_top_k
        # Use semantic search to get top-k most relevant tasks for this user
        planning_query = (
            f"{req.category} tasks for this week"
            if req.category
            else "all my tasks and to-dos for this week"
        )
        logger.info("Mode=semantic_top_k → using semantic query: '%s'", planning_query)
        query_embedding = embed_planning_query(planning_query)

        task_ids = query_task_ids(
            user_id=req.user_id,
            query_embedding=query_embedding,
            k=req.k or 10,
        )
        logger.info(
            "Mode=semantic_top_k → retrieved %d task IDs from vector DB for user_id=%s",
            len(task_ids),
            req.user_id,
        )

        if not task_ids:
            logger.info("No tasks found for user_id=%s in semantic_top_k mode", req.user_id)
            empty_plan = WeeklyPlan(
                id=str(uuid.uuid4()),
                user_id=req.user_id,
                plan_json="{}",
                created_at=datetime.utcnow(),
            )
            session.add(empty_plan)
            session.commit()
            return empty_plan

        tasks = session.exec(
            select(Task).where(Task.id.in_(task_ids))
        ).all()
        logger.info(
            "Mode=semantic_top_k → fetched %d full task objects from SQL for planning",
            len(tasks),
        )

    if not tasks:
        logger.info("No tasks available for planning for user_id=%s", req.user_id)
        empty_plan = WeeklyPlan(
            id=str(uuid.uuid4()),
            user_id=req.user_id,
            plan_json="{}",
            created_at=datetime.utcnow(),
        )
        session.add(empty_plan)
        session.commit()
        return empty_plan

    logger.info("Fetched %d full task objects from SQL for planning", len(tasks))
    
    if tasks:
        logger.info("Sample tasks for planning: %s", 
                   [f"{t.title[:40]}..." if len(t.title) > 40 else t.title for t in tasks[:3]])

    # 4) Prepare structured payload for planner
    task_payload = [
        {
            "title": t.title,
            "category": t.category,
            "priority": t.priority.value,
            "estimated_minutes": t.estimated_minutes,
        }
        for t in tasks
    ]
    logger.info(
        "Prepared task payload with %d tasks (total estimated minutes: %d)",
        len(task_payload),
        sum(t["estimated_minutes"] for t in task_payload),
    )

    # 5) Planner prompt (constraints + optional soft-focus category)
    prompt = build_planner_prompt(task_payload, focus_category=req.category)
    logger.debug("Built planner prompt (%d chars)", len(prompt))

    # 6) Planner → Critic → Fixer loop (max 2 LLM calls)
    logger.info("Calling LLM planner (first attempt)")
    plan_json = generate_plan_text(prompt)
    try:
        plan_dict = json.loads(plan_json)
        logger.info("Planner returned valid JSON (first attempt)")
    except json.JSONDecodeError as e:
        logger.error("Planner returned invalid JSON on first attempt: %s", e)
        plan_dict = {}

    errors = validate_plan(plan_dict if isinstance(plan_dict, dict) else {}, task_payload)

    if not errors:
        logger.info("Plan passed validation (no critic errors)")
    else:
        logger.info("Plan validation found %d errors: %s", len(errors), errors[:3])  # Show first 3

    # If invalid, one fixer round
    if errors:
        logger.info("Plan validation errors: %s", errors)
        fix_prompt = f"""
You are fixing a weekly plan.

Here is the INVALID plan:
{json.dumps(plan_dict, indent=2)}

Here are the violations:
{json.dumps(errors, indent=2)}

Rules:
- Fix ONLY the violations
- Do not add new tasks
- Keep JSON format identical

Return corrected JSON only.
"""
        logger.info("Calling LLM fixer to correct %d validation errors", len(errors))
        fixed_plan_json = generate_plan_text(fix_prompt)
        try:
            fixed_plan_dict = json.loads(fixed_plan_json)
            fixed_errors = validate_plan(fixed_plan_dict if isinstance(fixed_plan_dict, dict) else {}, task_payload)
            if not fixed_errors:
                logger.info("Fixer succeeded: plan now passes validation")
                plan_json = json.dumps(fixed_plan_dict)
            else:
                logger.warning("Fixed plan still has %d errors: %s", len(fixed_errors), fixed_errors[:3])
        except json.JSONDecodeError as e:
            logger.error("Fixer returned invalid JSON: %s; falling back to first plan.", e)

    # 7) Store plan
    plan_id = str(uuid.uuid4())
    logger.info("Storing weekly plan (plan_id=%s) for user_id=%s", plan_id[:8], req.user_id)
    
    # Log plan summary
    try:
        plan_summary = json.loads(plan_json)
        total_tasks_planned = sum(len(day_tasks) for day_tasks in plan_summary.values() if isinstance(day_tasks, list))
        logger.info("Plan summary: %d tasks scheduled across %d days", 
                   total_tasks_planned, sum(1 for v in plan_summary.values() if isinstance(v, list) and v))
    except:
        pass
    
    plan = WeeklyPlan(
        id=plan_id,
        user_id=req.user_id,
        plan_json=plan_json,
        created_at=datetime.utcnow(),
    )
    session.add(plan)
    session.commit()
    session.refresh(plan)
    
    logger.info("Weekly plan stored successfully (plan_id=%s)", plan_id[:8])

    return plan
