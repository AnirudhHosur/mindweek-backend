from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import SQLModel, Field


# ======================
# DATABASE MODELS
# ======================

class BrainDump(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str
    content: str
    created_at: datetime


class WeeklyPlan(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str
    plan_json: str
    created_at: datetime


class TaskPriority(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Task(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str

    # semantic core
    title: str

    # structured metadata
    category: str
    priority: TaskPriority
    estimated_minutes: int

    source_dump_id: str
    created_at: datetime


# ======================
# API SCHEMAS
# ======================

from pydantic import BaseModel

class BrainDumpCreate(BaseModel):
    content: str


class PlanMode(str, Enum):
    """How we choose which tasks to plan."""

    all = "all"  # plan ALL tasks for the user
    semantic_top_k = "semantic_top_k"  # plan only top-k most relevant tasks


class PlanRequest(BaseModel):
    # How many tasks to consider when using semantic_top_k mode.
    # Ignored when mode == PlanMode.all (we plan all tasks for the user).
    k: Optional[int] = 10
    # Planning mode: all tasks vs semantic top-k
    mode: PlanMode = PlanMode.all
    # Optional soft-focus category hint for planning (e.g. "finance", "health").
    category: Optional[str] = None


class WeeklyPlanRead(SQLModel):
    id: str
    user_id: str
    plan_json: str
    created_at: datetime


class WeeklyPlanWithTimeSlots(SQLModel):
    id: str
    user_id: str
    plan_json: str
    created_at: datetime