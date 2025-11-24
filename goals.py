# routers/goals.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from uuid import uuid4
from datetime import datetime

router = APIRouter()

class GoalCreate(BaseModel):
    agent_id: str
    goal_type: str
    goal_name: str
    description: str
    target_value: float
    metric_name: str
    priority: int = 5
    days_until_target: int = 30

class GoalUpdateProgress(BaseModel):
    current_value: float

class Goal(BaseModel):
    id: str
    agent_id: str
    goal_type: str
    goal_name: str
    description: str
    target_value: float
    current_value: float
    metric_name: str
    priority: int
    status: str
    progress_percentage: float
    created_at: str

GOALS: Dict[str, Goal] = {}

@router.get("", response_model=List[Goal])
async def list_goals():
    """
    Frontend: api.getGoals()
    Returns full list of goals for UI display/filter.
    """
    return list(GOALS.values())

@router.get("/summary")
async def get_goals_summary():
    """
    Frontend: api.getGoalsSummary()
    UI right now only uses summary, not list details.
    """
    total = len(GOALS)
    completed = len([g for g in GOALS.values() if g.status == "completed"])
    active = len([g for g in GOALS.values() if g.status == "active"])
    failed = len([g for g in GOALS.values() if g.status == "failed"])

    avg_progress = sum(g.progress_percentage for g in GOALS.values()) / total if total else 0.0

    return {
        "total_goals": total,
        "active_goals": active,
        "completed_goals": completed,
        "failed_goals": failed,
        "average_progress": avg_progress,
    }

@router.post("", response_model=Goal)
async def create_goal(req: GoalCreate):
    """
    Frontend: api.createGoal(newGoal)
    """
    gid = str(uuid4())[:8]
    goal = Goal(
        id=gid,
        agent_id=req.agent_id,
        goal_type=req.goal_type,
        goal_name=req.goal_name,
        description=req.description,
        target_value=req.target_value,
        current_value=0.0,
        metric_name=req.metric_name,
        priority=req.priority,
        status="active",
        progress_percentage=0.0,
        created_at=datetime.utcnow().isoformat(),
    )
    GOALS[gid] = goal
    return goal

@router.delete("/{goal_id}")
async def delete_goal(goal_id: str):
    """
    Frontend: api.deleteGoal(id)
    """
    if goal_id not in GOALS:
        raise HTTPException(status_code=404, detail="Goal not found")
    del GOALS[goal_id]
    return {"status": "deleted", "goal_id": goal_id}

@router.patch("/{goal_id}/progress")
async def update_goal_progress(goal_id: str, body: GoalUpdateProgress):
    """
    Frontend: api.updateGoalProgress(id, { current_value / progress })
    UI uses a % value; here we treat current_value as percentage toward target.
    """
    goal = GOALS.get(goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")

    pct = max(0.0, min(100.0, body.current_value))
    status = goal.status
    if pct >= 100:
        status = "completed"

    updated = goal.copy(update={
        "current_value": body.current_value,
        "progress_percentage": pct,
        "status": status,
    })
    GOALS[goal_id] = updated
    return updated
