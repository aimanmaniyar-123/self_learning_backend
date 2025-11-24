# routers/agents.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from uuid import uuid4
from datetime import datetime
import time

from learning_engine import SELF_LEARNING          # your learning engine
from shared_stats import AGENT_STATS               # global performance tracking

router = APIRouter()


# ---------------------------------------------------
# Models
# ---------------------------------------------------

class AgentMessage(BaseModel):
    timestamp: str
    role: str
    content: str


class AgentCreate(BaseModel):
    name: str
    agent_type: Optional[str] = "default"
    description: Optional[str] = ""
    config: Dict[str, Any] = {}


class AgentUpdate(BaseModel):
    status: Optional[str] = None
    description: Optional[str] = None
    name: Optional[str] = None
    agent_type: Optional[str] = None


class Agent(BaseModel):
    id: str
    name: str
    agent_type: str
    description: str
    status: str
    version: str
    accuracy: float
    tasks: int
    lastActive: str
    history: List[AgentMessage] = []


class AgentInteractRequest(BaseModel):
    input: str


class AgentInteractResponse(BaseModel):
    agent_id: str
    response: str


# ---------------------------------------------------
# Global in-memory agents list
# ---------------------------------------------------

AGENTS: Dict[str, Agent] = {}


def _seed_demo_agent():
    """Seed one sample agent so UI is not empty."""
    if AGENTS:
        return
    aid = "alpha-7"
    AGENTS[aid] = Agent(
        id=aid,
        name="Alpha-7",
        agent_type="Classification",
        description="Baseline classifier agent",
        status="active",
        version="2.3",
        accuracy=95.2,
        tasks=1247,
        lastActive="2 min ago",
        history=[],
    )


_seed_demo_agent()


# ---------------------------------------------------
# Utility: update agent stats
# ---------------------------------------------------

def update_agent_stats(agent_id: str, success: bool, response_time_ms: float):
    """Track success rate + avg response time for dashboard & performance UI."""
    if agent_id not in AGENT_STATS:
        AGENT_STATS[agent_id] = {
            "requests": 0,
            "success": 0,
            "failures": 0,
            "total_response_time": 0.0,
            "avg_response_time": 0.0,
            "success_rate": 0.0,
        }

    stats = AGENT_STATS[agent_id]

    stats["requests"] += 1
    stats["total_response_time"] += response_time_ms

    if success:
        stats["success"] += 1
    else:
        stats["failures"] += 1

    stats["avg_response_time"] = stats["total_response_time"] / stats["requests"]
    stats["success_rate"] = (stats["success"] / stats["requests"]) * 100


# ---------------------------------------------------
# CRUD
# ---------------------------------------------------

@router.get("", response_model=List[Agent])
async def list_agents():
    return list(AGENTS.values())


@router.get("/{agent_id}", response_model=Agent)
async def get_agent(agent_id: str):
    agent = AGENTS.get(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return agent


@router.post("", response_model=Agent)
async def create_agent(payload: AgentCreate):
    agent_id = str(uuid4())[:8]
    agent = Agent(
        id=agent_id,
        name=payload.name,
        agent_type=payload.agent_type or "default",
        description=payload.description or "",
        status="active",
        version="1.0",
        accuracy=90.0,
        tasks=0,
        lastActive="just now",
        history=[],
    )
    AGENTS[agent_id] = agent
    return agent


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    if agent_id not in AGENTS:
        raise HTTPException(404, "Agent not found")
    del AGENTS[agent_id]
    return {"status": "deleted", "agent_id": agent_id}


@router.patch("/{agent_id}", response_model=Agent)
async def update_agent(agent_id: str, payload: AgentUpdate):
    agent = AGENTS.get(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")

    data = agent.dict()

    if payload.status is not None:
        data["status"] = payload.status
        data["lastActive"] = "just now"

    if payload.description is not None:
        data["description"] = payload.description

    if payload.name is not None:
        data["name"] = payload.name

    if payload.agent_type is not None:
        data["agent_type"] = payload.agent_type

    updated = Agent(**data)
    AGENTS[agent_id] = updated
    return updated


# ---------------------------------------------------
# History
# ---------------------------------------------------

@router.get("/{agent_id}/history", response_model=List[AgentMessage])
async def get_agent_history(agent_id: str):
    agent = AGENTS.get(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return agent.history


# ---------------------------------------------------
# Interaction (Self-learning prediction + history + stats)
# ---------------------------------------------------

@router.post("/{agent_id}/interact", response_model=AgentInteractResponse)
async def interact_with_agent(agent_id: str, req: AgentInteractRequest):
    agent = AGENTS.get(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")

    start = time.time()

    # Get prediction from learning engine
    prediction = SELF_LEARNING.predict(agent_id, req.input)

    if prediction is None:
        reply = (
            f"[NO MODEL YET] Agent {agent_id} received: {req.input}\n"
            f"Tip: send feedback to /training/feedback to train this agent."
        )
        success = True
    else:
        label = int(prediction["label"])
        prob = prediction["proba"]
        reply = (
            f"Agent {agent_id} prediction:\n"
            f"- Label: {label}\n"
            f"- Confidence: {prob:.2f}\n\n"
            f"Input: {req.input}"
        )
        success = True   # prediction worked

    # Calculate response time
    response_time_ms = (time.time() - start) * 1000

    # Update agent stats
    update_agent_stats(agent_id, success, response_time_ms)

    # Log history
    now = datetime.utcnow().isoformat()
    history = list(agent.history)
    history.append(AgentMessage(timestamp=now, role="user", content=req.input))
    history.append(AgentMessage(timestamp=now, role="agent", content=reply))

    # Update agent data
    data = agent.dict()
    data["tasks"] += 1
    data["lastActive"] = "just now"
    data["history"] = history

    AGENTS[agent_id] = Agent(**data)

    return AgentInteractResponse(agent_id=agent_id, response=reply)
