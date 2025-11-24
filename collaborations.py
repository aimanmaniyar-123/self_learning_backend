# routers/collaborations.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from uuid import uuid4
from datetime import datetime

router = APIRouter()

class CollaborationCreate(BaseModel):
    name: str
    agent_ids: List[str]
    description: str

class Collaboration(BaseModel):
    id: str
    name: str
    status: str
    tasks: int
    efficiency: float
    agents: List[str]
    description: str
    created_at: str

COLLABORATIONS: Dict[str, Collaboration] = {}

def _seed():
    if COLLABORATIONS:
        return
    cid = str(uuid4())[:8]
    COLLABORATIONS[cid] = Collaboration(
        id=cid,
        name="Data Sync Cluster",
        status="active",
        tasks=1500,
        efficiency=92.0,
        agents=["Alpha-7", "Beta-4"],
        description="Cooperative data pipeline",
        created_at=datetime.utcnow().isoformat(),
    )

_seed()

@router.get("", response_model=List[Collaboration])
async def get_collaborations():
    """
    Frontend: api.getCollaborations()
    """
    return list(COLLABORATIONS.values())

@router.post("", response_model=Collaboration)
async def create_collaboration(body: CollaborationCreate):
    """
    Frontend: api.createCollaboration(newCollab)
    """
    cid = str(uuid4())[:8]
    collab = Collaboration(
        id=cid,
        name=body.name,
        status="active",
        tasks=0,
        efficiency=0.0,
        agents=body.agent_ids,
        description=body.description,
        created_at=datetime.utcnow().isoformat(),
    )
    COLLABORATIONS[cid] = collab
    return collab
