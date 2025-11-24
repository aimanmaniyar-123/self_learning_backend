# routers/training.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from uuid import uuid4
from datetime import datetime

from learning_engine import SELF_LEARNING  # 🔥 new import

router = APIRouter()


class TrainingStartRequest(BaseModel):
  agent_id: str
  epochs: int
  learning_rate: float
  batch_size: int


class TrainingSession(BaseModel):
  id: str
  agent: str
  duration: int
  status: str
  timestamp: str


class FeedbackExample(BaseModel):
  """
  User feedback / labeled example for self-learning.
  label: 1 = success / positive / correct
         0 = failure / negative / incorrect
  """
  agent_id: str
  input_text: str
  label: int


TRAINING_SESSIONS: Dict[str, TrainingSession] = {}


@router.get("/sessions", response_model=List[TrainingSession])
async def list_sessions():
  """
  Frontend: api.getTrainingSessions()
  """
  return list(TRAINING_SESSIONS.values())


@router.post("/start")
async def start_training(req: TrainingStartRequest):
  """
  Frontend: api.startTraining(trainingData)

  Now:
  - still creates a TrainingSession (for UI)
  - also calls SELF_LEARNING.train_agent(...) to actually train
  """
  sid = str(uuid4())[:8]
  session = TrainingSession(
    id=sid,
    agent=req.agent_id,
    duration=req.epochs,  # interpreted as epochs
    status="in_progress",
    timestamp=datetime.utcnow().isoformat(),
  )
  TRAINING_SESSIONS[sid] = session

  # 🔥 run actual training using saved feedback
  metrics = SELF_LEARNING.train_agent(req.agent_id, epochs=req.epochs)

  # Immediately mark as completed (for now) – you can extend later
  TRAINING_SESSIONS[sid] = session.copy(update={"status": "completed"})

  return {
    "status": "started",
    "session_id": sid,
    "agent_id": req.agent_id,
    "training_metrics": metrics,
  }


@router.post("/stop/{session_id}")
async def stop_training(session_id: str):
  """
  Frontend: api.stopTraining(sessionId)
  """
  session = TRAINING_SESSIONS.get(session_id)
  if not session:
    raise HTTPException(status_code=404, detail="Session not found")

  TRAINING_SESSIONS[session_id] = session.copy(update={"status": "completed"})
  return {"status": "stopped", "session_id": session_id}


# 🔥 NEW: feedback endpoint for self-learning
@router.post("/feedback")
async def submit_feedback(example: FeedbackExample):
  """
  Send labeled feedback so the agent can learn.

  Example body:
  {
    "agent_id": "alpha-7",
    "input_text": "This classification was wrong",
    "label": 0
  }
  """
  metrics = SELF_LEARNING.add_feedback(
    agent_id=example.agent_id,
    text=example.input_text,
    label=example.label,
  )

  return {
    "status": "ok",
    "agent_id": example.agent_id,
    "num_samples": metrics["num_samples"],
    "accuracy": metrics["accuracy"],
  }
