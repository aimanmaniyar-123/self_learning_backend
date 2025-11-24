# autonomous_loop.py
"""
Autonomous self-learning loop for the Self-Evolving Agent system.

This runs in the background and:
- Watches agents + goals + anomalies
- Auto-updates goal progress from agent metrics
- Auto-starts/finishes training sessions when needed
- Records evolution snapshots when training completes
- Auto-resolves stale anomalies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from agents import AGENTS, Agent  # type: ignore
from training import TRAINING_SESSIONS, TrainingSession  # type: ignore
from goals import GOALS, Goal  # type: ignore
from anomalies import ANOMALIES, Anomaly  # type: ignore
from evolution import record_evolution_snapshot  # type: ignore

logger = logging.getLogger("autonomous_loop")
logger.setLevel(logging.INFO)


# --- CONFIG -----------------------------------------------------------------


class AutonomyConfig:
  """Simple thresholds for when the loop should act."""

  # Run every N seconds
  POLL_INTERVAL_SECONDS: float = 10.0

  # If agent accuracy falls below this AND is active, trigger training
  MIN_ACCEPTABLE_ACCURACY: float = 93.0

  # How long (in seconds) a training session is considered "running"
  TRAINING_DURATION_SECONDS: int = 30

  # How long (in seconds) before an "investigating" anomaly is auto-resolved
  ANOMALY_AUTO_RESOLVE_SECONDS: int = 120


# internal state for evolution snapshots
_last_completed_trainings_count: int = 0


# --- HELPERS ----------------------------------------------------------------


def _now_utc() -> datetime:
  return datetime.utcnow()


def _parse_iso(ts: str) -> Optional[datetime]:
  try:
    return datetime.fromisoformat(ts)
  except Exception:
    return None


def _count_completed_trainings() -> int:
  return sum(1 for s in TRAINING_SESSIONS.values() if s.status == "completed")


# --- CORE AUTONOMOUS BEHAVIOUR ---------------------------------------------


def _auto_update_goals_from_agents() -> None:
  """
  For each active goal, update its progress based on the linked agent.
  - If goal.metric_name == "accuracy" -> progress from agent.accuracy vs target.
  - If >= 100%, mark as completed.
  """
  if not GOALS or not AGENTS:
    return

  for goal_id, goal in list(GOALS.items()):
    if goal.status != "active":
      continue

    agent: Optional[Agent] = AGENTS.get(goal.agent_id)  # type: ignore
    if not agent:
      continue

    # Only handle "accuracy" goals for now; others can be added later.
    if goal.metric_name.lower() == "accuracy":
      if goal.target_value <= 0:
        continue

      # progress based on agent accuracy
      pct = max(0.0, min(100.0, (agent.accuracy / goal.target_value) * 100.0))
      status = "completed" if pct >= 100.0 else goal.status

      updated = goal.copy(
        update={
          "current_value": agent.accuracy,
          "progress_percentage": pct,
          "status": status,
        }
      )
      GOALS[goal_id] = updated


def _auto_start_training_for_weak_agents() -> None:
  """
  If an agent is active and accuracy < threshold, and has no in-progress training,
  automatically start a training session.
  """
  from uuid import uuid4

  now = _now_utc()

  for agent_id, agent in list(AGENTS.items()):
    if agent.status != "active":
      continue

    if agent.accuracy >= AutonomyConfig.MIN_ACCEPTABLE_ACCURACY:
      continue

    # Any in-progress training for this agent?
    in_progress = any(
      s.agent == agent_id and s.status == "in_progress"
      for s in TRAINING_SESSIONS.values()
    )
    if in_progress:
      continue

    sid = str(uuid4())[:8]
    session = TrainingSession(
      id=sid,
      agent=agent_id,
      duration=5,  # pretend 5 epochs
      status="in_progress",
      timestamp=now.isoformat(),
    )
    TRAINING_SESSIONS[sid] = session
    logger.info(f"[AUTO] Started training session {sid} for weak agent {agent_id}")


def _auto_finish_old_trainings() -> None:
  """
  Mark in-progress training sessions as completed once they are older
  than TRAINING_DURATION_SECONDS.
  """
  now = _now_utc()
  for sid, session in list(TRAINING_SESSIONS.items()):
    if session.status != "in_progress":
      continue

    started_at = _parse_iso(session.timestamp)
    if not started_at:
      continue

    if now - started_at >= timedelta(seconds=AutonomyConfig.TRAINING_DURATION_SECONDS):
      updated = session.copy(update={"status": "completed"})
      TRAINING_SESSIONS[sid] = updated
      logger.info(f"[AUTO] Completed training session {sid}")


def _auto_resolve_stale_anomalies() -> None:
  """
  For demo: auto-resolve anomalies that have been 'investigating' for too long.
  """
  now = _now_utc()
  for aid, anomaly in list(ANOMALIES.items()):
    if anomaly.status != "investigating":
      continue

    created_at = _parse_iso(anomaly.timestamp)
    if not created_at:
      continue

    if now - created_at >= timedelta(
      seconds=AutonomyConfig.ANOMALY_AUTO_RESOLVE_SECONDS
    ):
      updated: Anomaly = anomaly.copy(update={"status": "resolved"})
      ANOMALIES[aid] = updated
      logger.info(f"[AUTO] Auto-resolved stale anomaly {aid}")


def _record_evolution_if_new_training_completed() -> None:
  """
  Whenever the number of completed training sessions increases,
  record an evolution snapshot based on current agents.
  """
  global _last_completed_trainings_count

  completed_now = _count_completed_trainings()
  if completed_now <= _last_completed_trainings_count:
    return

  _last_completed_trainings_count = completed_now

  if not AGENTS:
    return

  # Simple aggregate: average accuracy and "efficiency" as inverse of avg tasks
  agents_list = list(AGENTS.values())
  avg_accuracy = sum(a.accuracy for a in agents_list) / len(agents_list)
  avg_tasks = sum(a.tasks for a in agents_list) / len(agents_list)
  # fake efficiency: higher when tasks are lower
  efficiency = max(50.0, min(99.0, 100.0 - (avg_tasks / 1000.0)))

  record_evolution_snapshot(
    accuracy=avg_accuracy,
    efficiency=efficiency,
    agents_count=len(agents_list),
  )
  logger.info(
    f"[AUTO] Recorded evolution snapshot: acc={avg_accuracy:.2f}, eff={efficiency:.2f}"
  )


# --- BACKGROUND LOOP --------------------------------------------------------


async def autonomous_loop() -> None:
  """
  Long-running background task.

  IMPORTANT: Do not call this directly; let main.py schedule it on startup.
  """
  logger.info("[AUTO] Autonomous loop started")
  try:
    while True:
      try:
        _auto_update_goals_from_agents()
        _auto_start_training_for_weak_agents()
        _auto_finish_old_trainings()
        _auto_resolve_stale_anomalies()
        _record_evolution_if_new_training_completed()
      except Exception as e:
        logger.exception(f"[AUTO] Error in autonomous loop iteration: {e}")

      await asyncio.sleep(AutonomyConfig.POLL_INTERVAL_SECONDS)
  except asyncio.CancelledError:
    logger.info("[AUTO] Autonomous loop cancelled; shutting down")
    raise
