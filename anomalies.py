# routers/anomalies.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from uuid import uuid4
from datetime import datetime

router = APIRouter()

class Anomaly(BaseModel):
    id: str
    title: str
    severity: str
    status: str
    agent: str
    type: str
    impact: str
    description: str
    timestamp: str

ANOMALIES: Dict[str, Anomaly] = {}

def _seed():
    if ANOMALIES:
        return
    aid = str(uuid4())[:8]
    ANOMALIES[aid] = Anomaly(
        id=aid,
        title="High Latency Spike",
        severity="high",
        status="investigating",
        agent="Alpha-7",
        type="performance",
        impact="high",
        description="Latency exceeded 1s for 5% of requests",
        timestamp=datetime.utcnow().isoformat(),
    )

_seed()

@router.get("")
async def get_anomalies():
    """
    Frontend: api.getAnomalies()
    Expects: { total_alerts, alerts: [...] }
    """
    alerts = list(ANOMALIES.values())
    return {
        "total_alerts": len(alerts),
        "alerts": alerts,
    }

@router.patch("/{anomaly_id}/resolve")
async def resolve_anomaly(anomaly_id: str):
    """
    Frontend: api.resolveAnomaly(id) – currently UI just calls and ignores response.
    """
    anomaly = ANOMALIES.get(anomaly_id)
    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    updated = anomaly.copy(update={"status": "resolved"})
    ANOMALIES[anomaly_id] = updated
    return {"status": "resolved", "id": anomaly_id}
