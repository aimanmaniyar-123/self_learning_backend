# routers/performance.py
import psutil
from fastapi import APIRouter
from shared_stats import METRIC_HISTORY, add_history
from shared_stats import AGENT_STATS

router = APIRouter()


@router.get("/system")
async def get_system_metrics():
    cpu = psutil.cpu_percent(interval=0.05)
    mem = psutil.virtual_memory().percent

    total_agents = len(AGENT_STATS)

    avg_success_rate = (
        sum(a["success_rate"] for a in AGENT_STATS.values()) / total_agents
        if total_agents > 0 else 0
    )

    avg_response_time = (
        sum(a["avg_response_time"] for a in AGENT_STATS.values()) / total_agents
        if total_agents > 0 else 0
    )

    return {
        "cpu_usage": cpu,
        "memory_usage": mem,
        "total_agents": total_agents,
        "active_agents": total_agents,
        "overall_success_rate": round(avg_success_rate, 2),
        "avg_response_time": round(avg_response_time, 2),
        "queue_length": 0,
    }


@router.get("/metrics")
async def get_performance_history():
    # CPU drives accuracy trend
    add_history("accuracy", psutil.cpu_percent(interval=0.01))

    # RAM drives efficiency
    add_history("efficiency", psutil.virtual_memory().percent)

    return METRIC_HISTORY
