# shared_stats.py
import time

# Rolling performance history (used by Performance UI charts)
METRIC_HISTORY = {
    "accuracy": [],
    "efficiency": [],
}

MAX_HISTORY = 50


def add_history(metric_name: str, value: float):
    METRIC_HISTORY[metric_name].append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "value": float(value),
    })
    if len(METRIC_HISTORY[metric_name]) > MAX_HISTORY:
        METRIC_HISTORY[metric_name].pop(0)


# Global agent statistics (updated every interaction)
AGENT_STATS = {}
