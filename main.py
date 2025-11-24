# main.py
import asyncio
import contextlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents import router as agents_router
from anomalies import router as anomalies_router
from training import router as training_router
from goals import router as goals_router
from performance import router as performance_router
from evolution import router as evolution_router
from collaborations import router as collaborations_router
from prompts import router as prompts_router
from shared_stats import METRIC_HISTORY, AGENT_STATS

from autonomous_loop import autonomous_loop  # NEW

app = FastAPI(
    title="Self-Evolving Agent Dashboard Backend",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Background task handle
_autonomy_task: asyncio.Task | None = None


@app.on_event("startup")
async def _startup_autonomous_loop():
    global _autonomy_task
    # Start the background autonomous loop
    _autonomy_task = asyncio.create_task(autonomous_loop())


@app.on_event("shutdown")
async def _shutdown_autonomous_loop():
    global _autonomy_task
    if _autonomy_task:
        _autonomy_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _autonomy_task


# Health endpoint (used by BackendTest + direct links)
@app.get("/health")
async def health_check():
    from datetime import datetime

    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# Root
@app.get("/")
async def root():
    return {"service": "Agent Dashboard Backend", "docs": "/docs"}


# Register routers with paths that match frontend
app.include_router(agents_router, prefix="/agents", tags=["Agents"])
app.include_router(anomalies_router, prefix="/anomalies", tags=["Anomalies"])
app.include_router(training_router, prefix="/training", tags=["Training"])
app.include_router(goals_router, prefix="/goals", tags=["Goals"])
app.include_router(performance_router, prefix="/performance", tags=["Performance"])
app.include_router(evolution_router, prefix="/evolution", tags=["Evolution"])
app.include_router(collaborations_router, prefix="/collaborations", tags=["Collaborations"])
app.include_router(prompts_router, prefix="/prompts", tags=["Prompts"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
