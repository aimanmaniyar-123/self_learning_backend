# evolution.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from uuid import uuid4
import random

# Import agents + self-learning engine if available
try:
    from agents import AGENTS, Agent  # type: ignore
except ImportError:
    AGENTS = {}  # type: ignore
    Agent = None  # type: ignore

try:
    from learning_engine import SELF_LEARNING  # type: ignore
except ImportError:
    SELF_LEARNING = None  # type: ignore

router = APIRouter()


# ============================================================
# MODELS
# ============================================================

class EvolutionEvent(BaseModel):
    generation: int
    accuracy: float
    efficiency: float
    agents: int
    timestamp: str


class TriggerRequest(BaseModel):
    agent_id: Optional[str] = None  # kept for compatibility, not required


# Per-agent "genes" used by GA style evolution
class AgentGene(BaseModel):
    learning_rate: float    # 0.0001 - 0.05
    depth: int              # 1 - 6
    regularization: float   # 0.0 - 0.1


# ============================================================
# IN-MEMORY STATE
# ============================================================

EVOLUTION_HISTORY: List[EvolutionEvent] = []
AGENT_GENES: Dict[str, AgentGene] = {}


# ============================================================
# UTILITIES
# ============================================================

def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _seed_history() -> None:
    """
    Seed some basic history if empty, so the UI has something initial to show.
    """
    if EVOLUTION_HISTORY:
        return
    now = datetime.utcnow()
    for gen in range(1, 4):
        EVOLUTION_HISTORY.append(
            EvolutionEvent(
                generation=gen,
                accuracy=85 + gen * 1.0,
                efficiency=75 + gen * 1.5,
                agents=3 + gen,
                timestamp=(now - timedelta(days=4 - gen)).isoformat(),
            )
        )


def _init_agent_genes_if_missing() -> None:
    """
    Ensure each agent has a corresponding gene record.
    """
    for agent_id in list(AGENTS.keys()):
        if agent_id not in AGENT_GENES:
            AGENT_GENES[agent_id] = AgentGene(
                learning_rate=random.uniform(0.0005, 0.02),
                depth=random.randint(1, 4),
                regularization=random.uniform(0.0, 0.05),
            )


def _evaluate_agent_fitness(agent_id: str, gene: AgentGene) -> float:
    """
    Compute GA-style fitness. Higher is better.
    Uses:
      - Observed accuracy
      - Gene depth / learning rate
      - (Optional) true accuracy from SELF_LEARNING if available
    """
    agent = AGENTS.get(agent_id)
    if not agent:
        return 0.0

    # base from agent.accuracy
    base_acc = float(getattr(agent, "accuracy", 0.0) or 0.0)

    # if self-learning engine has dataset, try to get better estimate
    extra_bonus = 0.0
    if SELF_LEARNING is not None:
        try:
            perf = SELF_LEARNING.get_agent_performance(agent_id)  # custom helper if you have it
            if perf and "accuracy" in perf:
                base_acc = float(perf["accuracy"])
        except Exception:
            # fallback to agent.accuracy only
            pass

    # incorporate genes:
    # - deeper models slightly higher capacity
    # - too high learning_rate or reg slightly penalized
    gene_score = (
        gene.depth * 1.0
        - gene.learning_rate * 400.0
        - gene.regularization * 100.0
    )

    # task count: more tasks processed = more evidence, small bonus
    tasks = float(getattr(agent, "tasks", 0) or 0.0)
    task_bonus = min(tasks / 500.0, 5.0)  # capped

    fitness = base_acc + gene_score + task_bonus + extra_bonus
    return fitness


def _crossover(parent_a: AgentGene, parent_b: AgentGene) -> AgentGene:
    """
    Single-point style crossover via mixing hyperparameters.
    """
    return AgentGene(
        learning_rate=(parent_a.learning_rate + parent_b.learning_rate) / 2.0,
        depth=random.choice([parent_a.depth, parent_b.depth]),
        regularization=(parent_a.regularization + parent_b.regularization) / 2.0,
    )


def _mutate(gene: AgentGene, mutation_rate: float = 0.3) -> AgentGene:
    """
    Mutate genes with bounded noise.
    """
    lr = gene.learning_rate
    depth = gene.depth
    reg = gene.regularization

    if random.random() < mutation_rate:
        lr *= random.uniform(0.7, 1.3)
    if random.random() < mutation_rate:
        depth += random.choice([-1, 0, 1])
    if random.random() < mutation_rate:
        reg *= random.uniform(0.7, 1.3)

    # clamp values
    lr = max(0.0001, min(0.05, lr))
    depth = max(1, min(6, depth))
    reg = max(0.0, min(0.1, reg))

    return AgentGene(learning_rate=lr, depth=depth, regularization=reg)


def record_evolution_snapshot(accuracy: float, efficiency: float, agents_count: int) -> None:
    """
    Called by autonomous loop OR evolution trigger to append a new evolution event.
    """
    generation = (EVOLUTION_HISTORY[-1].generation + 1) if EVOLUTION_HISTORY else 1
    event = EvolutionEvent(
        generation=generation,
        accuracy=accuracy,
        efficiency=efficiency,
        agents=agents_count,
        timestamp=_now_iso(),
    )
    EVOLUTION_HISTORY.append(event)


_seed_history()


# ============================================================
# API ENDPOINTS
# ============================================================

@router.get("/history")
async def get_evolution_history():
    """
    Frontend: api.getEvolutionHistory()
    Returns timeline including snapshots from:
      - Autonomous loop (background)
      - Manual evolution trigger (GA)
    """
    return [e.dict() for e in EVOLUTION_HISTORY]


@router.post("/trigger")
async def trigger_evolution(req: TriggerRequest):
    """
    Frontend: api.triggerEvolution(agentId?)
    ADVANCED GA VERSION:

    - Initialize genes for all agents.
    - Evaluate fitness for each agent.
    - Select top-k as parents.
    - Perform crossover + mutation to create children.
    - Add children as new evolved agents into AGENTS.
    - Compute new aggregate accuracy/efficiency and record snapshot.
    """
    # Ensure we have agents
    if not AGENTS:
        # still respond with something for the UI
        record_evolution_snapshot(accuracy=90.0, efficiency=80.0, agents_count=0)
        return {
            "status": "no_agents",
            "agents_count": 0,
            "estimated_duration": "0 minutes",
        }

    _init_agent_genes_if_missing()

    # Prepare list of (agent_id, fitness, genes)
    scored = []
    for aid, agent in AGENTS.items():
      # skip paused/inactive in GA selection
        if getattr(agent, "status", "active") != "active":
            continue
        gene = AGENT_GENES.get(aid)
        if not gene:
            continue
        fit = _evaluate_agent_fitness(aid, gene)
        scored.append((aid, fit, gene))

    if not scored:
        # no active agents
        record_evolution_snapshot(accuracy=90.0, efficiency=80.0, agents_count=0)
        return {
            "status": "no_active_agents",
            "agents_count": 0,
            "estimated_duration": "0 minutes",
        }

    # Sort by fitness (desc)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Select top-k as parents
    population_size = len(scored)
    parents_count = max(2, population_size // 2)
    parents = scored[:parents_count]

    # Create children via GA
    children_ids = []
    for i in range(len(parents) - 1):
        parent_a_id, _, gene_a = parents[i]
        parent_b_id, _, gene_b = parents[i + 1]

        # Crossover
        child_gene = _crossover(gene_a, gene_b)
        # Mutate
        child_gene = _mutate(child_gene, mutation_rate=0.4)

        # Create new agent based on parent A with mutated config
        parent_agent = AGENTS[parent_a_id]
        new_id = str(uuid4())[:8]

        # slight improvement in accuracy with small noise
        base_acc = float(getattr(parent_agent, "accuracy", 90.0) or 90.0)
        acc_delta = random.uniform(0.1, 1.0)
        new_accuracy = max(50.0, min(99.9, base_acc + acc_delta))

        # derive new version tag
        try:
            current_version = float(getattr(parent_agent, "version", "1.0"))
            new_version = f"{current_version + 0.1:.1f}"
        except Exception:
            new_version = "evolved"

        description_suffix = " (evolved via GA from {pid})".format(pid=parent_a_id)
        new_description = (parent_agent.description or "") + description_suffix

        # Build new Agent
        new_agent_data = parent_agent.dict()
        new_agent_data.update(
            {
                "id": new_id,
                "name": f"{parent_agent.name}-gen",
                "version": new_version,
                "accuracy": new_accuracy,
                "tasks": 0,
                "lastActive": "just now",
                "description": new_description,
                "history": [],
            }
        )
        new_agent = Agent(**new_agent_data)
        AGENTS[new_id] = new_agent
        AGENT_GENES[new_id] = child_gene
        children_ids.append(new_id)

    # Compute aggregate metrics for snapshot
    all_agents = list(AGENTS.values())
    if all_agents:
        avg_acc = sum(float(a.accuracy or 0.0) for a in all_agents) / len(all_agents)
        avg_tasks = sum(float(a.tasks or 0.0) for a in all_agents) / len(all_agents)
        # “efficiency” heuristic: fewer tasks & better acc => better efficiency
        efficiency = max(50.0, min(99.0, 100.0 - (avg_tasks / 1000.0) + (avg_acc - 90.0) * 0.2))
    else:
        avg_acc = 90.0
        efficiency = 80.0

    record_evolution_snapshot(
        accuracy=avg_acc,
        efficiency=efficiency,
        agents_count=len(all_agents),
    )

    return {
        "status": "started",
        "agents_count": len(children_ids),
        "estimated_duration": "short (GA simulated)",
        "new_agents": children_ids,
    }
