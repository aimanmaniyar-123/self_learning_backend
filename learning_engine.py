"""
learning_engine.py

Simple self-learning engine for your agents.

- Stores per-agent feedback examples (text + label)
- Converts text -> numeric features
- Trains a tiny online logistic regression model per agent
- Can predict for new inputs
- Optionally updates Agent.accuracy in agents.py
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class OnlineLinearModel:
    """Very small online logistic regression model."""
    input_dim: int
    learning_rate: float = 0.01
    weights: List[float] = field(default_factory=list)
    bias: float = 0.0

    def __post_init__(self):
        if not self.weights:
            self.weights = [0.0] * self.input_dim

    def _sigmoid(self, x: float) -> float:
        # Numerical safety
        if x < -50:
            x = -50
        elif x > 50:
            x = 50
        return 1.0 / (1.0 + math.exp(-x))

    def predict_proba(self, features: List[float]) -> float:
        z = sum(w * f for w, f in zip(self.weights, features)) + self.bias
        return self._sigmoid(z)

    def update(self, features: List[float], label: int):
        """
        One gradient descent step on a single example.
        label ∈ {0, 1}
        """
        p = self.predict_proba(features)
        error = label - p
        # Gradient: dL/dw = (label - p) * x, dL/db = (label - p)
        for i in range(self.input_dim):
            self.weights[i] += self.learning_rate * error * features[i]
        self.bias += self.learning_rate * error


class SelfLearningManager:
    """
    Manages:
    - Per-agent dataset
    - Per-agent online model
    - Simple evaluation + accuracy
    """

    def __init__(self):
        # agent_id -> list of (features, label)
        self.data: Dict[str, List[Tuple[List[float], int]]] = {}
        # agent_id -> OnlineLinearModel
        self.models: Dict[str, OnlineLinearModel] = {}

    # --------- Feature extraction ---------
    def text_to_features(self, text: str) -> List[float]:
        """
        Very simple numeric features from text:
        [length, word_count, avg_word_len, digit_count, upper_count, special_count]
        Enough to demonstrate learning, lightweight and dependency-free.
        """
        text = text or ""
        length = len(text)
        words = text.split()
        wc = len(words)
        avg_word_len = (length / wc) if wc > 0 else 0.0
        digits = sum(c.isdigit() for c in text)
        uppers = sum(c.isupper() for c in text)
        specials = sum(not c.isalnum() and not c.isspace() for c in text)

        return [
            float(length),
            float(wc),
            float(avg_word_len),
            float(digits),
            float(uppers),
            float(specials),
        ]

    # --------- Core operations ---------
    def _ensure_agent_model(self, agent_id: str, sample_features: List[float]):
        if agent_id not in self.models:
            self.models[agent_id] = OnlineLinearModel(input_dim=len(sample_features))

    def add_feedback(self, agent_id: str, text: str, label: int) -> Dict[str, float]:
        """
        Add a labeled example (0/1) and immediately update the model.

        Returns simple metrics: num_samples, accuracy.
        """
        label = 1 if label else 0
        features = self.text_to_features(text)
        self._ensure_agent_model(agent_id, features)

        # Store
        self.data.setdefault(agent_id, []).append((features, label))

        # Online update
        self.models[agent_id].update(features, label)

        # Evaluate on all data for this agent
        metrics = self.evaluate_agent(agent_id)
        # Update agent accuracy field, if possible
        self._update_agent_accuracy(agent_id, metrics["accuracy"])

        return metrics

    def train_agent(self, agent_id: str, epochs: int = 1) -> Dict[str, float]:
        """
        Offline-style training: run several passes over stored data.
        """
        examples = self.data.get(agent_id, [])
        if not examples:
            return {"num_samples": 0, "accuracy": 0.0}

        # Make sure model exists
        self._ensure_agent_model(agent_id, examples[0][0])

        model = self.models[agent_id]
        for _ in range(max(1, epochs)):
            for features, label in examples:
                model.update(features, label)

        metrics = self.evaluate_agent(agent_id)
        self._update_agent_accuracy(agent_id, metrics["accuracy"])
        return metrics

    def evaluate_agent(self, agent_id: str) -> Dict[str, float]:
        """
        Compute accuracy over stored examples.
        """
        examples = self.data.get(agent_id, [])
        if not examples:
            return {"num_samples": 0, "accuracy": 0.0}

        model = self.models.get(agent_id)
        if not model:
            return {"num_samples": len(examples), "accuracy": 0.0}

        correct = 0
        for features, label in examples:
            proba = model.predict_proba(features)
            pred = 1 if proba >= 0.5 else 0
            if pred == label:
                correct += 1

        acc = correct / len(examples) if examples else 0.0
        return {"num_samples": float(len(examples)), "accuracy": acc * 100.0}

    def predict(self, agent_id: str, text: str) -> Optional[Dict[str, float]]:
        """
        Predict probability & label for a new input.
        Returns None if the agent has no model/feedback yet.
        """
        model = self.models.get(agent_id)
        if not model:
            return None

        features = self.text_to_features(text)
        proba = model.predict_proba(features)
        label = 1 if proba >= 0.5 else 0
        return {
            "proba": proba,
            "label": float(label),
        }

    # --------- (Optional) sync to agents.py ---------
    def _update_agent_accuracy(self, agent_id: str, accuracy: float):
        """
        Try to update Agent.accuracy field in the in-memory store.

        Soft-imports agents to avoid circular import at module load.
        """
        try:
            from agents import AGENTS, Agent  # type: ignore
        except Exception:
            return

        agent = AGENTS.get(agent_id)
        if not agent:
            return

        data = agent.dict()
        data["accuracy"] = round(accuracy, 2)
        AGENTS[agent_id] = Agent(**data)


# Global singleton instance used by routers
SELF_LEARNING = SelfLearningManager()
