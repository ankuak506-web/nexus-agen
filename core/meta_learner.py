"""
Meta-Learner — MAML-Style Few-Shot Adaptation
===============================================
Enables rapid task adaptation through:
  - Task-specific parameter snapshots
  - Gradient-based inner-loop adaptation
  - Cross-project knowledge transfer
  - Learning rate meta-optimization

The meta-learner wraps the RL policy and adapts it
to new task domains with minimal examples.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
import json
import time


# ──────────────────────────────────────────────
#  Task Representation
# ──────────────────────────────────────────────

@dataclass
class TaskProfile:
    """Profile describing a specific task domain."""
    task_id: str
    domain: str                          # e.g., "finance", "document", "video"
    complexity: float = 0.5
    typical_steps: int = 5
    preferred_actions: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    total_episodes: int = 0
    avg_reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Convert task profile to embedding vector."""
        domain_map = {
            "finance": [1, 0, 0, 0, 0, 0],
            "document": [0, 1, 0, 0, 0, 0],
            "video": [0, 0, 1, 0, 0, 0],
            "sql": [0, 0, 0, 1, 0, 0],
            "content": [0, 0, 0, 0, 1, 0],
            "general": [0, 0, 0, 0, 0, 1],
        }
        domain_vec = domain_map.get(self.domain, [0, 0, 0, 0, 0, 1])
        return np.array(
            domain_vec + [
                self.complexity,
                self.typical_steps / 20.0,
                self.success_rate,
                min(self.total_episodes / 100.0, 1.0),
                self.avg_reward / 10.0,
            ],
            dtype=np.float32,
        )


# ──────────────────────────────────────────────
#  Parameter Snapshot
# ──────────────────────────────────────────────

@dataclass
class ParameterSnapshot:
    """Stores a snapshot of model parameters for a specific task."""
    task_id: str
    timestamp: float
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    performance: float = 0.0
    episodes_trained: int = 0


# ──────────────────────────────────────────────
#  Meta-Learner
# ──────────────────────────────────────────────

class MetaLearner:
    """
    MAML-style meta-learner for rapid task adaptation.
    
    Key Mechanisms:
    1. Maintains a meta-policy (shared initialization)
    2. Creates task-specific adaptations via inner-loop updates
    3. Transfers knowledge across task domains
    4. Meta-optimizes the learning rate
    """

    def __init__(
        self,
        state_dim: int = 15,
        n_actions: int = 8,
        hidden_dim: int = 64,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        n_inner_steps: int = 5,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_inner_steps = n_inner_steps

        # Meta-parameters (shared initialization)
        self.meta_W1 = np.random.randn(state_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_dim)
        self.meta_b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.meta_W2 = np.random.randn(hidden_dim, n_actions).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.meta_b2 = np.zeros(n_actions, dtype=np.float32)

        # Task registry
        self.task_profiles: Dict[str, TaskProfile] = {}
        self.task_snapshots: Dict[str, ParameterSnapshot] = {}

        # Adaptation history
        self._adaptation_log: List[Dict[str, Any]] = []

        # Meta-learning rate adaptation
        self._lr_history: List[float] = [inner_lr]

    def register_task(self, profile: TaskProfile):
        """Register a new task domain."""
        self.task_profiles[profile.task_id] = profile

    def adapt_to_task(
        self,
        task_id: str,
        support_set: Optional[List[Tuple[np.ndarray, int, float]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Adapt meta-parameters to a specific task using inner-loop updates.
        
        Args:
            task_id: ID of the target task
            support_set: List of (state_vec, action_idx, reward) tuples for adaptation
            
        Returns:
            Adapted parameters {W1, b1, W2, b2}
        """
        # Start from meta-parameters or previous snapshot
        if task_id in self.task_snapshots:
            snap = self.task_snapshots[task_id]
            W1 = snap.W1.copy()
            b1 = snap.b1.copy()
            W2 = snap.W2.copy()
            b2 = snap.b2.copy()
        else:
            W1 = self.meta_W1.copy()
            b1 = self.meta_b1.copy()
            W2 = self.meta_W2.copy()
            b2 = self.meta_b2.copy()

        # If we have a support set, do inner-loop adaptation
        if support_set and len(support_set) > 0:
            for step in range(self.n_inner_steps):
                for state_vec, action_idx, reward in support_set:
                    # Forward pass
                    h = np.maximum(0, state_vec @ W1 + b1)
                    logits = h @ W2 + b2
                    logits = logits - np.max(logits)
                    exp_logits = np.exp(logits)
                    probs = exp_logits / (np.sum(exp_logits) + 1e-8)

                    # REINFORCE gradient with reward as advantage
                    d_logits = -probs.copy()
                    d_logits[action_idx] += 1.0
                    d_logits *= reward

                    # Backprop
                    dW2 = np.outer(h, d_logits)
                    db2 = d_logits
                    dh = d_logits @ W2.T
                    dh[h <= 0] = 0
                    dW1 = np.outer(state_vec, dh)
                    db1 = dh

                    # Inner-loop update
                    lr = self._get_adapted_lr(task_id)
                    W1 += lr * dW1
                    b1 += lr * db1
                    W2 += lr * dW2
                    b2 += lr * db2

        # Save snapshot
        self.task_snapshots[task_id] = ParameterSnapshot(
            task_id=task_id,
            timestamp=time.time(),
            W1=W1.copy(),
            b1=b1.copy(),
            W2=W2.copy(),
            b2=b2.copy(),
        )

        self._adaptation_log.append({
            "task_id": task_id,
            "timestamp": time.time(),
            "support_size": len(support_set) if support_set else 0,
            "inner_steps": self.n_inner_steps,
            "lr_used": self._get_adapted_lr(task_id),
        })

        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def meta_update(self, task_gradients: Dict[str, Dict[str, np.ndarray]]):
        """
        Outer-loop meta-update: update meta-parameters using gradients
        from multiple task adaptations.
        
        Args:
            task_gradients: {task_id: {W1_grad, b1_grad, W2_grad, b2_grad}}
        """
        if not task_gradients:
            return

        n_tasks = len(task_gradients)

        # Average gradients across tasks
        avg_dW1 = np.zeros_like(self.meta_W1)
        avg_db1 = np.zeros_like(self.meta_b1)
        avg_dW2 = np.zeros_like(self.meta_W2)
        avg_db2 = np.zeros_like(self.meta_b2)

        for grads in task_gradients.values():
            avg_dW1 += grads.get("W1_grad", np.zeros_like(self.meta_W1))
            avg_db1 += grads.get("b1_grad", np.zeros_like(self.meta_b1))
            avg_dW2 += grads.get("W2_grad", np.zeros_like(self.meta_W2))
            avg_db2 += grads.get("b2_grad", np.zeros_like(self.meta_b2))

        avg_dW1 /= n_tasks
        avg_db1 /= n_tasks
        avg_dW2 /= n_tasks
        avg_db2 /= n_tasks

        # Meta-parameter update
        self.meta_W1 += self.meta_lr * avg_dW1
        self.meta_b1 += self.meta_lr * avg_db1
        self.meta_W2 += self.meta_lr * avg_dW2
        self.meta_b2 += self.meta_lr * avg_db2

    def transfer_knowledge(self, source_task: str, target_task: str, blend: float = 0.3):
        """
        Transfer learned parameters from source to target task.
        
        Args:
            source_task: Task to transfer from
            target_task: Task to transfer to
            blend: How much to blend source into target (0=none, 1=full)
        """
        if source_task not in self.task_snapshots:
            return

        source = self.task_snapshots[source_task]

        if target_task in self.task_snapshots:
            target = self.task_snapshots[target_task]
            # Blend parameters
            new_W1 = (1 - blend) * target.W1 + blend * source.W1
            new_b1 = (1 - blend) * target.b1 + blend * source.b1
            new_W2 = (1 - blend) * target.W2 + blend * source.W2
            new_b2 = (1 - blend) * target.b2 + blend * source.b2
        else:
            new_W1 = (1 - blend) * self.meta_W1 + blend * source.W1
            new_b1 = (1 - blend) * self.meta_b1 + blend * source.b1
            new_W2 = (1 - blend) * self.meta_W2 + blend * source.W2
            new_b2 = (1 - blend) * self.meta_b2 + blend * source.b2

        self.task_snapshots[target_task] = ParameterSnapshot(
            task_id=target_task,
            timestamp=time.time(),
            W1=new_W1,
            b1=new_b1,
            W2=new_W2,
            b2=new_b2,
        )

    def get_task_similarity(self, task_a: str, task_b: str) -> float:
        """Compute similarity between two task profiles."""
        if task_a not in self.task_profiles or task_b not in self.task_profiles:
            return 0.0

        vec_a = self.task_profiles[task_a].to_vector()
        vec_b = self.task_profiles[task_b].to_vector()

        # Cosine similarity
        dot = np.dot(vec_a, vec_b)
        norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if norm < 1e-8:
            return 0.0
        return float(dot / norm)

    def find_best_source_task(self, target_task: str) -> Optional[str]:
        """Find the most similar task to transfer from."""
        if target_task not in self.task_profiles:
            return None

        best_task = None
        best_sim = -1.0

        for task_id in self.task_profiles:
            if task_id == target_task:
                continue
            if task_id not in self.task_snapshots:
                continue
            sim = self.get_task_similarity(target_task, task_id)
            if sim > best_sim:
                best_sim = sim
                best_task = task_id

        return best_task if best_sim > 0.3 else None

    def _get_adapted_lr(self, task_id: str) -> float:
        """Get adapted learning rate for a specific task."""
        if task_id in self.task_profiles:
            profile = self.task_profiles[task_id]
            # Higher LR for less-trained tasks, lower for well-trained
            experience_factor = 1.0 / (1.0 + profile.total_episodes * 0.01)
            return self.inner_lr * experience_factor
        return self.inner_lr

    def update_task_stats(self, task_id: str, reward: float, success: bool):
        """Update task profile statistics after an episode."""
        if task_id not in self.task_profiles:
            return

        profile = self.task_profiles[task_id]
        profile.total_episodes += 1
        # Exponential moving average
        alpha = 0.1
        profile.avg_reward = (1 - alpha) * profile.avg_reward + alpha * reward
        profile.success_rate = (1 - alpha) * profile.success_rate + alpha * (1.0 if success else 0.0)

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Return a summary of all task adaptations."""
        return {
            "registered_tasks": len(self.task_profiles),
            "adapted_tasks": len(self.task_snapshots),
            "total_adaptations": len(self._adaptation_log),
            "tasks": {
                tid: {
                    "domain": p.domain,
                    "success_rate": round(p.success_rate, 3),
                    "episodes": p.total_episodes,
                    "avg_reward": round(p.avg_reward, 3),
                }
                for tid, p in self.task_profiles.items()
            },
        }
