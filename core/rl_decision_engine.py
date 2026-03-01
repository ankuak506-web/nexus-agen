"""
RL Decision Engine — Reinforcement Learning Policy
====================================================
Policy-gradient based action selection for the MDP.
Uses a lightweight neural policy network with:
  - Softmax action selection
  - Experience replay buffer
  - Epsilon-greedy exploration
  - Q-value estimation
  
No training loops run at inference time — the policy is
applied directly to choose optimal actions.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import json
import os

from .mdp_engine import MDPState, AgentAction, MDPController


# ──────────────────────────────────────────────
#  Experience Replay
# ──────────────────────────────────────────────

@dataclass
class Experience:
    """Single experience tuple for replay."""
    state_vec: np.ndarray
    action_idx: int
    reward: float
    next_state_vec: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self._buffer: deque = deque(maxlen=capacity)

    def push(self, exp: Experience):
        self._buffer.append(exp)

    def sample(self, batch_size: int) -> List[Experience]:
        indices = np.random.choice(len(self._buffer), min(batch_size, len(self._buffer)), replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


# ──────────────────────────────────────────────
#  Lightweight Policy Network (NumPy-only)
# ──────────────────────────────────────────────

class PolicyNetwork:
    """
    A simple 2-layer softmax policy network using only NumPy.
    Maps state vectors → action probabilities.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W1 = np.random.randn(state_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, n_actions).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(n_actions, dtype=np.float32)

    def forward(self, state_vec: np.ndarray) -> np.ndarray:
        """Forward pass → action probabilities."""
        h = np.maximum(0, state_vec @ self.W1 + self.b1)  # ReLU
        logits = h @ self.W2 + self.b2
        # Stable softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / (np.sum(exp_logits) + 1e-8)
        return probs

    def update(self, state_vec: np.ndarray, action_idx: int, advantage: float, lr: float = 0.01):
        """
        Simple REINFORCE-style update (single sample).
        """
        # Forward
        h = np.maximum(0, state_vec @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / (np.sum(exp_logits) + 1e-8)

        # Gradient of log π(a|s) w.r.t. logits
        d_logits = -probs.copy()
        d_logits[action_idx] += 1.0
        d_logits *= advantage

        # Backprop through W2
        dW2 = np.outer(h, d_logits)
        db2 = d_logits

        # Backprop through ReLU
        dh = d_logits @ self.W2.T
        dh[h <= 0] = 0  # ReLU gradient

        # Backprop through W1
        dW1 = np.outer(state_vec, dh)
        db1 = dh

        # Apply gradients
        self.W1 += lr * dW1
        self.b1 += lr * db1
        self.W2 += lr * dW2
        self.b2 += lr * db2

    def save(self, path: str):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path: str):
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]


# ──────────────────────────────────────────────
#  Q-Value Estimator
# ──────────────────────────────────────────────

class QValueEstimator:
    """
    Tabular-ish Q-value estimator using feature hashing.
    Falls back to neural Q when enough data is available.
    """

    def __init__(self, n_actions: int, state_dim: int):
        self.n_actions = n_actions
        self.state_dim = state_dim
        # Simple linear Q: Q(s,a) = s^T @ W_a + b_a
        self.W = np.zeros((n_actions, state_dim), dtype=np.float32)
        self.b = np.zeros(n_actions, dtype=np.float32)
        self._update_count = 0

    def predict(self, state_vec: np.ndarray) -> np.ndarray:
        """Predict Q-values for all actions."""
        return self.W @ state_vec + self.b

    def update(self, state_vec: np.ndarray, action_idx: int, target: float, lr: float = 0.01):
        """TD update for Q-value."""
        pred = self.W[action_idx] @ state_vec + self.b[action_idx]
        error = target - pred
        self.W[action_idx] += lr * error * state_vec
        self.b[action_idx] += lr * error
        self._update_count += 1


# ──────────────────────────────────────────────
#  RL Decision Engine
# ──────────────────────────────────────────────

class RLDecisionEngine:
    """
    Main RL decision engine that selects actions using policy gradients
    combined with Q-value estimates and epsilon-greedy exploration.
    """

    ALL_ACTIONS = list(AgentAction)

    def __init__(
        self,
        state_dim: int = 15,  # MDPState.dim
        epsilon: float = 0.15,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.02,
        gamma: float = 0.99,
        learning_rate: float = 0.005,
        replay_capacity: int = 5000,
        model_path: Optional[str] = None,
    ):
        self.n_actions = len(self.ALL_ACTIONS)
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.lr = learning_rate

        self.policy = PolicyNetwork(state_dim, self.n_actions)
        self.q_estimator = QValueEstimator(self.n_actions, state_dim)
        self.replay_buffer = ReplayBuffer(replay_capacity)

        # Stats
        self._decision_count = 0
        self._explore_count = 0

        # Load pre-trained if available
        if model_path and os.path.exists(model_path):
            self.policy.load(model_path)

    def select_action(
        self,
        state: MDPState,
        available_actions: Optional[List[AgentAction]] = None,
    ) -> Tuple[AgentAction, Dict[str, Any]]:
        """
        Select an action using ε-greedy policy with Q-value tiebreaking.

        Returns:
            (selected_action, decision_info)
        """
        state_vec = state.to_vector()

        if available_actions is None:
            available_actions = self.ALL_ACTIONS

        available_indices = [self.ALL_ACTIONS.index(a) for a in available_actions]

        explored = False

        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(available_indices)
            explored = True
            self._explore_count += 1
        else:
            # Policy probabilities (filtered to available actions)
            probs = self.policy.forward(state_vec)
            q_values = self.q_estimator.predict(state_vec)

            # Combine policy probs with Q-values
            combined_scores = np.full(self.n_actions, -np.inf)
            for idx in available_indices:
                combined_scores[idx] = 0.6 * probs[idx] + 0.4 * self._normalize_q(q_values[idx])

            action_idx = int(np.argmax(combined_scores))

        self._decision_count += 1
        selected_action = self.ALL_ACTIONS[action_idx]

        # Build decision info
        probs = self.policy.forward(state_vec)
        q_vals = self.q_estimator.predict(state_vec)

        decision_info = {
            "action": selected_action.value,
            "explored": explored,
            "epsilon": round(self.epsilon, 4),
            "policy_probs": {
                self.ALL_ACTIONS[i].value: round(float(probs[i]), 4)
                for i in available_indices
            },
            "q_values": {
                self.ALL_ACTIONS[i].value: round(float(q_vals[i]), 4)
                for i in available_indices
            },
            "decision_number": self._decision_count,
            "explore_rate": round(self._explore_count / max(1, self._decision_count), 4),
        }

        return selected_action, decision_info

    def record_experience(
        self,
        state: MDPState,
        action: AgentAction,
        reward: float,
        next_state: MDPState,
        done: bool,
    ):
        """Record an experience and perform online updates."""
        state_vec = state.to_vector()
        next_state_vec = next_state.to_vector()
        action_idx = self.ALL_ACTIONS.index(action)

        exp = Experience(state_vec, action_idx, reward, next_state_vec, done)
        self.replay_buffer.push(exp)

        # Online Q-value update (TD(0))
        if done:
            target = reward
        else:
            next_q = self.q_estimator.predict(next_state_vec)
            target = reward + self.gamma * np.max(next_q)

        self.q_estimator.update(state_vec, action_idx, target, self.lr)

        # Policy gradient update using advantage
        q_current = self.q_estimator.predict(state_vec)
        baseline = np.mean(q_current)
        advantage = target - baseline
        self.policy.update(state_vec, action_idx, advantage, self.lr * 0.1)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def batch_update(self, batch_size: int = 32):
        """Perform a batch update from the replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        for exp in batch:
            if exp.done:
                target = exp.reward
            else:
                next_q = self.q_estimator.predict(exp.next_state_vec)
                target = exp.reward + self.gamma * np.max(next_q)

            self.q_estimator.update(exp.state_vec, exp.action_idx, target, self.lr * 0.5)

            q_current = self.q_estimator.predict(exp.state_vec)
            advantage = target - np.mean(q_current)
            self.policy.update(exp.state_vec, exp.action_idx, advantage, self.lr * 0.05)

    def _normalize_q(self, q: float) -> float:
        """Sigmoid normalization for Q-values."""
        return 1.0 / (1.0 + np.exp(-q))

    def save_model(self, path: str):
        self.policy.save(path)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_decisions": self._decision_count,
            "total_explorations": self._explore_count,
            "explore_rate": round(self._explore_count / max(1, self._decision_count), 4),
            "epsilon": round(self.epsilon, 4),
            "replay_buffer_size": len(self.replay_buffer),
            "q_updates": self.q_estimator._update_count,
        }
