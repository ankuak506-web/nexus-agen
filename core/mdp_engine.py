"""
MDP Engine — Markov Decision Process Framework
================================================
Models each agent's decision-making as an MDP with:
  - States: Task context, confidence, conversation phase
  - Actions: Retrieve, Generate, Delegate, Reflect, Escalate
  - Transitions: Probabilistic state updates
  - Rewards: Task completion signals + quality metrics
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import time


# ──────────────────────────────────────────────
#  State & Action Definitions
# ──────────────────────────────────────────────

class AgentAction(Enum):
    """Available actions in the MDP."""
    RETRIEVE = "retrieve"          # Search knowledge base / vector store
    GENERATE = "generate"          # Produce new content via LLM
    DELEGATE = "delegate"          # Hand off to a specialist sub-agent
    REFLECT = "reflect"            # Self-evaluate current reasoning
    ESCALATE = "escalate"          # Escalate to user for clarification
    REFINE = "refine"              # Refine previous output
    EXPLORE = "explore"            # Explore alternative strategies
    SYNTHESIZE = "synthesize"      # Combine multiple sources


class ConversationPhase(Enum):
    """Phases of an agent conversation."""
    INIT = "init"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETE = "complete"


@dataclass
class MDPState:
    """Represents the current state in the MDP."""
    phase: ConversationPhase = ConversationPhase.INIT
    confidence: float = 0.5              # 0.0–1.0, agent confidence
    context_richness: float = 0.0        # how much context is available
    task_complexity: float = 0.5         # estimated complexity
    retrieval_count: int = 0             # number of retrievals done
    generation_count: int = 0            # number of generations done
    reflection_count: int = 0            # number of reflections done
    error_count: int = 0                 # errors encountered
    user_feedback_score: float = 0.0     # cumulative user satisfaction
    step: int = 0                        # current step number
    history: List[str] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for RL."""
        phase_onehot = [0.0] * len(ConversationPhase)
        phase_onehot[list(ConversationPhase).index(self.phase)] = 1.0
        return np.array(
            phase_onehot
            + [
                self.confidence,
                self.context_richness,
                self.task_complexity,
                min(self.retrieval_count / 10.0, 1.0),
                min(self.generation_count / 5.0, 1.0),
                min(self.reflection_count / 3.0, 1.0),
                min(self.error_count / 5.0, 1.0),
                self.user_feedback_score,
                min(self.step / 20.0, 1.0),
            ],
            dtype=np.float32,
        )

    @property
    def dim(self) -> int:
        return len(ConversationPhase) + 9

    def clone(self) -> "MDPState":
        return MDPState(
            phase=self.phase,
            confidence=self.confidence,
            context_richness=self.context_richness,
            task_complexity=self.task_complexity,
            retrieval_count=self.retrieval_count,
            generation_count=self.generation_count,
            reflection_count=self.reflection_count,
            error_count=self.error_count,
            user_feedback_score=self.user_feedback_score,
            step=self.step,
            history=list(self.history),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "confidence": round(self.confidence, 3),
            "context_richness": round(self.context_richness, 3),
            "task_complexity": round(self.task_complexity, 3),
            "retrieval_count": self.retrieval_count,
            "generation_count": self.generation_count,
            "reflection_count": self.reflection_count,
            "error_count": self.error_count,
            "user_feedback_score": round(self.user_feedback_score, 3),
            "step": self.step,
        }


# ──────────────────────────────────────────────
#  Transition Model
# ──────────────────────────────────────────────

class TransitionModel:
    """
    Probabilistic transition model for the MDP.
    Given (state, action) → next_state with learned probabilities.
    """

    # Default transition rules (phase progressions)
    PHASE_TRANSITIONS: Dict[ConversationPhase, Dict[AgentAction, ConversationPhase]] = {
        ConversationPhase.INIT: {
            AgentAction.RETRIEVE: ConversationPhase.UNDERSTANDING,
            AgentAction.GENERATE: ConversationPhase.PLANNING,
            AgentAction.EXPLORE: ConversationPhase.UNDERSTANDING,
        },
        ConversationPhase.UNDERSTANDING: {
            AgentAction.RETRIEVE: ConversationPhase.UNDERSTANDING,
            AgentAction.REFLECT: ConversationPhase.PLANNING,
            AgentAction.GENERATE: ConversationPhase.PLANNING,
            AgentAction.ESCALATE: ConversationPhase.UNDERSTANDING,
        },
        ConversationPhase.PLANNING: {
            AgentAction.GENERATE: ConversationPhase.EXECUTING,
            AgentAction.DELEGATE: ConversationPhase.EXECUTING,
            AgentAction.REFLECT: ConversationPhase.PLANNING,
            AgentAction.RETRIEVE: ConversationPhase.UNDERSTANDING,
        },
        ConversationPhase.EXECUTING: {
            AgentAction.GENERATE: ConversationPhase.EXECUTING,
            AgentAction.REFINE: ConversationPhase.REVIEWING,
            AgentAction.REFLECT: ConversationPhase.REVIEWING,
            AgentAction.SYNTHESIZE: ConversationPhase.REVIEWING,
            AgentAction.RETRIEVE: ConversationPhase.EXECUTING,
        },
        ConversationPhase.REVIEWING: {
            AgentAction.REFLECT: ConversationPhase.COMPLETE,
            AgentAction.REFINE: ConversationPhase.EXECUTING,
            AgentAction.GENERATE: ConversationPhase.COMPLETE,
        },
        ConversationPhase.COMPLETE: {},  # Terminal state
    }

    def __init__(self):
        # Learned transition probabilities (state_hash → action → next_state probs)
        self._transition_counts: Dict[str, Dict[str, Dict[str, int]]] = {}

    def transition(self, state: MDPState, action: AgentAction) -> MDPState:
        """Apply action to state and return next state."""
        next_state = state.clone()
        next_state.step += 1
        next_state.history.append(action.value)

        # Phase transition
        phase_map = self.PHASE_TRANSITIONS.get(state.phase, {})
        if action in phase_map:
            next_state.phase = phase_map[action]

        # State updates based on action
        if action == AgentAction.RETRIEVE:
            next_state.retrieval_count += 1
            next_state.context_richness = min(1.0, next_state.context_richness + 0.15)
            next_state.confidence = min(1.0, next_state.confidence + 0.05)

        elif action == AgentAction.GENERATE:
            next_state.generation_count += 1
            next_state.confidence = min(1.0, next_state.confidence + 0.1)

        elif action == AgentAction.REFLECT:
            next_state.reflection_count += 1
            # Reflection can either boost or reduce confidence
            if next_state.confidence > 0.7:
                next_state.confidence = min(1.0, next_state.confidence + 0.05)
            else:
                next_state.confidence = max(0.0, next_state.confidence - 0.05)

        elif action == AgentAction.DELEGATE:
            next_state.confidence = min(1.0, next_state.confidence + 0.15)

        elif action == AgentAction.REFINE:
            next_state.confidence = min(1.0, next_state.confidence + 0.08)

        elif action == AgentAction.EXPLORE:
            next_state.context_richness = min(1.0, next_state.context_richness + 0.1)

        elif action == AgentAction.SYNTHESIZE:
            next_state.confidence = min(1.0, next_state.confidence + 0.12)
            next_state.context_richness = min(1.0, next_state.context_richness + 0.08)

        elif action == AgentAction.ESCALATE:
            next_state.confidence = max(0.0, next_state.confidence - 0.1)

        # Record transition
        self._record_transition(state, action, next_state)

        return next_state

    def _record_transition(self, state: MDPState, action: AgentAction, next_state: MDPState):
        phase_key = state.phase.value
        action_key = action.value
        next_phase_key = next_state.phase.value

        if phase_key not in self._transition_counts:
            self._transition_counts[phase_key] = {}
        if action_key not in self._transition_counts[phase_key]:
            self._transition_counts[phase_key][action_key] = {}
        count = self._transition_counts[phase_key][action_key].get(next_phase_key, 0)
        self._transition_counts[phase_key][action_key][next_phase_key] = count + 1


# ──────────────────────────────────────────────
#  Reward Model
# ──────────────────────────────────────────────

class RewardModel:
    """Computes rewards for MDP transitions."""

    # Reward shaping constants
    TASK_COMPLETE_REWARD = 10.0
    PROGRESS_REWARD = 1.0
    REDUNDANT_PENALTY = -0.5
    ERROR_PENALTY = -2.0
    EFFICIENCY_BONUS = 0.5

    def compute_reward(
        self,
        prev_state: MDPState,
        action: AgentAction,
        next_state: MDPState,
        task_complete: bool = False,
        quality_score: float = 0.0,
    ) -> float:
        """
        Compute reward for a state transition.

        Args:
            prev_state: State before action
            action: Action taken
            next_state: State after action
            task_complete: Whether the task finished
            quality_score: External quality signal (0–1)
        """
        reward = 0.0

        # Task completion bonus
        if task_complete:
            reward += self.TASK_COMPLETE_REWARD * (0.5 + 0.5 * quality_score)
            # Efficiency bonus for fewer steps
            if next_state.step <= 5:
                reward += self.EFFICIENCY_BONUS * 2
            elif next_state.step <= 10:
                reward += self.EFFICIENCY_BONUS

        # Confidence improvement
        confidence_delta = next_state.confidence - prev_state.confidence
        reward += confidence_delta * 3.0

        # Context richness improvement
        context_delta = next_state.context_richness - prev_state.context_richness
        reward += context_delta * 2.0

        # Phase progression reward
        phases = list(ConversationPhase)
        phase_progress = phases.index(next_state.phase) - phases.index(prev_state.phase)
        if phase_progress > 0:
            reward += self.PROGRESS_REWARD * phase_progress

        # Penalize redundant actions
        if action.value in prev_state.history[-3:] if len(prev_state.history) >= 3 else []:
            reward += self.REDUNDANT_PENALTY

        # Penalize errors
        if next_state.error_count > prev_state.error_count:
            reward += self.ERROR_PENALTY

        return reward


# ──────────────────────────────────────────────
#  MDP Controller (orchestrates the full MDP)
# ──────────────────────────────────────────────

class MDPController:
    """
    High-level MDP controller that manages state, transitions, and rewards
    for a single agent episode.
    """

    def __init__(self):
        self.state = MDPState()
        self.transition_model = TransitionModel()
        self.reward_model = RewardModel()
        self.episode_rewards: List[float] = []
        self.episode_actions: List[AgentAction] = []
        self.episode_states: List[MDPState] = [self.state.clone()]
        self._start_time = time.time()

    def reset(self, task_complexity: float = 0.5) -> MDPState:
        """Reset MDP to initial state."""
        self.state = MDPState(task_complexity=task_complexity)
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_states = [self.state.clone()]
        self._start_time = time.time()
        return self.state

    def step(
        self,
        action: AgentAction,
        task_complete: bool = False,
        quality_score: float = 0.0,
    ) -> Tuple[MDPState, float, bool]:
        """
        Execute one MDP step.

        Returns:
            (next_state, reward, done)
        """
        prev_state = self.state.clone()
        self.state = self.transition_model.transition(self.state, action)

        reward = self.reward_model.compute_reward(
            prev_state, action, self.state, task_complete, quality_score
        )

        done = (
            self.state.phase == ConversationPhase.COMPLETE
            or task_complete
            or self.state.step >= 20  # Safety limit
        )

        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_states.append(self.state.clone())

        return self.state, reward, done

    def get_available_actions(self) -> List[AgentAction]:
        """Get valid actions for the current state."""
        phase_map = TransitionModel.PHASE_TRANSITIONS.get(self.state.phase, {})
        available = list(phase_map.keys())
        # Always allow escalate (except in COMPLETE)
        if self.state.phase != ConversationPhase.COMPLETE and AgentAction.ESCALATE not in available:
            available.append(AgentAction.ESCALATE)
        return available

    @property
    def total_reward(self) -> float:
        return sum(self.episode_rewards)

    @property
    def episode_length(self) -> int:
        return len(self.episode_actions)

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return a summary of the current episode."""
        return {
            "total_reward": round(self.total_reward, 3),
            "steps": self.episode_length,
            "actions": [a.value for a in self.episode_actions],
            "final_state": self.state.to_dict(),
            "final_confidence": round(self.state.confidence, 3),
            "duration_seconds": round(time.time() - self._start_time, 2),
        }
