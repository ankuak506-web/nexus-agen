"""
Agentic Controller — Goal Manager + Planner + Reflector
=========================================================
The orchestration layer that makes agents truly agentic:
  - Goal Manager: Decompose user intents into sub-goals
  - Planner: MDP-guided strategy selection with RL
  - Executor: Action dispatch with meta-learning adaptation
  - Reflector: Self-evaluate and adjust strategy
  - Stuck Detector: Identify and recover from stuck states

This controller wraps MDP, RL, Meta-Learning, RAG, and Memory
into a unified decision loop.
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from .mdp_engine import MDPController, MDPState, AgentAction, ConversationPhase
from .rl_decision_engine import RLDecisionEngine
from .meta_learner import MetaLearner, TaskProfile
from .rag_engine import RAGEngine
from .memory import AgentMemory


# ──────────────────────────────────────────────
#  Goal System
# ──────────────────────────────────────────────

class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    ACHIEVED = "achieved"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Goal:
    """A single agent goal."""
    goal_id: str
    description: str
    priority: float = 0.5
    status: GoalStatus = GoalStatus.PENDING
    sub_goals: List["Goal"] = field(default_factory=list)
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "strategy": self.strategy,
            "sub_goals": [sg.to_dict() for sg in self.sub_goals],
        }


class GoalManager:
    """Decomposes user intents into hierarchical goals."""

    def __init__(self):
        self.goals: List[Goal] = []
        self._goal_counter = 0

    def create_goal(
        self,
        description: str,
        priority: float = 0.5,
        parent_goal: Optional[str] = None,
    ) -> Goal:
        """Create a new goal."""
        self._goal_counter += 1
        goal = Goal(
            goal_id=f"goal_{self._goal_counter}",
            description=description,
            priority=priority,
        )

        if parent_goal:
            parent = self._find_goal(parent_goal)
            if parent:
                parent.sub_goals.append(goal)
        else:
            self.goals.append(goal)

        return goal

    def decompose_goal(self, goal_id: str, sub_descriptions: List[str]) -> List[Goal]:
        """Decompose a goal into sub-goals."""
        sub_goals = []
        for desc in sub_descriptions:
            sg = self.create_goal(desc, parent_goal=goal_id)
            sub_goals.append(sg)
        return sub_goals

    def update_goal(self, goal_id: str, progress: float = None, status: GoalStatus = None):
        """Update goal progress or status."""
        goal = self._find_goal(goal_id)
        if not goal:
            return

        if progress is not None:
            goal.progress = min(1.0, progress)
        if status is not None:
            goal.status = status
            if status == GoalStatus.ACHIEVED:
                goal.progress = 1.0
                goal.completed_at = time.time()

        # Auto-update parent based on sub-goals
        self._propagate_progress()

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals, sorted by priority."""
        all_goals = self._flatten_goals()
        active = [g for g in all_goals if g.status in (GoalStatus.ACTIVE, GoalStatus.PENDING)]
        return sorted(active, key=lambda g: g.priority, reverse=True)

    def get_goal_tree(self) -> List[Dict[str, Any]]:
        """Get the full goal hierarchy."""
        return [g.to_dict() for g in self.goals]

    def _find_goal(self, goal_id: str) -> Optional[Goal]:
        for g in self._flatten_goals():
            if g.goal_id == goal_id:
                return g
        return None

    def _flatten_goals(self) -> List[Goal]:
        result = []
        stack = list(self.goals)
        while stack:
            g = stack.pop(0)
            result.append(g)
            stack.extend(g.sub_goals)
        return result

    def _propagate_progress(self):
        """Propagate sub-goal progress to parents."""
        for goal in self.goals:
            if goal.sub_goals:
                total = sum(sg.progress for sg in goal.sub_goals)
                goal.progress = total / len(goal.sub_goals)
                if all(sg.status == GoalStatus.ACHIEVED for sg in goal.sub_goals):
                    goal.status = GoalStatus.ACHIEVED
                    goal.completed_at = time.time()


# ──────────────────────────────────────────────
#  Stuck Detector
# ──────────────────────────────────────────────

class StuckDetector:
    """Detect when the agent is stuck in a loop or making no progress."""

    def __init__(self, window_size: int = 5, min_progress: float = 0.05):
        self.window_size = window_size
        self.min_progress = min_progress
        self._action_history: List[str] = []
        self._confidence_history: List[float] = []
        self._stuck_count = 0

    def check(self, state: MDPState, action: AgentAction) -> Dict[str, Any]:
        """Check if the agent is stuck."""
        self._action_history.append(action.value)
        self._confidence_history.append(state.confidence)

        is_stuck = False
        reason = ""

        if len(self._action_history) >= self.window_size:
            recent_actions = self._action_history[-self.window_size:]
            recent_confidence = self._confidence_history[-self.window_size:]

            # Check: repeating same action
            if len(set(recent_actions)) == 1:
                is_stuck = True
                reason = f"Repeating action '{recent_actions[0]}' {self.window_size} times"

            # Check: oscillating between two actions
            if len(set(recent_actions)) == 2 and len(recent_actions) >= 4:
                pattern = recent_actions[-4:]
                if pattern[0] == pattern[2] and pattern[1] == pattern[3]:
                    is_stuck = True
                    reason = f"Oscillating between '{pattern[0]}' and '{pattern[1]}'"

            # Check: no confidence improvement
            conf_delta = recent_confidence[-1] - recent_confidence[0]
            if abs(conf_delta) < self.min_progress and len(recent_actions) >= self.window_size:
                is_stuck = True
                reason = f"No confidence improvement over {self.window_size} steps"

        if is_stuck:
            self._stuck_count += 1

        return {
            "is_stuck": is_stuck,
            "reason": reason,
            "stuck_count": self._stuck_count,
            "recovery_suggestion": self._suggest_recovery(reason) if is_stuck else None,
        }

    def _suggest_recovery(self, reason: str) -> str:
        """Suggest a recovery action."""
        if "Repeating" in reason:
            return "Try a different action type: explore or escalate"
        elif "Oscillating" in reason:
            return "Break the cycle: reflect on current state and re-plan"
        elif "No confidence" in reason:
            return "Retrieve more context or delegate to a specialist"
        return "Reset strategy and re-evaluate goals"

    def reset(self):
        self._action_history = []
        self._confidence_history = []


# ──────────────────────────────────────────────
#  Self-Reflector
# ──────────────────────────────────────────────

class SelfReflector:
    """
    Self-evaluation and strategy adjustment module.
    Analyzes agent performance and suggests improvements.
    """

    def __init__(self):
        self._reflections: List[Dict[str, Any]] = []

    def reflect(
        self,
        state: MDPState,
        goals: List[Goal],
        episode_rewards: List[float],
        stuck_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform self-reflection on current agent state.
        
        Returns:
            Reflection report with insights and strategy adjustments
        """
        reflection = {
            "timestamp": time.time(),
            "state_assessment": self._assess_state(state),
            "goal_assessment": self._assess_goals(goals),
            "performance_assessment": self._assess_performance(episode_rewards),
            "stuck_assessment": stuck_info,
            "strategy_adjustments": [],
            "insights": [],
        }

        # Generate strategy adjustments
        if state.confidence < 0.3:
            reflection["strategy_adjustments"].append({
                "type": "boost_confidence",
                "suggestion": "Retrieve more context before generating",
                "priority": "high",
            })

        if state.context_richness < 0.3:
            reflection["strategy_adjustments"].append({
                "type": "enrich_context",
                "suggestion": "Perform additional RAG retrieval",
                "priority": "high",
            })

        if stuck_info.get("is_stuck"):
            reflection["strategy_adjustments"].append({
                "type": "unstuck",
                "suggestion": stuck_info.get("recovery_suggestion", "Change approach"),
                "priority": "critical",
            })

        if episode_rewards and sum(episode_rewards[-3:]) < 0:
            reflection["strategy_adjustments"].append({
                "type": "improve_quality",
                "suggestion": "Recent actions have negative rewards; consider reflecting and refining",
                "priority": "medium",
            })

        # Generate insights
        if state.retrieval_count > 5 and state.confidence < 0.5:
            reflection["insights"].append(
                "Many retrievals but low confidence — the knowledge base may lack relevant content"
            )

        if state.generation_count > 3 and state.reflection_count == 0:
            reflection["insights"].append(
                "Multiple generations without reflection — consider adding quality checks"
            )

        self._reflections.append(reflection)
        return reflection

    def _assess_state(self, state: MDPState) -> Dict[str, str]:
        """Assess the current MDP state."""
        assessments = {}

        # Confidence level
        if state.confidence >= 0.8:
            assessments["confidence"] = "HIGH — ready to finalize"
        elif state.confidence >= 0.5:
            assessments["confidence"] = "MODERATE — acceptable but could improve"
        else:
            assessments["confidence"] = "LOW — needs more context or reflection"

        # Phase assessment
        phase_expectations = {
            ConversationPhase.INIT: "Just started — need to understand the task",
            ConversationPhase.UNDERSTANDING: "Gathering context — retrieve relevant information",
            ConversationPhase.PLANNING: "Planning approach — formulate strategy",
            ConversationPhase.EXECUTING: "Executing plan — generating results",
            ConversationPhase.REVIEWING: "Reviewing output — ensure quality",
            ConversationPhase.COMPLETE: "Task complete",
        }
        assessments["phase"] = phase_expectations.get(state.phase, "Unknown phase")

        return assessments

    def _assess_goals(self, goals: List[Goal]) -> Dict[str, Any]:
        """Assess goal progress."""
        total = len(goals)
        if total == 0:
            return {"status": "No goals set"}

        achieved = sum(1 for g in goals if g.status == GoalStatus.ACHIEVED)
        active = sum(1 for g in goals if g.status == GoalStatus.ACTIVE)
        blocked = sum(1 for g in goals if g.status == GoalStatus.BLOCKED)

        return {
            "total": total,
            "achieved": achieved,
            "active": active,
            "blocked": blocked,
            "overall_progress": round(sum(g.progress for g in goals) / total, 3),
        }

    def _assess_performance(self, rewards: List[float]) -> Dict[str, Any]:
        """Assess recent performance."""
        if not rewards:
            return {"status": "No performance data"}

        return {
            "total_reward": round(sum(rewards), 3),
            "avg_reward": round(sum(rewards) / len(rewards), 3),
            "recent_trend": "improving" if len(rewards) >= 3 and rewards[-1] > rewards[-3] else "declining",
            "steps": len(rewards),
        }

    def get_last_reflection(self) -> Optional[Dict[str, Any]]:
        return self._reflections[-1] if self._reflections else None


# ──────────────────────────────────────────────
#  Agentic Controller (Master Orchestrator)
# ──────────────────────────────────────────────

class AgenticController:
    """
    The master orchestrator that ties everything together.
    
    Workflow:
    1. User input → Goal decomposition
    2. RAG retrieval → Enrich context
    3. MDP state → RL action selection (with meta-learning)
    4. Execute action → Update state + memory
    5. Self-reflect → Adjust strategy
    6. Repeat until goals achieved or budget exhausted
    """

    def __init__(
        self,
        agent_id: str = "default",
        domain: str = "general",
        max_steps: int = 15,
        persist_memory: bool = True,
    ):
        self.agent_id = agent_id
        self.domain = domain
        self.max_steps = max_steps

        # Core modules
        self.mdp = MDPController()
        self.rl = RLDecisionEngine(state_dim=15)
        self.meta = MetaLearner(state_dim=15)
        self.rag = RAGEngine()
        self.memory = AgentMemory(
            agent_id=agent_id,
            persist_path=f"data/memory/{agent_id}_memory.json" if persist_memory else None,
        )

        # Orchestration
        self.goal_manager = GoalManager()
        self.stuck_detector = StuckDetector()
        self.reflector = SelfReflector()

        # Register task profile for meta-learning
        self.task_profile = TaskProfile(
            task_id=f"{agent_id}_{domain}",
            domain=domain,
        )
        self.meta.register_task(self.task_profile)

        # Action handlers
        self._action_handlers: Dict[AgentAction, Callable] = {}

        # Reasoning trace
        self._trace: List[Dict[str, Any]] = []

    def register_action_handler(self, action: AgentAction, handler: Callable):
        """
        Register a handler function for a specific action.
        Handler signature: (state: MDPState, context: Dict) -> (result: str, quality: float)
        """
        self._action_handlers[action] = handler

    def run(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full agentic loop.
        
        Args:
            user_input: User's query or instruction
            context: Additional context (e.g., uploaded files, settings)
            
        Returns:
            Full result with output, reasoning trace, and performance metrics
        """
        context = context or {}
        self._trace = []

        # Step 1: Store input in memory
        self.memory.add_message("user", user_input)

        # Step 2: Goal creation
        main_goal = self.goal_manager.create_goal(
            description=f"Respond to: {user_input[:100]}",
            priority=0.8,
        )
        main_goal.status = GoalStatus.ACTIVE

        # Step 3: Estimate task complexity
        complexity = self._estimate_complexity(user_input)
        state = self.mdp.reset(task_complexity=complexity)

        # Step 4: Meta-learning adaptation
        task_id = self.task_profile.task_id
        best_source = self.meta.find_best_source_task(task_id)
        if best_source:
            self.meta.transfer_knowledge(best_source, task_id, blend=0.2)

        # Step 5: Recall relevant episodic memories
        past_episodes = self.memory.recall_episodes(task_type=self.domain, n=3)
        if past_episodes:
            lessons = [ep.lesson for ep in past_episodes if ep.lesson]
            if lessons:
                self._trace.append({
                    "step": "memory_recall",
                    "lessons_from_past": lessons[:3],
                })

        # Step 6: RAG context enrichment
        rag_context = ""
        if self.rag.vector_store.size > 0:
            rag_context, rag_results = self.rag.build_context(user_input)
            if rag_context:
                state.context_richness = min(1.0, len(rag_results) * 0.2)
                self._trace.append({
                    "step": "rag_retrieval",
                    "chunks_retrieved": len(rag_results),
                    "context_richness": state.context_richness,
                })

        # Step 7: Main decision loop
        final_output = ""
        step_count = 0

        while step_count < self.max_steps:
            step_count += 1

            # Get available actions
            available = self.mdp.get_available_actions()
            if not available:
                break

            # RL selects action
            action, decision_info = self.rl.select_action(state, available)

            # Stuck detection
            stuck_info = self.stuck_detector.check(state, action)
            if stuck_info["is_stuck"]:
                self._trace.append({
                    "step": f"stuck_detected_{step_count}",
                    "reason": stuck_info["reason"],
                    "recovery": stuck_info["recovery_suggestion"],
                })
                # Force a different action — cycle through recovery options
                if action != AgentAction.GENERATE and state.generation_count == 0:
                    action = AgentAction.GENERATE
                elif action != AgentAction.REFLECT:
                    action = AgentAction.REFLECT
                else:
                    action = AgentAction.GENERATE

            # Execute action
            result_text, quality = self._execute_action(
                action, state, user_input, rag_context, context
            )

            # MDP step — completion check
            is_complete = (
                (quality > 0.5 and state.confidence > 0.3 and step_count >= 2 and state.generation_count >= 1)
                or (step_count >= 5 and state.generation_count >= 1)  # Budget fallback
            )
            state, reward, done = self.mdp.step(action, is_complete, quality)

            # Record experience for RL
            self.rl.record_experience(
                self.mdp.episode_states[-2],  # previous state
                action,
                reward,
                state,
                done,
            )

            # Trace
            self._trace.append({
                "step": step_count,
                "action": action.value,
                "reward": round(reward, 3),
                "confidence": round(state.confidence, 3),
                "phase": state.phase.value,
                "explored": decision_info["explored"],
                "result_preview": result_text[:100] if result_text else "",
            })

            if result_text:
                # Prefer outputs from high-value actions
                high_value = (AgentAction.GENERATE, AgentAction.SYNTHESIZE, AgentAction.DELEGATE, AgentAction.REFINE)
                if action in high_value:
                    final_output = result_text
                elif not final_output:
                    final_output = result_text

            if done:
                break

            # Periodic reflection
            if step_count % 3 == 0:
                reflection = self.reflector.reflect(
                    state,
                    self.goal_manager.get_active_goals(),
                    self.mdp.episode_rewards,
                    stuck_info,
                )
                self._trace.append({
                    "step": f"reflection_{step_count}",
                    "insights": reflection["insights"],
                    "adjustments": [a["suggestion"] for a in reflection["strategy_adjustments"]],
                })

        # Step 8: Finalize
        self.goal_manager.update_goal(main_goal.goal_id, progress=1.0, status=GoalStatus.ACHIEVED)

        # Store episode in memory
        episode_summary = self.mdp.get_episode_summary()
        self.memory.store_episode(
            task_type=self.domain,
            actions=episode_summary["actions"],
            total_reward=episode_summary["total_reward"],
            success=episode_summary["final_confidence"] > 0.5,
            summary=f"Handled: {user_input[:100]}",
            lesson=self._extract_lesson(episode_summary),
        )

        # Store response in memory
        self.memory.add_message("assistant", final_output[:500] if final_output else "No output generated")

        # Meta-learning update
        self.meta.update_task_stats(
            task_id,
            episode_summary["total_reward"],
            episode_summary["final_confidence"] > 0.5,
        )

        return {
            "output": final_output,
            "reasoning_trace": self._trace,
            "performance": episode_summary,
            "goals": self.goal_manager.get_goal_tree(),
            "memory_stats": self.memory.get_stats(),
            "rl_stats": self.rl.get_stats(),
            "meta_stats": self.meta.get_adaptation_summary(),
        }

    def _execute_action(
        self,
        action: AgentAction,
        state: MDPState,
        user_input: str,
        rag_context: str,
        context: Dict[str, Any],
    ) -> Tuple[str, float]:
        """
        Execute a selected action.
        Returns (result_text, quality_score).
        """
        # Use registered handler if available
        if action in self._action_handlers:
            try:
                handler_context = {
                    "user_input": user_input,
                    "rag_context": rag_context,
                    "conversation_history": self.memory.get_conversation_context(),
                    "extra": context,
                }
                result, quality = self._action_handlers[action](state, handler_context)
                return result, quality
            except Exception as e:
                state.error_count += 1
                return f"Error executing {action.value}: {str(e)}", 0.0

        # Default handlers
        if action == AgentAction.RETRIEVE:
            if self.rag.vector_store.size > 0:
                context_str, results = self.rag.build_context(user_input)
                return context_str, min(1.0, len(results) * 0.2)
            return "No knowledge base available for retrieval.", 0.1

        elif action == AgentAction.GENERATE:
            # In a full system, this would call the LLM
            prompt = self._build_prompt(user_input, rag_context, state)
            return prompt, 0.75

        elif action == AgentAction.REFLECT:
            reflection = self.reflector.reflect(
                state,
                self.goal_manager.get_active_goals(),
                self.mdp.episode_rewards,
                {"is_stuck": False},
            )
            insights = reflection.get("insights", [])
            return f"Reflection: {'; '.join(insights) if insights else 'No new insights'}", 0.3

        elif action == AgentAction.DELEGATE:
            return "Delegating to specialist sub-agent...", 0.65

        elif action == AgentAction.REFINE:
            return "Refining previous output with additional context...", 0.5

        elif action == AgentAction.EXPLORE:
            return "Exploring alternative approaches...", 0.3

        elif action == AgentAction.SYNTHESIZE:
            return "Synthesizing information from multiple sources...", 0.7

        elif action == AgentAction.ESCALATE:
            return "Need clarification from user.", 0.2

        return "", 0.0

    def _build_prompt(
        self,
        user_input: str,
        rag_context: str,
        state: MDPState,
    ) -> str:
        """Build an LLM-ready prompt with RAG context and conversation history."""
        parts = []

        # System context
        parts.append(f"You are an AI agent in the '{self.domain}' domain.")
        parts.append(f"Current confidence: {state.confidence:.2f}")
        parts.append(f"Phase: {state.phase.value}")

        # RAG context
        if rag_context:
            parts.append(f"\n--- Retrieved Context ---\n{rag_context}\n--- End Context ---")

        # Conversation history
        history = self.memory.get_conversation_context()
        if history:
            parts.append(f"\n--- Conversation History ---\n{history}\n--- End History ---")

        # Relevant past lessons
        episodes = self.memory.recall_episodes(task_type=self.domain, success_only=True, n=2)
        if episodes:
            lessons = [ep.lesson for ep in episodes if ep.lesson]
            if lessons:
                parts.append(f"\n--- Past Lessons ---\n" + "\n".join(f"- {l}" for l in lessons))

        # User query
        parts.append(f"\nUser Query: {user_input}")
        parts.append("\nProvide a comprehensive, well-structured response:")

        return "\n".join(parts)

    def _estimate_complexity(self, text: str) -> float:
        """Estimate task complexity from user input."""
        complexity = 0.3

        # Length-based
        words = len(text.split())
        if words > 100:
            complexity += 0.2
        elif words > 50:
            complexity += 0.1

        # Keyword-based complexity indicators
        complex_keywords = [
            "analyze", "compare", "evaluate", "explain", "summarize",
            "research", "investigate", "synthesize", "design", "implement",
        ]
        for kw in complex_keywords:
            if kw in text.lower():
                complexity += 0.05

        return min(1.0, complexity)

    def _extract_lesson(self, episode_summary: Dict[str, Any]) -> str:
        """Extract a lesson from the episode for future reference."""
        actions = episode_summary.get("actions", [])
        reward = episode_summary.get("total_reward", 0)
        confidence = episode_summary.get("final_confidence", 0)

        if reward > 5:
            action_seq = " → ".join(actions[:5])
            return f"Successful strategy: {action_seq} (reward={reward:.1f}, confidence={confidence:.2f})"
        elif reward < 0:
            return f"Underperforming sequence with {len(actions)} steps — consider shorter paths"
        return f"Moderate outcome ({len(actions)} steps, reward={reward:.1f})"

    def ingest_knowledge(self, text: str, source: str = "", doc_id: str = ""):
        """Add knowledge to the RAG engine."""
        self.rag.ingest_text(text, doc_id=doc_id, source=source)

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete agent state for dashboard visualization."""
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "mdp_state": self.mdp.state.to_dict(),
            "goals": self.goal_manager.get_goal_tree(),
            "memory": self.memory.get_stats(),
            "rl": self.rl.get_stats(),
            "meta": self.meta.get_adaptation_summary(),
            "rag": self.rag.get_stats(),
            "last_reflection": self.reflector.get_last_reflection(),
        }
