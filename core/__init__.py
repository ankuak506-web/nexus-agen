"""
Core Agentic AI Framework
==========================
Unified RL + MDP + Meta-Learning + RAG engine for all agentic projects.

Modules:
  - mdp_engine: Markov Decision Process state-action-transition framework
  - rl_decision_engine: Reinforcement Learning policy-based decision engine
  - meta_learner: MAML-style meta-learning for rapid task adaptation
  - rag_engine: Retrieval-Augmented Generation pipeline
  - memory: Persistent episodic + semantic agent memory
  - agentic_controller: Goal manager, planner, and self-reflector
"""

__version__ = "2.0.0"
__all__ = [
    "mdp_engine",
    "rl_decision_engine",
    "meta_learner",
    "rag_engine",
    "memory",
    "agentic_controller",
]
