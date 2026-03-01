"""
Agent Memory — Persistent Episodic + Semantic Memory
======================================================
Multi-tier memory system for agents:
  - Short-term: Current conversation context
  - Episodic: Past interaction episodes with rewards
  - Semantic: Extracted facts and patterns
  - Working: Active reasoning scratchpad

Persists to JSON for cross-session continuity.
"""

import json
import os
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque


# ──────────────────────────────────────────────
#  Memory Entry Types
# ──────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """Base memory entry."""
    entry_id: str
    content: str
    timestamp: float
    memory_type: str    # "episodic", "semantic", "working"
    importance: float = 0.5   # 0–1, how important this memory is
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryEntry":
        return cls(**d)


@dataclass
class EpisodicMemory(MemoryEntry):
    """
    Episode-level memory: stores a complete interaction sequence.
    """
    episode_id: str = ""
    task_type: str = ""
    actions_taken: List[str] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    lesson: str = ""  # What was learned

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "episode_id": self.episode_id,
            "task_type": self.task_type,
            "actions_taken": self.actions_taken,
            "total_reward": self.total_reward,
            "success": self.success,
            "lesson": self.lesson,
        })
        return d


@dataclass
class SemanticMemory(MemoryEntry):
    """
    Semantic memory: stores extracted facts and patterns.
    """
    fact_type: str = ""        # "rule", "preference", "fact", "pattern"
    confidence: float = 0.5
    source_episodes: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "fact_type": self.fact_type,
            "confidence": self.confidence,
            "source_episodes": self.source_episodes,
            "contradicts": self.contradicts,
        })
        return d


# ──────────────────────────────────────────────
#  Memory Store
# ──────────────────────────────────────────────

class AgentMemory:
    """
    Multi-tier memory system with persistence.
    
    Tiers:
      1. Short-term: deque of recent messages (volatile)
      2. Episodic: past episodes with outcomes
      3. Semantic: extracted facts and learned rules
      4. Working: active scratchpad for current reasoning
    """

    def __init__(
        self,
        agent_id: str = "default",
        short_term_capacity: int = 20,
        max_episodic: int = 500,
        max_semantic: int = 200,
        persist_path: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.short_term_capacity = short_term_capacity
        self.max_episodic = max_episodic
        self.max_semantic = max_semantic

        # Memory tiers
        self.short_term: deque = deque(maxlen=short_term_capacity)
        self.episodic: List[EpisodicMemory] = []
        self.semantic: List[SemanticMemory] = []
        self.working: Dict[str, Any] = {}  # Active scratchpad

        # Persistence
        self.persist_path = persist_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "memory", f"{agent_id}_memory.json"
        )

        # Load existing memory
        self._load()

    # ── Short-Term ──

    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to short-term memory."""
        self.short_term.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })

    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the last n messages."""
        return list(self.short_term)[-n:]

    def get_conversation_context(self) -> str:
        """Build a context string from recent messages."""
        messages = self.get_recent_messages()
        parts = []
        for msg in messages:
            role = msg["role"].upper()
            parts.append(f"[{role}]: {msg['content']}")
        return "\n".join(parts)

    # ── Episodic ──

    def store_episode(
        self,
        task_type: str,
        actions: List[str],
        total_reward: float,
        success: bool,
        summary: str,
        lesson: str = "",
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Store a completed episode."""
        episode_id = hashlib.md5(f"{time.time()}:{summary[:50]}".encode()).hexdigest()[:12]

        ep = EpisodicMemory(
            entry_id=episode_id,
            content=summary,
            timestamp=time.time(),
            memory_type="episodic",
            importance=min(1.0, abs(total_reward) / 10.0),
            episode_id=episode_id,
            task_type=task_type,
            actions_taken=actions,
            total_reward=total_reward,
            success=success,
            lesson=lesson,
            metadata=metadata or {},
            tags=[task_type, "success" if success else "failure"],
        )

        self.episodic.append(ep)

        # Trim to max (remove least important + oldest)
        if len(self.episodic) > self.max_episodic:
            self.episodic.sort(key=lambda e: e.importance * 0.7 + (1 - (time.time() - e.timestamp) / 86400) * 0.3, reverse=True)
            self.episodic = self.episodic[:self.max_episodic]

        self._save()
        return episode_id

    def recall_episodes(
        self,
        task_type: Optional[str] = None,
        success_only: bool = False,
        n: int = 5,
    ) -> List[EpisodicMemory]:
        """Recall relevant past episodes."""
        filtered = self.episodic

        if task_type:
            filtered = [e for e in filtered if e.task_type == task_type]

        if success_only:
            filtered = [e for e in filtered if e.success]

        # Sort by recency * importance
        filtered.sort(
            key=lambda e: e.importance * 0.5 + (1.0 / (1.0 + (time.time() - e.timestamp) / 3600)) * 0.5,
            reverse=True,
        )

        result = filtered[:n]

        # Update access counts
        for ep in result:
            ep.access_count += 1
            ep.last_accessed = time.time()

        return result

    def get_success_patterns(self, task_type: str) -> Dict[str, Any]:
        """Analyze successful episodes to find patterns."""
        successes = [e for e in self.episodic if e.task_type == task_type and e.success]
        failures = [e for e in self.episodic if e.task_type == task_type and not e.success]

        if not successes:
            return {"patterns": [], "success_rate": 0.0}

        # Find common action sequences in successes
        from collections import Counter
        action_freqs = Counter()
        for ep in successes:
            for action in ep.actions_taken:
                action_freqs[action] += 1

        total = len(successes) + len(failures)
        return {
            "patterns": action_freqs.most_common(5),
            "success_rate": len(successes) / max(total, 1),
            "avg_reward": sum(e.total_reward for e in successes) / max(len(successes), 1),
            "total_episodes": total,
        }

    # ── Semantic ──

    def store_fact(
        self,
        content: str,
        fact_type: str = "fact",
        confidence: float = 0.7,
        source_episodes: List[str] = None,
        tags: List[str] = None,
    ) -> str:
        """Store a semantic fact or pattern."""
        entry_id = hashlib.md5(f"{content[:50]}:{time.time()}".encode()).hexdigest()[:12]

        fact = SemanticMemory(
            entry_id=entry_id,
            content=content,
            timestamp=time.time(),
            memory_type="semantic",
            importance=confidence,
            fact_type=fact_type,
            confidence=confidence,
            source_episodes=source_episodes or [],
            tags=tags or [fact_type],
        )

        # Check for contradictions
        for existing in self.semantic:
            if self._is_contradictory(existing.content, content):
                if confidence > existing.confidence:
                    existing.contradicts.append(entry_id)
                    fact.contradicts.append(existing.entry_id)
                else:
                    fact.contradicts.append(existing.entry_id)

        self.semantic.append(fact)

        # Trim
        if len(self.semantic) > self.max_semantic:
            self.semantic.sort(key=lambda f: f.confidence, reverse=True)
            self.semantic = self.semantic[:self.max_semantic]

        self._save()
        return entry_id

    def recall_facts(
        self,
        query: Optional[str] = None,
        fact_type: Optional[str] = None,
        n: int = 10,
    ) -> List[SemanticMemory]:
        """Recall relevant semantic memories."""
        filtered = self.semantic

        if fact_type:
            filtered = [f for f in filtered if f.fact_type == fact_type]

        if query:
            # Simple keyword matching
            query_words = set(query.lower().split())
            scored = []
            for f in filtered:
                content_words = set(f.content.lower().split())
                overlap = len(query_words & content_words) / max(len(query_words), 1)
                scored.append((f, overlap))
            scored.sort(key=lambda x: x[1], reverse=True)
            filtered = [f for f, s in scored if s > 0]

        result = filtered[:n]
        for f in result:
            f.access_count += 1
            f.last_accessed = time.time()
        return result

    # ── Working Memory ──

    def set_working(self, key: str, value: Any):
        """Set a working memory value."""
        self.working[key] = {
            "value": value,
            "timestamp": time.time(),
        }

    def get_working(self, key: str, default: Any = None) -> Any:
        """Get a working memory value."""
        entry = self.working.get(key)
        return entry["value"] if entry else default

    def clear_working(self):
        """Clear working memory."""
        self.working = {}

    # ── Persistence ──

    def _save(self):
        """Save memory to disk."""
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        data = {
            "agent_id": self.agent_id,
            "episodic": [e.to_dict() for e in self.episodic],
            "semantic": [f.to_dict() for f in self.semantic],
            "saved_at": time.time(),
        }
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        """Load memory from disk."""
        if not os.path.exists(self.persist_path):
            return

        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)

            for e_data in data.get("episodic", []):
                # Handle extra fields gracefully
                valid_fields = {f.name for f in EpisodicMemory.__dataclass_fields__.values()}
                clean = {k: v for k, v in e_data.items() if k in valid_fields}
                self.episodic.append(EpisodicMemory(**clean))

            for f_data in data.get("semantic", []):
                valid_fields = {f.name for f in SemanticMemory.__dataclass_fields__.values()}
                clean = {k: v for k, v in f_data.items() if k in valid_fields}
                self.semantic.append(SemanticMemory(**clean))

        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Start fresh if corrupted

    def _is_contradictory(self, text_a: str, text_b: str) -> bool:
        """Simple heuristic for contradiction detection."""
        negation_words = {"not", "never", "no", "don't", "doesn't", "isn't", "aren't", "won't", "can't"}
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        shared = words_a & words_b
        a_neg = bool(words_a & negation_words)
        b_neg = bool(words_b & negation_words)

        return len(shared) > 3 and a_neg != b_neg

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "short_term_messages": len(self.short_term),
            "episodic_memories": len(self.episodic),
            "semantic_facts": len(self.semantic),
            "working_memory_keys": list(self.working.keys()),
            "total_episodes_successful": sum(1 for e in self.episodic if e.success),
        }
