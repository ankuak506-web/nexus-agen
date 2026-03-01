# 🧠 Agentic AI Projects — RL + MDP + Meta-Learning + RAG

> **A unified framework for building truly agentic AI systems powered by Reinforcement Learning, Markov Decision Processes, Meta-Learning, and Retrieval-Augmented Generation.**

---

## 🏗️ Architecture

```
agentic-ai-projects/
│
├── core/                              ← 🧠 Shared Intelligence Engine
│   ├── mdp_engine.py                  ← MDP: States, Actions, Transitions, Rewards
│   ├── rl_decision_engine.py          ← RL: Policy Gradient + Q-Learning + ε-Greedy
│   ├── meta_learner.py                ← MAML-style Few-Shot Adaptation
│   ├── rag_engine.py                  ← RAG: Chunking + Embeddings + Hybrid Retrieval
│   ├── memory.py                      ← Episodic + Semantic + Working Memory
│   └── agentic_controller.py          ← Goal Manager + Planner + Reflector
│
├── 01-Financial-Agent-With-Phidata/   ← 🏦 RL-Driven Financial Analysis
├── 02-Multi-Agentic-AI-RAG/           ← 📄 Meta-Learning Document RAG
├── 03-Video-Summarizer/               ← 🎥 Agentic Video Analysis
├── 04-Agentic-RAG-with-Langgraph/     ← 🧠 Full MDP-Planned RAG Pipeline
├── 05-SQL-Database-Agents/            ← 🗄️ RL-Guided SQL Exploration
├── 06-Content-Generation/             ← ✍️ Multi-Agent Content Crew
│
├── requirements.txt                   ← Unified dependencies
├── .env.example                       ← API key template
└── README.md
```

---

## 🔬 Core Framework

### 1. 📊 MDP Engine (`core/mdp_engine.py`)

Models every agent interaction as a **Markov Decision Process**:

| Component | Description |
|-----------|-------------|
| **States** | Phase (Init→Understanding→Planning→Executing→Reviewing→Complete), confidence, context richness, task complexity |
| **Actions** | Retrieve, Generate, Delegate, Reflect, Escalate, Refine, Explore, Synthesize |
| **Transitions** | Probabilistic state updates with learned transition probabilities |
| **Rewards** | Task completion, confidence gain, context enrichment, efficiency bonuses |

### 2. 🎯 RL Decision Engine (`core/rl_decision_engine.py`)

**Policy gradient** action selection with:
- 2-layer softmax neural policy (NumPy-only, no PyTorch required)
- Q-value estimation with TD(0) updates
- Experience replay buffer (5000 capacity)
- ε-greedy exploration with annealing (0.15 → 0.02)
- Combined policy + Q-value scoring for action selection

### 3. 🧬 Meta-Learner (`core/meta_learner.py`)

**MAML-style** adaptation enabling:
- Task-specific parameter snapshots
- Inner-loop gradient adaptation (5 steps)
- Cross-project knowledge transfer (cosine similarity matching)
- Learning rate meta-optimization per task
- Support set adaptation for few-shot learning

### 4. 📚 RAG Engine (`core/rag_engine.py`)

Full **Retrieval-Augmented Generation** pipeline:
- Intelligent text chunking with overlap
- TF-IDF embeddings (zero-dependency) or pluggable external embeddings
- NumPy vector store (cosine similarity search)
- BM25 keyword-based fallback retrieval
- Hybrid retrieval with automatic re-ranking
- Context window management

### 5. 🧠 Memory (`core/memory.py`)

Multi-tier **persistent memory** system:
- **Short-term**: Conversation context (deque)
- **Episodic**: Past interaction episodes with rewards and lessons
- **Semantic**: Extracted facts, rules, and patterns
- **Working**: Active reasoning scratchpad
- JSON persistence for cross-session continuity

### 6. 🎮 Agentic Controller (`core/agentic_controller.py`)

The **master orchestrator** combining all modules:
- Goal decomposition and hierarchical tracking
- MDP-guided planning with RL action selection
- RAG context enrichment at every step
- Stuck detection and automatic recovery
- Self-reflection with strategy adjustment
- Full reasoning trace for transparency

---

## 🚀 Projects

### 🏦 01 — Financial Agent
**RL-driven portfolio analysis** with MDP-guided research workflow, RAG for financial news retrieval, and meta-learning across market domains (stocks, crypto, forex).

```bash
cd 01-Financial-Agent-With-Phidata
python financial_agent.py        # CLI mode
streamlit run financial_agent.py # Dashboard (call create_financial_dashboard)
```

### 📄 02 — Document RAG Agent
**Hybrid retrieval** (vector + BM25) with meta-learning domain adaptation, RL-optimized retrieval depth, and multi-document cross-referencing.

```bash
cd 02-Multi-Agentic-AI-RAG-With-Vector-Database
python pdf_assistant.py
```

### 🎥 03 — Video Analyzer
**Agentic video analysis** with MDP-guided workflow, transcript RAG indexing, RL strategy selection (visual/audio/comprehensive), and genre-aware meta-learning.

```bash
cd 03-End-To-End-Video-Summarizer-Agentic-AI-With-Phidata-And-Google-Gemini
python app.py
```

### 🧠 04 — Full Agentic RAG
**The most advanced RAG project** — MDP plans retrieval depth, RL optimizes query reformulation, multi-hop reasoning chains, and self-reflective quality assessment.

```bash
cd "04-Agentic RAG with Langgraphh"
python app.py
```

### 🗄️ 05 — SQL Agent
**RL-guided schema exploration** with MDP query planning lifecycle, RAG for SQL pattern retrieval, and meta-learning for database-type adaptation.

```bash
cd "05-Sql-Database-Agents with Langgraph"
python sql_agent.py
```

### ✍️ 06 — Content Generator
**Multi-agent content crew** (Researcher + Writer + Editor) with RL-guided delegation, MDP workflow, meta-learning style adaptation, and RAG reference retrieval.

```bash
cd "06-Content-Generation with Crew AI"
python app.py
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/agentic-ai-projects.git
cd agentic-ai-projects

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Run any project
python 01-Financial-Agent-With-Phidata/financial_agent.py
```

### Prerequisites
- Python 3.10+
- NumPy (core framework, no PyTorch needed)
- Streamlit (for dashboards)
- API keys optional (framework works standalone for demo)

---

## 🧪 How It Works

Every agent follows the same intelligent loop:

```
User Input
    ↓
┌───────────────────────────────────┐
│  1. Goal Decomposition            │
│  2. RAG Context Enrichment        │
│  3. MDP State Assessment          │
│  4. RL Action Selection           │  ← Policy + Q-value + ε-greedy
│  5. Action Execution              │
│  6. Experience Recording          │  ← Replay buffer + TD update
│  7. Stuck Detection               │
│  8. Self-Reflection (every 3 steps)│
│  9. Meta-Learning Update          │  ← MAML adaptation
│  10. Memory Storage               │  ← Episodic + Semantic
└───────────────────────────────────┘
    ↓
Output + Reasoning Trace + Performance Metrics
```

---

## 📊 Key Metrics Exposed

Every agent response includes:
- **Confidence**: Agent's self-assessed confidence (0–1)
- **Reasoning Trace**: Step-by-step decision log
- **Strategy**: Sequence of actions taken
- **RL Stats**: Exploration rate, total decisions, Q-values
- **Meta-Learning**: Task adaptation history
- **Memory**: Episodic count, semantic facts, success patterns

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

**Built with ❤️ — Transforming reactive agents into truly intelligent, adaptive AI systems.**
