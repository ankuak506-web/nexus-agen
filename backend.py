"""
NexusAI Backend — Full Agentic Pipeline Server
================================================
FastAPI server that connects:
  - AgenticController (Goal Manager + Planner + Reflector)
  - RL Decision Engine (Policy-gradient action selection)
  - MDP Engine (State-Action-Transition framework)
  - Meta-Learner (MAML-style rapid adaptation)
  - RAG Engine (Vector + BM25 hybrid retrieval)
  - Agent Memory (Episodic + Semantic + Working)
  - Groq LLM (Llama 3.3 70B for generation)
"""

import os
import sys
import json
import time
import hashlib
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Core agentic imports
from core.agentic_controller import AgenticController, GoalStatus
from core.mdp_engine import AgentAction, MDPState
from core.rag_engine import RAGEngine, Document
from core.meta_learner import TaskProfile

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ──────────────────────────────────────────────
#  FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="NexusAI — Agentic Intelligence Platform",
    version="2.0.0",
    description="Enterprise AI Operations Platform with RL + MDP + Meta-Learning + RAG",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
#  Agent Pool (per-domain agents)
# ──────────────────────────────────────────────

agents: Dict[str, AgenticController] = {}


def get_or_create_agent(domain: str = "general") -> AgenticController:
    """Get or create an agent for a given domain."""
    if domain not in agents:
        os.makedirs("data/memory", exist_ok=True)
        agent = AgenticController(
            agent_id=f"nexus_{domain}",
            domain=domain,
            max_steps=10,
            persist_memory=True,
        )

        # Register the LLM-powered GENERATE handler
        agent.register_action_handler(AgentAction.GENERATE, llm_generate_handler)
        agent.register_action_handler(AgentAction.SYNTHESIZE, llm_synthesize_handler)
        agent.register_action_handler(AgentAction.REFINE, llm_refine_handler)

        agents[domain] = agent
    return agents[domain]


# ──────────────────────────────────────────────
#  LLM-Powered Action Handlers
# ──────────────────────────────────────────────

async def _call_groq(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Call Groq API with retry."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(3):
            try:
                response = await client.post(
                    GROQ_API_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                    },
                    json={
                        "model": GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": 4096,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return f"LLM API error ({response.status_code}): {response.text[:200]}"
            except Exception as e:
                if attempt == 2:
                    return f"LLM connection failed: {str(e)}"
                await asyncio.sleep(1)
    return "LLM unavailable after retries."


def _call_groq_sync(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Synchronous wrapper for Groq calls (used in action handlers)."""
    import httpx as httpx_sync
    try:
        with httpx_sync.Client(timeout=60.0) as client:
            response = client.post(
                GROQ_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": 4096,
                },
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            return f"LLM error: {response.status_code}"
    except Exception as e:
        return f"LLM call failed: {str(e)}"


def llm_generate_handler(state: MDPState, context: Dict[str, Any]):
    """LLM-powered GENERATE action — the core intelligence."""
    user_input = context.get("user_input", "")
    rag_context = context.get("rag_context", "")
    conversation = context.get("conversation_history", "")
    extra = context.get("extra", {})
    file_context = extra.get("file_context", "")

    system_prompt = """You are NexusAI, an enterprise AI operations analyst. 
You provide precise, data-driven, actionable insights.
- Reference actual data when available
- Use **bold** for emphasis and • for bullet points
- Keep language professional and executive-ready
- Never mention internal workings (RL, MDP, policies, rewards)"""

    parts = [f"USER GOAL: {user_input}"]

    if file_context:
        parts.append(f"\nUPLOADED DATA:\n{file_context}")
    if rag_context:
        parts.append(f"\nRETRIEVED CONTEXT (from knowledge base):\n{rag_context}")
    if conversation:
        parts.append(f"\nCONVERSATION HISTORY:\n{conversation}")

    parts.append("\nProvide a comprehensive, structured analysis:")

    result = _call_groq_sync(system_prompt, "\n".join(parts))
    quality = 0.8 if len(result) > 100 else 0.5
    return result, quality


def llm_synthesize_handler(state: MDPState, context: Dict[str, Any]):
    """LLM-powered SYNTHESIZE — combine multiple sources."""
    user_input = context.get("user_input", "")
    rag_context = context.get("rag_context", "")
    extra = context.get("extra", {})
    file_context = extra.get("file_context", "")

    system_prompt = "Synthesize information from multiple sources into a unified, executive-ready analysis."
    parts = [f"OBJECTIVE: {user_input}"]
    if file_context:
        parts.append(f"\nDATA SOURCE:\n{file_context}")
    if rag_context:
        parts.append(f"\nKNOWLEDGE BASE:\n{rag_context}")
    parts.append("\nSynthesize all available information:")

    result = _call_groq_sync(system_prompt, "\n".join(parts))
    return result, 0.75


def llm_refine_handler(state: MDPState, context: Dict[str, Any]):
    """LLM-powered REFINE — improve previous output."""
    user_input = context.get("user_input", "")
    conversation = context.get("conversation_history", "")

    system_prompt = "Refine and improve the previous analysis. Add depth, fix gaps, and enhance clarity."
    prompt = f"ORIGINAL GOAL: {user_input}\n\nPREVIOUS CONTEXT:\n{conversation}\n\nRefine and improve:"

    result = _call_groq_sync(system_prompt, prompt, temperature=0.2)
    return result, 0.7


# ──────────────────────────────────────────────
#  Request / Response Models
# ──────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    goal: str
    domain: str = "general"
    priority: str = "normal"
    file_context: str = ""  # Pre-parsed file data from frontend


class ExecuteResponse(BaseModel):
    task_id: str
    goal: str
    domain: str
    status: str
    summary: str
    sections: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    reasoning_trace: List[Dict[str, Any]]
    agent_state: Dict[str, Any]


# ──────────────────────────────────────────────
#  API Routes
# ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "NexusAI — Agentic Intelligence Platform",
        "version": "2.0.0",
        "engine": "RL + MDP + Meta-Learning + RAG + Groq LLM",
        "agents_active": len(agents),
    }


@app.post("/execute", response_model=ExecuteResponse)
async def execute_goal(request: ExecuteRequest):
    """
    Execute a goal through the full agentic pipeline:  
    1. Goal decomposition → 2. RAG retrieval → 3. RL action selection →  
    4. MDP state transitions → 5. LLM generation → 6. Self-reflection →  
    7. Meta-learning update → 8. Memory storage
    """
    agent = get_or_create_agent(request.domain)

    # Pass file context as extra context
    context = {
        "file_context": request.file_context,
        "priority": request.priority,
    }

    # Run the full agentic loop (sync — runs RL/MDP/Meta-learning)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: agent.run(request.goal, context=context),
    )

    # Parse the LLM output into structured sections
    output_text = result.get("output", "")
    sections = _parse_output_to_sections(output_text)

    # Build metrics from the agentic pipeline
    performance = result.get("performance", {})
    episode_summary = performance

    task_id = f"task_{int(time.time() * 1000)}"

    return ExecuteResponse(
        task_id=task_id,
        goal=request.goal,
        domain=request.domain,
        status="completed",
        summary=output_text[:300] if output_text else "Analysis complete.",
        sections=sections,
        metrics={
            "confidence": performance.get("final_confidence", 0.0),
            "duration": performance.get("total_duration", 0.0),
            "total_reward": performance.get("total_reward", 0.0),
            "steps": performance.get("episode_length", 0),
            "actions_taken": performance.get("actions", []),
            "impact_score": min(1.0, performance.get("total_reward", 0) / 10.0),
            "domain_context": request.domain,
        },
        reasoning_trace=result.get("reasoning_trace", []),
        agent_state={
            "goals": result.get("goals", []),
            "memory": result.get("memory_stats", {}),
            "rl": result.get("rl_stats", {}),
            "meta": result.get("meta_stats", {}),
        },
    )


@app.post("/ingest")
async def ingest_knowledge(
    text: str = Form(...),
    source: str = Form(""),
    domain: str = Form("general"),
):
    """Ingest text into the RAG knowledge base."""
    agent = get_or_create_agent(domain)
    chunks = agent.ingest_knowledge(text, source=source)
    return {
        "status": "ingested",
        "domain": domain,
        "rag_stats": agent.rag.get_stats(),
    }


@app.get("/agent/{domain}/state")
def get_agent_state(domain: str):
    """Get the full state of a domain agent."""
    if domain not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{domain}' not found")
    return agents[domain].get_full_state()


@app.get("/agents")
def list_agents():
    """List all active agents."""
    return {
        domain: {
            "agent_id": agent.agent_id,
            "domain": agent.domain,
            "memory": agent.memory.get_stats(),
            "rag": agent.rag.get_stats(),
        }
        for domain, agent in agents.items()
    }


@app.get("/health")
def health():
    return {
        "status": "operational",
        "agents": len(agents),
        "llm": "groq/llama-3.3-70b-versatile",
        "engine": "RL + MDP + Meta-Learning + RAG",
    }


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _parse_output_to_sections(output: str) -> List[Dict[str, Any]]:
    """Parse LLM output into structured sections."""
    if not output:
        return [{"title": "Analysis", "content": "No output generated.", "type": "text"}]

    # Try to detect section headers (## or **Title**: pattern)
    sections = []
    current_title = "Analysis"
    current_content = []

    for line in output.split("\n"):
        stripped = line.strip()

        # Detect markdown headers
        if stripped.startswith("## "):
            if current_content:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_content).strip(),
                    "type": _infer_section_type(current_title),
                })
            current_title = stripped[3:].strip()
            current_content = []
        elif stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 6:
            if current_content:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_content).strip(),
                    "type": _infer_section_type(current_title),
                })
            current_title = stripped.strip("*").strip()
            current_content = []
        else:
            current_content.append(line)

    # Last section
    if current_content:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_content).strip(),
            "type": _infer_section_type(current_title),
        })

    return sections if sections else [{"title": "Analysis", "content": output, "type": "text"}]


def _infer_section_type(title: str) -> str:
    """Infer section display type from its title."""
    title_lower = title.lower()
    if any(w in title_lower for w in ["answer", "result", "direct", "finding"]):
        return "recommendation"
    elif any(w in title_lower for w in ["insight", "key", "data", "analysis"]):
        return "insight"
    elif any(w in title_lower for w in ["recommend", "action", "next", "strategy"]):
        return "recommendation"
    elif any(w in title_lower for w in ["risk", "caveat", "warning", "limitation"]):
        return "risk"
    return "text"


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 NexusAI Agentic Backend")
    print("   Engine: RL + MDP + Meta-Learning + RAG + Groq LLM")
    print("   Server: http://localhost:8000")
    print("   Docs:   http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
