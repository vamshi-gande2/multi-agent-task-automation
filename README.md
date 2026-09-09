# Multi-Agent Task Automation (LangGraph)

A compact, interview-ready Agentic AI MVP designed for the FlytBase Agentic AI Engineer application.

## What it demonstrates

- Supervisor agent
- Specialist research agent
- Specialist calculation agent
- Shared workflow state
- Conditional LangGraph routing
- Specialist execution
- Finalizer agent
- Deterministic offline mock mode
- Loop/step tracking
- Clear separation of agent responsibilities

## Architecture

```text
                       USER TASK
                           |
                           v
                    +--------------+
                    |  SUPERVISOR  |
                    | route task   |
                    +------+-------+
                           |
                +----------+----------+
                |                     |
          RESEARCH              CALCULATE
                |                     |
                v                     v
       +----------------+     +----------------+
       | Research Agent |     | Calculator     |
       | KB retrieval   |     | Specialist     |
       +--------+-------+     +--------+-------+
                |                      |
                +----------+-----------+
                           v
                    +--------------+
                    |  FINALIZER   |
                    | final answer |
                    +--------------+
```

## Why this counts as multi-agent

The workflow has multiple specialized agent roles with distinct responsibilities: a supervisor chooses the route, a research specialist handles knowledge retrieval, a calculation specialist handles arithmetic, and a finalizer converts the specialist output into the user-facing response.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Create `.env`:

```env
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_MODEL=claude-sonnet-4-5
```

Then:

```bash
python agent.py "What is LangGraph?"
python agent.py "Calculate 240 / 40"
```

Without an API key or installed LangGraph dependencies, the project runs the same routing design in deterministic mock mode.

## Resume-ready description

> Built a multi-agent task automation system using LangGraph with a supervisor agent that routes tasks to specialized research and calculation agents, followed by a finalization step. Implemented shared state, conditional routing, and tool-backed task execution to demonstrate action-oriented agent workflows rather than single prompt-response generation.

## What to say in an interview

> "I wanted to understand agentic systems beyond calling an LLM once. I separated the workflow into specialized roles. The supervisor first decides which capability is needed, then the relevant specialist performs the task, and a finalizer produces the response. The graph has explicit state and conditional routing, so the execution path depends on the task instead of being a fixed chain."

## Next upgrades after the FlytBase application

1. Replace the local knowledge base with web search or a vector database.
2. Add RAG with Chroma and embeddings.
3. Add an external API tool.
4. Add human approval before side-effecting actions.
5. Add persistence and evaluation traces.
