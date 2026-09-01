# AgentForge

**AI Agent Reliability & Evaluation Platform**

> A production-oriented platform for testing, evaluating, debugging, and monitoring tool-using AI agents across RAG, MCP, API, and code-execution workflows.

---

## What is AgentForge?

AgentForge is a developer workbench that answers a single question:

> **"Can I trust this AI agent to perform this task reliably?"**

Traditional LLM evaluation checks whether the final answer *looks* correct. AgentForge evaluates **what actually happened**: which tools were called, what arguments were passed, what database state resulted, whether policies were followed, and whether the system remained consistent.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Agent Runtime** | Multi-step LLM orchestration with tool calling, retry logic, and conversation state |
| **MCP Tool Layer** | Realistic tool implementations (customer, order, document) exposed via MCP |
| **RAG / Knowledge** | Policy retrieval and document search for grounded agent decisions |
| **Evaluation Engine** | Multi-metric scoring: functional correctness, tool usage, state integrity, safety, latency, cost |
| **Deterministic Verifiers** | Code-based verification (no LLM judge) for state invariants, tool patterns, and policy compliance |
| **Reliability Lab** | Controlled failure experiments: concurrency storms, idempotency, timeouts, duplicate requests |
| **Observability** | Structured logging, trace IDs, token/cost tracking per trial |
| **Security** | Tool-level ACL (allow/deny/approve), prompt-injection detection |
| **Regression Testing** | Version-to-version metric comparison with automatic regression detection |

---

## Quick Start

```bash
# Clone and enter the project
cd "AGENT FORGE"

# Option A: Docker Compose (recommended)
cp .env.example .env
docker compose up -d --build
open http://localhost:8000/docs

# Option B: Local dev
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn backend.main:app --reload
```

## Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Property-based tests
pytest tests/property/

# Reliability lab
pytest tests/integration/test_reliability_lab.py
```

---

## Project Structure

```
AGENT FORGE/
├── backend/                 # FastAPI API layer
│   ├── main.py              # App entry point
│   ├── config.py            # Pydantic settings
│   ├── database.py          # Async SQLAlchemy
│   ├── redis_client.py      # Redis connection
│   ├── models.py            # ORM models
│   ├── schemas.py           # API schemas
│   └── routers/             # API endpoints
├── agent/                   # Agent runtime
│   ├── runtime.py           # Orchestration loop
│   ├── state.py             # Execution context
│   ├── tool_executor.py     # Tool dispatch + registry
│   └── retry.py             # Retry / timeout utils
├── tools/                   # MCP tool implementations
│   ├── customer/            # search, tickets, create
│   ├── order/               # create, get, refund
│   └── document/            # search, policy retrieval
├── evaluation/              # Scoring + verification
│   ├── engine.py            # Weighted scoring
│   ├── metrics.py           # Metric data models
│   └── verifiers/           # Deterministic checks
├── lab/                     # Reliability experiments
│   ├── concurrency.py       # Race-condition tests
│   ├── idempotency.py       # At-most-once tests
│   └── experiments.py       # Suite runner
├── observability/           # Logging, tracing, metrics
├── security/                # ACL + prompt guard
├── tests/                   # Public test suites
│   ├── unit/
│   ├── integration/
│   └── property/
├── docs/                    # Design documentation
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Architecture

```
User / API
    │
    ▼
FastAPI Backend
    │
    ├── Agent Runtime ──► MCP Tools ──► Mock Services
    │                  └── RAG Layer
    │
    ├── Evaluation Engine
    │   ├── pytest (Test Runner)
    │   └── Deterministic Verifiers
    │
    ├── Reliability Lab
    │   ├── Concurrency experiments
    │   └── Idempotency experiments
    │
    ├── Observability
    │   └── Structured logs + trace IDs
    │
    └── Security
        ├── Tool ACL
        └── Prompt injection guard
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, Pydantic |
| Database | PostgreSQL + SQLAlchemy (async) |
| Cache | Redis |
| AI | OpenAI API |
| Testing | pytest, pytest-asyncio, Hypothesis |
| Observability | loguru (structured JSON) |
| Containers | Docker + Docker Compose |

---

## License

MIT
