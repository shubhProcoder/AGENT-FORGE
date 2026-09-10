# AgentForge Task Tracker

## Phase 1: Core Execution Architecture
- `[x]` **TASK 1**: Architecture + domain contracts (Define domain entities in `backend/domain/`, set up evidence hierarchy rules)
- `[x]` **TASK 2**: FastAPI + PostgreSQL (Implement `api/`, `application/`, and `infrastructure/database/` layers)
- `[x]` **TASK 3**: Evaluation / Run / Trial lifecycle (State machines, Dataset/Task structures)
- `[x]` **TASK 4**: Mock Agent (A deterministic agent runner with explicit stopping controls)
- `[x]` **TASK 5**: MCP Layer (Separate Platform Control API vs Agent MCP Server)
- `[x]` **TASK 6**: Customer-support environment (Execution Plane DB fixtures, provisioning/freezing)
- `[x]` **TASK 7**: Test Runner adapter (Abstracting pytest integration)
- `[x]` **TASK 8**: Verifier + State Diff Engine (Before/after snapshots, evaluating against the frozen environment)

## Phase 2: Reliability & Edge Cases
- `[x]` **TASK 9**: Failure Injection & Idempotency (Testing at-most-once semantics)
- `[x]` **TASK 10**: Concurrency (Testing race conditions)
- `[x]` **TASK 11**: Hidden tests & Test versioning

## Phase 3: Advanced Agent Capabilities
- `[x]` **TASK 12**: Real LLM integration
  - `[x]` **TASK 12.1**: ModelProvider abstraction & OpenAI provider
  - `[x]` **TASK 12.2**: Real Agent Runtime (controlled tool loop, stopping bounds, loop detection, execution event telemetry)
- `[x]` **TASK 13**: Code sandbox (Isolated execution service)
  - `[x]` **TASK 13.1**: CodeExecutionService abstraction & ExecutionResult domain contracts
  - `[x]` **TASK 13.2**: DockerCodeExecutor hardened container sandbox & process runners
  - `[x]` **TASK 13.3**: Code evaluation pipeline (pytest, JUnit XML, public vs hidden isolation, second_largest_unique scenario)
- `[/]` **TASK 14**: Knowledge & RAG Evaluation Layer
  - `[x]` **TASK 14.1**: Knowledge Domain + Retrieval Contract
  - `[x]` **TASK 14.2**: Document Ingestion Pipeline
  - `[ ]` **TASK 14.3**: Hybrid Retrieval Engine
  - `[ ]` **TASK 14.4**: Evidence / Citation Layer
  - `[ ]` **TASK 14.5**: Agent RAG Integration
  - `[ ]` **TASK 14.6**: RAG Evaluation & Verifiers
  - `[ ]` **TASK 14.7**: Retrieval Failure Lab
  - `[ ]` **TASK 14.8**: RAG Regression Benchmark

## Phase 4: Polish
- `[ ]` **TASK 15**: Observability (Telemetry, tracing)
- `[ ]` **TASK 16**: Regression + benchmarking
- `[ ]` **TASK 17**: UI Dashboard
