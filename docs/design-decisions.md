# Design Decisions

This document captures the rationale behind key architectural and technology choices in AgentForge.

---

## Why PostgreSQL instead of MongoDB?

Agent evaluation requires **relational integrity**. Trials reference agents and tasks via foreign keys. Verification results must be atomically linked to trials. PostgreSQL provides:

- ACID transactions (critical for the Reliability Lab).
- JSONB columns for flexible schema (tool call arguments, metrics).
- pgvector extension for future RAG embedding storage.
- Mature async driver (asyncpg) with connection pooling.

MongoDB would offer schema flexibility, but at the cost of transaction guarantees that our state-invariant verifiers depend on.

---

## Why separate Agent MCP vs Platform Control API?

Using a single MCP server for both the agent's tools and the platform's evaluation controls (e.g., `run_evaluation()`) creates a dangerous trust boundary. An evaluated agent must never have access to the controls grading it.

By separating them:
1. **Agent MCP Server**: Exposes tools the agent is allowed to use (`search_customer`, `create_ticket`).
2. **Platform Control API**: A separate FastAPI interface for scheduling runs, inspecting results, and managing the lifecycle.

## Why a Trial Manager?

A Trial Manager is distinct from the Run Manager. 
- **Run Manager** tracks the execution of a job across multiple attempts or parallel iterations.
- **Trial Manager** is responsible for a single, isolated attempt. It provisions the environment, enforces timeouts, cancels executions, freezes state, and destroys the sandbox afterwards.

Without this abstraction, the Run Manager or the Agent Runtime itself becomes a massive, tangled orchestration class.

## Why a Failure Injection Engine?

The Reliability Lab requires testing edge cases like database timeouts, network failures, or duplicate responses. Without a failure injection engine, reliability testing relies on race conditions or non-deterministic mocks.
By adding a failure injection layer between the Agent MCP and the backend tools, we can deterministically say: "Make `refund_order()` return a timeout on attempt 1", ensuring our reliability experiments are 100% reproducible.

## Why a State Snapshot / Diff Engine?

Silent mutations (e.g., an agent updating a customer's email when it was only supposed to change ticket status) are hard to catch with manual assertions. 
A State Diff Engine takes a snapshot of the environment *before* and *after* the agent acts, and compares the delta against a strict **Mutation Policy**. If unexpected fields are changed, the evaluation automatically fails.

---

## The Evidence Hierarchy

Do not make the LLM the evaluator of everything. Evaluation relies on an evidence hierarchy, prioritizing the strongest, most objective signals over subjective or generated text:

1. **System Invariants** (Strongest - e.g. total balance equals sum of transactions)
2. **Database State** (e.g. `refund_count = 2`)
3. **Executable Tests** (e.g. `pytest` passing against a generated file)
4. **API Responses** (e.g. HTTP 201 Created)
5. **Tool-call Records** (e.g. tracing exactly which MCP tools were invoked)
6. **Structured Output** (e.g. JSON extraction)
7. **LLM-as-judge** (Useful for subjective tone grading)
8. **Natural-language self-report** (Weakest - e.g. the agent saying "Payment refunded successfully")

If the agent claims success but the database state says otherwise, **the database wins**.

## Two Environments: Control Plane vs. Execution Plane

To ensure reliability, the system is strictly divided:
- **Control Plane**: AgentForge itself (projects, evaluations, runs, UI, API).
- **Execution Plane**: Where the agent actually operates (temporary database, MCP servers, code sandboxes).

You do not want arbitrary agent behavior executing inside your primary API service. By freezing the Execution Plane at the end of a trial, evaluators can safely inspect the state without risking corruption to the Control Plane.

## Agent Runtime Controls

While the Agent Runtime executes a relatively simple loop (call LLM -> execute tool -> append result -> repeat), it must be wrapped in strict operational safety controls:
- `max_iterations`
- `max_tool_calls`
- `timeout`
- `budget`
- `allowed_tools`

Autonomous loops can otherwise continue indefinitely, running up API costs or getting stuck in failure loops.

---

## Why deterministic graders instead of LLM-as-judge?

An LLM judge introduces:

- **Non-determinism**: Same input can produce different verdicts.
- **Cost**: Every evaluation run incurs API charges.
- **Circular reasoning**: Using an LLM to evaluate an LLM.
- **Opacity**: Hard to debug why a particular score was given.

Our verifiers are pure Python/SQL functions. They check observable state (database rows, tool call sequences, field values) against specifications. This means:

- Tests are **reproducible** — same input always gives same result.
- Verification is **free** — no API calls.
- Failures are **explainable** — you see exactly which invariant was violated.

We may add LLM-based evaluators later for natural-language quality, but deterministic verifiers remain the primary scoring mechanism.

---

## Why hidden tests?

Separating `tests/` from `private_tests/` mirrors real-world evaluation platforms:

- **Public tests** show candidates (or the agent) what is expected.
- **Hidden tests** verify edge cases that weren't explicitly stated.

This prevents the agent from being "trained to the test" and measures genuine task understanding.

---

## Why use idempotency keys?

Real APIs face network retries. Without idempotency:

- A retried `POST /orders` creates a duplicate order.
- A retried `POST /refunds` issues a double refund.

Idempotency keys guarantee **at-most-once** semantics. Our tools implement this pattern so we can test whether agents (or callers) correctly use idempotency — a practical backend engineering skill.

---

## What happens when the evaluator itself fails?

The evaluation pipeline has three failure modes:

1. **Agent fails** (timeout, error, loop) → Trial status = `failed`, verifiers still run on partial state.
2. **Verifier fails** (uncaught exception) → Caught by the scoring engine, metric defaults to 0.0, failure is logged.
3. **Scoring engine fails** (configuration error) → Trial is marked `error`, raw verification data is preserved for debugging.

We never silently swallow failures.

---

## What are the limitations of our sandbox?

In the MVP:

- Isolation is **per-trial in-memory** (mock stores with `reset_*` functions), not per-process or per-container.
- There is no true process-level sandboxing for code execution.
- Mock tools do not simulate network latency or partial failures (the Reliability Lab covers this separately).

Future versions could use Docker-based isolation or database schema prefixing for stronger guarantees.

---

## Where can the verifier produce false positives?

- **Tool usage verifier**: If the expected tool sequence is too rigid, a valid alternative path may be marked as incorrect. We mitigate this by making sequence checks optional.
- **State invariant verifier**: Relies on the mock store being the single source of truth. If a tool modifies state outside the store, the verifier won't catch it.
- **Policy verifier**: The keyword-based approach may flag legitimate uses of words like "delete" in benign contexts. This is a known limitation of heuristic guards.

---

## Why don't we grade every natural-language response with an LLM?

Because:

1. It's expensive at scale.
2. It introduces non-determinism into the evaluation pipeline.
3. For agent tasks, the **state** matters more than the **prose**.

An agent that says "I successfully processed your refund" but leaves `refund_count = 0` has failed, regardless of how eloquent the response is.

We focus on **behavioral correctness** over **textual quality**.

---

## Why OpenAI instead of [other provider]?

No strong preference. The runtime is designed to be provider-agnostic:

- `openai` library is used because it has the most mature function-calling API.
- The `model` parameter is configurable per agent.
- Swapping to Anthropic or Gemini requires changing the client in `runtime.py`.

The choice was driven by ecosystem maturity, not a philosophical commitment.

---

## Why structured logging with loguru instead of Python's logging?

- **Contextual binding**: `logger.bind(trace_id=...)` propagates context without passing loggers around.
- **Human-readable development output**: Colored, formatted logs for local dev.
- **JSON serialization**: One flag switch for production JSON output.
- **Less boilerplate**: No handler/formatter configuration ceremony.

Standard library `logging` works fine, but loguru reduces setup friction for a portfolio project.
