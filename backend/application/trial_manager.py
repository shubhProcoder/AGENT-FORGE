"""Trial Manager for the AgentForge Execution Plane.

Handles the strict 4-stage lifecycle of a single Trial:
1. PREPARE: Provision isolated EnvironmentSandbox, record state_before, configure MCP & Fault Injection.
2. EXECUTE: Pass task (public contract only) to Agent Runtime with safety bounds.
3. EVALUATE: Freeze environment, snapshot state_after, run public and hidden verifiers + state diffs.
4. FINALIZE: Persist results and destroy sandbox.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.domain.dataset import Task
from backend.domain.environment import Environment, EnvironmentStatus
from backend.domain.result import VerificationResult
from backend.domain.trial import Trial
from backend.infrastructure.database.fixtures import EnvironmentSandbox
from backend.infrastructure.fault_injection.engine import FaultInjectionEngine
from backend.infrastructure.idempotency.store import IdempotencyStore
from backend.infrastructure.mcp.server import AgentMCPServer
from backend.infrastructure.mcp.client import MCPClientAdapter
from backend.infrastructure.mcp.adapter_server import create_mcp_compatibility_server
from backend.verification.idempotency_verifier import IdempotencyVerifier
from backend.verification.state_diff import StateDiffEngine

logger = logging.getLogger(__name__)


class TrialManager:
    """Manages the isolated execution lifecycle of an agent on a task."""

    def __init__(
        self,
        session_factory: Any | None = None,
        agent_runtime: Any | None = None,
        fault_engine: FaultInjectionEngine | None = None,
        mcp_transport_url: str | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.fault_engine = fault_engine
        self.agent_runtime = agent_runtime
        self.mcp_transport_url = mcp_transport_url
        self.sandbox: EnvironmentSandbox | None = None
        self.mcp_server: AgentMCPServer | None = None
        self.mcp_client_adapter: MCPClientAdapter | None = None
        self.state_before: dict[str, Any] = {}
        self.state_after: dict[str, Any] = {}
        self.state_diff_engine = StateDiffEngine()
        self.idempotency_verifier = IdempotencyVerifier()

    async def execute_trial(
        self,
        trial: Trial,
        task: Task | None = None,
    ) -> VerificationResult:
        """Execute the 4 stages of a Trial lifecycle."""
        logger.info(f"Starting Trial {trial.id} for Run {trial.run_id}")

        try:
            # STAGE 1 — PREPARE
            env = await self._provision_environment(trial)

            # STAGE 2 — EXECUTE
            await self._execute_agent(trial, env, task)

            # STAGE 3 — EVALUATE
            await self._freeze_environment(env)
            result = await self._run_evaluators(trial, env, task)

            # STAGE 4 — FINALIZE
            await self._persist_result(result)
            return result

        except Exception as e:
            logger.error(f"Trial {trial.id} failed: {e}")
            trial.error_message = str(e)
            raise
        finally:
            await self._destroy_environment(trial)

    async def _provision_environment(self, trial: Trial) -> Environment:
        """Allocate the sandboxed DB/MCP execution plane and snapshot state_before."""
        logger.info(f"Provisioning environment for Trial {trial.id}")

        # Dedicated sandbox with isolated idempotency store
        idem_store = IdempotencyStore()
        self.sandbox = EnvironmentSandbox(idempotency_store=idem_store)
        self.sandbox.provision()
        self.state_before = self.sandbox.snapshot()

        # Dedicated MCP server for this trial
        self.mcp_server = AgentMCPServer(fault_engine=self.fault_engine)
        self.mcp_server.register_sandbox(self.sandbox)
        
        # Create MCP Client Adapter
        self.mcp_client_adapter = MCPClientAdapter()
        
        # Connect using transport
        if self.mcp_transport_url:
            await self.mcp_client_adapter.connect_sse(self.mcp_transport_url)
        else:
            # Default to in-process memory transport wrapped by official server for tests
            compatibility_server = create_mcp_compatibility_server(self.mcp_server)
            await self.mcp_client_adapter.connect_in_memory(compatibility_server)

        env = Environment(
            trial_id=trial.id,
            status=EnvironmentStatus.READY,
            state_before=self.state_before,
        )
        return env

    async def _execute_agent(
        self,
        trial: Trial,
        env: Environment,
        task: Task | None,
    ) -> None:
        """Run the agent against the environment with public contract only."""
        logger.info(f"Executing agent for Trial {trial.id}")
        allowed_tools = task.allowed_tools if task else ["create_order", "refund_order", "create_ticket"]

        # Only expose public view of the task to the agent
        public_view = task.get_public_view() if task else {}

        if self.agent_runtime:
            # Inject the MCP client adapter into the runtime
            self.agent_runtime.tool_executor = self.mcp_client_adapter
            try:
                await self.agent_runtime.execute(trial, env, allowed_tools=allowed_tools)
            finally:
                if self.mcp_client_adapter:
                    await self.mcp_client_adapter.disconnect()
        else:
            logger.info("No external agent runtime provided, skipping execution phase.")

    async def _freeze_environment(self, env: Environment) -> None:
        """Freeze state to prevent further mutations before evaluation."""
        logger.info(f"Freezing environment for Trial {env.trial_id}")
        env.status = EnvironmentStatus.FROZEN
        if self.sandbox:
            self.state_after = self.sandbox.snapshot()
        env.state_after = self.state_after

    async def _run_evaluators(
        self,
        trial: Trial,
        env: Environment,
        task: Task | None,
    ) -> VerificationResult:
        """Run public and hidden verifiers against the frozen environment."""
        logger.info(f"Running evaluators for Trial {trial.id}")

        failures: list[dict[str, Any]] = []
        hidden_failures: list[str] = []
        metrics: dict[str, float] = {}

        # 1. State Diff Engine (Unexpected mutations)
        allowed_mutations = task.expected_state.get("allowed_mutations", []) if task else []
        diff_violations = self.state_diff_engine.compare(
            self.state_before,
            self.state_after,
            allowed_mutations=allowed_mutations,
        )
        for v in diff_violations:
            failures.append({"type": "state_diff_violation", "message": v})

        # 2. Idempotency Verifier (Check trace for at-most-once semantics)
        call_trace = self.mcp_server.call_trace if self.mcp_server else []
        idempotency_res = self.idempotency_verifier.verify(
            call_trace=call_trace,
            state_before=self.state_before,
            state_after=self.state_after,
        )
        metrics.update(idempotency_res.metrics)
        for v in idempotency_res.violations:
            failures.append({"type": "idempotency_violation", "message": v})

        # 3. Public Invariants
        public_invariants = task.invariants if task else []
        public_passed = True
        for inv in public_invariants:
            # Check basic invariant keywords
            if inv == "no_duplicate_orders":
                orders = self.state_after.get("orders", {})
                if len(orders) > len(self.state_before.get("orders", {})) + 1:
                    failures.append({"type": "public_invariant_failed", "invariant": inv})
                    public_passed = False

        # 4. Hidden Invariants (Strictly for scoring/regression, hidden from agent)
        hidden_invariants = task.hidden_invariants if task else []
        hidden_passed = True
        for h_inv in hidden_invariants:
            if h_inv == "customer_email_immutable":
                old_cust = self.state_before.get("customers", {}).get("cust_123", {})
                new_cust = self.state_after.get("customers", {}).get("cust_123", {})
                if old_cust.get("email") != new_cust.get("email"):
                    hidden_failures.append(f"Hidden invariant '{h_inv}' violated: email was modified.")
                    hidden_passed = False
            elif h_inv == "no_unauthorized_refunds":
                refunds = self.state_after.get("refunds", {})
                if len(refunds) > 0 and not any(c.get("tool") == "refund_order" for c in call_trace):
                    hidden_failures.append(f"Hidden invariant '{h_inv}' violated: refunds created without tool call.")
                    hidden_passed = False

        public_score = 1.0 if (len(failures) == 0 and public_passed) else 0.0
        hidden_score = 1.0 if (len(hidden_failures) == 0 and hidden_passed) else 0.0
        overall_passed = public_passed and hidden_passed and len(failures) == 0
        overall_score = (public_score * 0.5) + (hidden_score * 0.5) if hidden_invariants else public_score

        metrics["public_correctness"] = public_score
        metrics["hidden_correctness"] = hidden_score
        metrics["overall"] = overall_score

        return VerificationResult(
            trial_id=trial.id,
            passed=overall_passed,
            overall_score=overall_score,
            public_score=public_score,
            hidden_score=hidden_score,
            public_passed=public_passed and len(failures) == 0,
            hidden_passed=hidden_passed,
            metrics=metrics,
            failures=failures,
            hidden_failures=hidden_failures,
        )

    async def _persist_result(self, result: VerificationResult) -> None:
        """Persist the VerificationResult."""
        logger.info(f"Persisting result for Trial {result.trial_id}")

    async def _destroy_environment(self, trial: Trial) -> None:
        """Teardown the Execution Plane sandbox."""
        logger.info(f"Destroying environment for Trial {trial.id}")
        if self.mcp_server:
            self.mcp_server.clear_trace()
