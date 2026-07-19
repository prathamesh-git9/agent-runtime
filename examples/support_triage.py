from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_runtime import (  # noqa: E402
    AgentRuntime,
    Approval,
    CallTool,
    Finish,
    Journal,
    ScriptedPlanner,
    ToolRegistry,
)


def print_state(label, state) -> None:
    print(f"{label}: run_id={state.run_id} status={state.status}")
    if state.pending_approval:
        print(f"  pending={state.pending_approval}")
    if state.output:
        print(f"  output={state.output}")
    if state.transcript:
        print(f"  transcript={state.transcript}")


def build_runtime(journal: Journal, refunds: list[str]) -> AgentRuntime:
    registry = ToolRegistry()

    @registry.tool(
        "fetch_ticket",
        description="Fetch a support ticket by id.",
        parameters={"ticket_id": "string"},
    )
    def fetch_ticket(ticket_id: str) -> str:
        return (
            f"ticket {ticket_id}: duplicate annual subscription charge; "
            "customer asks for refund"
        )

    @registry.tool(
        "search_kb",
        description="Search the support knowledge base.",
        parameters={"query": "string"},
    )
    def search_kb(query: str) -> str:
        return f"policy for '{query}': duplicate charges qualify for a refund"

    @registry.tool(
        "issue_refund",
        description="Issue a customer refund.",
        approval=Approval.REQUIRED,
        idempotent=False,
        parameters={"ticket_id": "string", "amount": "number"},
    )
    def issue_refund(ticket_id: str, amount: int) -> str:
        print(f"MONEY MOVED: refunded ${amount} for {ticket_id}")
        refunds.append(ticket_id)
        return f"refunded ${amount}"

    planner = ScriptedPlanner(
        [
            CallTool("fetch_ticket", {"ticket_id": "TCK-1042"}),
            CallTool("search_kb", {"query": "duplicate subscription charge"}),
            CallTool("issue_refund", {"ticket_id": "TCK-1042", "amount": 49}),
            Finish(output="Refund issued and ticket marked ready for closure."),
        ]
    )
    return AgentRuntime(journal, planner, registry)


def main() -> None:
    refunds: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "support-triage.db"
        journal = Journal(path)
        runtime = build_runtime(journal, refunds)

        state = runtime.start("Triage support ticket TCK-1042")
        print_state("after start", state)

        state = runtime.approve(
            state.run_id,
            state.pending_approval["call_id"],
            allowed=True,
            reason="duplicate charge confirmed",
        )
        print_state("after approval", state)

        recovered = runtime.advance(state.run_id)
        print_state("after recovery advance", recovered)
        print(f"refund executions={len(refunds)}")

        journal.close()


if __name__ == "__main__":
    main()
