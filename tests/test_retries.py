from __future__ import annotations

import pytest

from agent_runtime import AgentRuntime, CallTool, Finish, Journal, RunStatus
from agent_runtime.events import EventType
from agent_runtime.llm import ScriptedPlanner
from agent_runtime.tools import Tool, ToolRegistry


@pytest.fixture
def journal(tmp_path):
    j = Journal(tmp_path / "retries.db")
    yield j
    j.close()


def test_tool_is_retried_until_success(journal):
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError(f"temporary failure {attempts['count']}")
        return "recovered"

    registry = ToolRegistry()
    registry.register(Tool(name="flaky", description="Flaky tool.", handler=flaky))
    runtime = AgentRuntime(
        journal,
        ScriptedPlanner([CallTool("flaky"), Finish(output="done")]),
        registry,
        max_tool_retries=2,
    )

    state = runtime.start("recover")
    failures = journal.read(state.run_id)
    failed_events = [event for event in failures if event.type is EventType.TOOL_FAILED]

    assert state.status is RunStatus.COMPLETED
    assert state.transcript[0]["result"] == "recovered"
    assert attempts["count"] == 3
    assert len(failed_events) == 2
    assert [event.payload["final"] for event in failed_events] == [False, False]


def test_tool_failure_after_retries_is_feedback_to_agent(journal):
    def broken() -> str:
        raise ValueError("still broken")

    registry = ToolRegistry()
    registry.register(Tool(name="broken", description="Broken tool.", handler=broken))
    runtime = AgentRuntime(
        journal,
        ScriptedPlanner([CallTool("broken"), Finish(output="handled failure")]),
        registry,
        max_tool_retries=2,
    )

    state = runtime.start("handle broken tool")
    failures = [
        event
        for event in journal.read(state.run_id)
        if event.type is EventType.TOOL_FAILED
    ]

    assert state.status is RunStatus.COMPLETED
    assert state.output == "handled failure"
    assert "ValueError: still broken" in state.transcript[0]["error"]
    assert len(failures) == 3
    assert [event.payload["attempt"] for event in failures] == [0, 1, 2]
    assert failures[-1].payload["final"] is True
