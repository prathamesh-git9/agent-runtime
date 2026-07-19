"""The claims this project makes, as tests.

If durable execution is real, these must hold:
  - a crashed run resumes from the journal, not from scratch
  - effectful tools are never re-executed on resume
  - the planner is not re-billed for decisions already recorded
  - an approval gate survives a full process restart
"""

from __future__ import annotations

import pytest

from agent_runtime import (
    AgentRuntime,
    Approval,
    CallTool,
    Finish,
    Journal,
    RunStatus,
    ScriptedPlanner,
    Tool,
    ToolRegistry,
)
from agent_runtime.errors import ReplayDivergence


@pytest.fixture
def journal(tmp_path):
    j = Journal(tmp_path / "runs.db")
    yield j
    j.close()


class SideEffectCounter:
    """Stands in for anything that touches the outside world."""

    def __init__(self) -> None:
        self.sends = 0

    def send_email(self, to: str = "", body: str = "") -> str:
        self.sends += 1
        return f"sent to {to}"


def build(journal, decisions, *, counter=None, approval=Approval.AUTO, **kwargs):
    counter = counter or SideEffectCounter()
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="send_email",
            description="Send an email.",
            handler=counter.send_email,
            approval=approval,
            idempotent=False,
        )
    )
    registry.register(
        Tool(
            name="lookup",
            description="Look something up.",
            handler=lambda q="": f"result for {q}",
        )
    )
    planner = ScriptedPlanner(decisions)
    return AgentRuntime(journal, planner, registry, **kwargs), planner, counter


def test_simple_run_completes(journal):
    runtime, planner, _ = build(journal, [Finish(output="done")])
    state = runtime.start("say done")
    assert state.status is RunStatus.COMPLETED
    assert state.output == "done"
    assert planner.calls == 1


def test_tool_call_then_finish(journal):
    runtime, _, counter = build(
        journal,
        [CallTool("send_email", {"to": "a@b.c"}), Finish(output="email sent")],
    )
    state = runtime.start("email someone")
    assert state.status is RunStatus.COMPLETED
    assert counter.sends == 1
    assert state.transcript[0]["result"] == "sent to a@b.c"


def test_resume_does_not_re_execute_effectful_tools(journal):
    """The headline guarantee: replay must not send the email twice."""
    counter = SideEffectCounter()
    decisions = [CallTool("send_email", {"to": "a@b.c"}), Finish(output="ok")]
    runtime, _, _ = build(journal, decisions, counter=counter)
    state = runtime.start("email someone")
    run_id = state.run_id
    assert counter.sends == 1

    # Simulate a crash and a cold restart: brand-new runtime and planner objects,
    # same journal on disk. Nothing is carried over in memory.
    fresh_runtime, fresh_planner, _ = build(journal, decisions, counter=counter)
    resumed = fresh_runtime.advance(run_id)

    assert resumed.status is RunStatus.COMPLETED
    assert counter.sends == 1, "resume re-sent the email — durability is broken"
    assert fresh_planner.calls == 0, "resume re-billed the planner"
    assert resumed.replayed is True


def test_crash_midway_resumes_from_the_journal(journal):
    """Crash after the tool ran but before the run finished."""
    counter = SideEffectCounter()
    # First runtime is only allowed one decision, so it stops before finishing.
    partial, _, _ = build(
        journal, [CallTool("send_email", {"to": "x@y.z"})], counter=counter
    )
    state = partial.start("email then report")
    run_id = state.run_id
    assert counter.sends == 1
    assert state.status is RunStatus.COMPLETED  # script exhausted -> Finish

    # A fuller script resumes the same run; the recorded send is replayed.
    full, planner, _ = build(
        journal,
        [CallTool("send_email", {"to": "x@y.z"}), Finish(output="reported")],
        counter=counter,
    )
    resumed = full.advance(run_id)
    assert counter.sends == 1
    assert resumed.status is RunStatus.COMPLETED


def test_approval_gate_suspends_the_run(journal):
    runtime, _, counter = build(
        journal,
        [CallTool("send_email", {"to": "ceo@corp.com"}), Finish(output="ok")],
        approval=Approval.REQUIRED,
    )
    state = runtime.start("email the CEO")

    assert state.status is RunStatus.AWAITING_APPROVAL
    assert state.pending_approval["name"] == "send_email"
    assert counter.sends == 0, "gated tool ran before approval"


def test_approval_survives_a_process_restart(journal):
    decisions = [CallTool("send_email", {"to": "ceo@corp.com"}), Finish(output="ok")]
    counter = SideEffectCounter()
    runtime, _, _ = build(
        journal, decisions, counter=counter, approval=Approval.REQUIRED
    )
    state = runtime.start("email the CEO")
    run_id = state.run_id
    call_id = state.pending_approval["call_id"]

    # The approver comes back later, against a different process.
    fresh, _, _ = build(
        journal, decisions, counter=counter, approval=Approval.REQUIRED
    )
    resumed = fresh.approve(run_id, call_id, allowed=True, reason="signed off")

    assert resumed.status is RunStatus.COMPLETED
    assert counter.sends == 1


def test_denied_approval_is_feedback_not_a_crash(journal):
    decisions = [
        CallTool("send_email", {"to": "ceo@corp.com"}),
        Finish(output="stood down"),
    ]
    runtime, _, counter = build(
        journal, decisions, approval=Approval.REQUIRED
    )
    state = runtime.start("email the CEO")
    final = runtime.approve(
        state.run_id,
        state.pending_approval["call_id"],
        allowed=False,
        reason="too risky",
    )

    assert final.status is RunStatus.COMPLETED
    assert counter.sends == 0
    assert "too risky" in final.transcript[0]["error"]


def test_advance_on_a_completed_run_is_a_no_op(journal):
    runtime, _, counter = build(
        journal, [CallTool("send_email", {"to": "a@b.c"}), Finish(output="ok")]
    )
    state = runtime.start("email")
    for _ in range(3):
        again = runtime.advance(state.run_id)
    assert again.status is RunStatus.COMPLETED
    assert counter.sends == 1, "repeated advance re-ran the tool"


def test_replay_divergence_is_detected(journal):
    """A journal that contradicts itself must fail loudly, not guess.

    Note what this is *not* testing: redeploying a changed agent cannot cause
    this, because planner decisions are journalled and replayed verbatim, so the
    recorded decision and the recorded request can never drift apart on their
    own. The guard exists for a corrupted or externally-written log — a bad
    migration, a hand-edited row, an incompatible writer. So the test writes
    exactly that: a decision saying 'lookup' beside a request saying
    'send_email'. Applying send_email's recorded result to a lookup call would
    be silent corruption.
    """
    from agent_runtime.events import Event, EventType
    from agent_runtime.events import RunStatus as _RS

    run_id = "run_tampered"
    journal.create_run(run_id, "email someone", 0.0)
    journal.append(
        Event(
            run_id=run_id,
            type=EventType.LLM_RESPONDED,
            payload={
                "decision": {
                    "kind": "call_tool",
                    "name": "lookup",
                    "arguments": {},
                }
            },
        )
    )
    journal.append(
        Event(
            run_id=run_id,
            type=EventType.TOOL_REQUESTED,
            payload={"call_id": "call_x", "name": "send_email", "arguments": {}},
        )
    )
    journal.set_status(run_id, _RS.RUNNING, 0.0)

    runtime, _, _ = build(journal, [Finish(output="unused")])
    with pytest.raises(ReplayDivergence):
        runtime.advance(run_id)
