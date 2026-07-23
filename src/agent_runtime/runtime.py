"""The durable agent loop.

The loop is a pure function of the journal. Starting a run and resuming a run
after a crash execute the *same* code path: both fold the existing events into
state, then continue. There is no separate "recovery" branch to drift out of
sync with the happy path — recovery correctness is exercised by every normal run.

Three rules make that work:

1. Planner decisions and human approvals are journalled before they are acted on;
   a tool execution-start marker is committed before entering its handler.
2. On replay, those recorded outcomes are *consumed* instead of re-produced. The
   planner is not re-asked and completed tool calls are not re-executed.
3. A non-idempotent start without an outcome is ambiguous, not retryable. The run
   stops for authoritative operator evidence because the external commit and
   local journal append cannot be one transaction.

When the journal runs out of recorded outcomes, the loop transparently goes
live. That boundary is the only difference between replay and fresh execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.errors import (
    MaxStepsExceeded,
    OutcomeResolutionError,
    ReplayDivergence,
    ToolExecutionError,
    ToolNotFound,
)
from agent_runtime.events import Event, EventType, RunStatus, new_id
from agent_runtime.journal import Journal
from agent_runtime.llm import (
    CallTool,
    Decision,
    Finish,
    Planner,
    decision_from_dict,
)
from agent_runtime.tools import Approval, ToolRegistry


@dataclass
class RunState:
    run_id: str
    status: RunStatus
    output: str | None = None
    error: str | None = None
    pending_approval: dict[str, Any] | None = None
    pending_recovery: dict[str, Any] | None = None
    steps: int = 0
    transcript: list[dict[str, Any]] = field(default_factory=list)
    # True when this pass served at least one outcome from the journal rather
    # than producing it live. Surfaced mainly so tests and operators can prove
    # a resume actually replayed instead of quietly redoing the work.
    replayed: bool = False


@dataclass
class _Folded:
    """Everything the loop needs, reconstructed from the event log."""

    decisions: list[Decision] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    outcomes: dict[str, tuple[str, str]] = field(default_factory=dict)
    approvals: dict[str, tuple[bool, str]] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    started: dict[str, dict[str, Any]] = field(default_factory=dict)
    unknown: dict[str, dict[str, Any]] = field(default_factory=dict)
    terminal: tuple[RunStatus, str] | None = None


def _fold(events: list[Event]) -> _Folded:
    f = _Folded()
    for ev in events:
        p = ev.payload
        match ev.type:
            case EventType.LLM_RESPONDED:
                f.decisions.append(decision_from_dict(p["decision"]))
            case EventType.TOOL_REQUESTED:
                f.requests.append(p)
            case EventType.APPROVAL_DECIDED:
                f.approvals[p["call_id"]] = (p["allowed"], p.get("reason", ""))
            case EventType.TOOL_EXECUTION_STARTED:
                f.started[p["call_id"]] = p
            case EventType.TOOL_OUTCOME_UNKNOWN:
                f.unknown[p["call_id"]] = p
            case EventType.TOOL_SUCCEEDED:
                f.outcomes[p["call_id"]] = ("ok", p["result"])
            case EventType.TOOL_FAILED:
                f.attempts[p["call_id"]] = f.attempts.get(p["call_id"], 0) + 1
                if p.get("final"):
                    f.outcomes[p["call_id"]] = ("error", p["error"])
            case EventType.RUN_COMPLETED:
                f.terminal = (RunStatus.COMPLETED, p["output"])
            case EventType.RUN_FAILED:
                f.terminal = (RunStatus.FAILED, p["error"])
            case _:
                pass
    return f


class AgentRuntime:
    def __init__(
        self,
        journal: Journal,
        planner: Planner,
        registry: ToolRegistry,
        *,
        max_steps: int = 25,
        max_tool_retries: int = 2,
    ) -> None:
        self._journal = journal
        self._planner = planner
        self._registry = registry
        self._max_steps = max_steps
        self._max_retries = max_tool_retries

    # -- public API ---------------------------------------------------------

    def start(self, goal: str) -> RunState:
        run_id = new_id("run")
        now = time.time()
        self._journal.create_run(run_id, goal, now)
        self._emit(run_id, EventType.RUN_STARTED, {"goal": goal})
        return self.advance(run_id)

    def approve(
        self, run_id: str, call_id: str, *, allowed: bool, reason: str = ""
    ) -> RunState:
        """Record a human decision, then continue the run."""
        self._emit(
            run_id,
            EventType.APPROVAL_DECIDED,
            {"call_id": call_id, "allowed": allowed, "reason": reason},
        )
        return self.advance(run_id)

    def resolve_tool_outcome(
        self,
        run_id: str,
        call_id: str,
        *,
        succeeded: bool,
        actor: str,
        evidence: dict[str, Any],
        result: str = "",
        error: str = "",
    ) -> RunState:
        """Record authoritative evidence for one ambiguous external call."""
        if not actor.strip():
            raise OutcomeResolutionError("actor is required")
        if not evidence:
            raise OutcomeResolutionError("resolution evidence is required")
        run = self._journal.get_run(run_id)
        if run is None:
            from agent_runtime.errors import RunNotFound

            raise RunNotFound(run_id)
        folded = _fold(self._journal.read(run_id))
        requested = next(
            (item for item in folded.requests if item["call_id"] == call_id),
            None,
        )
        if (
            requested is None
            or call_id not in folded.started
            or call_id in folded.outcomes
        ):
            raise OutcomeResolutionError(
                f"tool call {call_id} is not awaiting outcome resolution"
            )
        event_type = EventType.TOOL_SUCCEEDED if succeeded else EventType.TOOL_FAILED
        payload: dict[str, Any] = {
            "call_id": call_id,
            "resolved_by": actor,
            "resolution_evidence": evidence,
            "manual_resolution": True,
        }
        if succeeded:
            payload["result"] = result
        else:
            payload.update(
                {"error": error or "operator confirmed failure", "final": True}
            )
        self._emit(run_id, event_type, payload)
        self._journal.set_status(run_id, RunStatus.RUNNING, time.time())
        return self.advance(run_id)

    def advance(self, run_id: str) -> RunState:
        """Drive the run as far as it can go, then return its state.

        Safe to call repeatedly: a completed run returns its recorded output, and
        a suspended run returns its pending gate without re-doing any work.
        """
        run = self._journal.get_run(run_id)
        if run is None:
            from agent_runtime.errors import RunNotFound

            raise RunNotFound(run_id)

        events = self._journal.read(run_id)
        folded = _fold(events)

        if folded.terminal is not None:
            status, value = folded.terminal
            return RunState(
                run_id=run_id,
                status=status,
                output=value if status is RunStatus.COMPLETED else None,
                error=value if status is RunStatus.FAILED else None,
                transcript=self._rebuild_transcript(folded),
                # Answering from the journal is a replay, and callers rely on
                # this flag to prove no work was redone.
                replayed=True,
            )

        return self._loop(run_id, run["goal"], folded)

    # -- the loop -----------------------------------------------------------

    def _loop(self, run_id: str, goal: str, folded: _Folded) -> RunState:
        transcript: list[dict[str, Any]] = []
        specs = self._registry.specs()
        di = 0  # cursor into recorded planner decisions
        ri = 0  # cursor into recorded tool requests
        replayed = False

        for step in range(self._max_steps):
            # --- decide -----------------------------------------------------
            if di < len(folded.decisions):
                decision = folded.decisions[di]
                replayed = True  # served from the journal; planner not consulted
            else:
                decision = self._planner.decide(goal, transcript, specs)
                self._emit(
                    run_id, EventType.LLM_RESPONDED, {"decision": decision.as_dict()}
                )
            di += 1

            if isinstance(decision, Finish):
                self._emit(run_id, EventType.RUN_COMPLETED, {"output": decision.output})
                self._journal.set_status(run_id, RunStatus.COMPLETED, time.time())
                return RunState(
                    run_id=run_id,
                    status=RunStatus.COMPLETED,
                    output=decision.output,
                    steps=step,
                    transcript=transcript,
                    replayed=replayed,
                )

            assert isinstance(decision, CallTool)

            # --- identify the call ------------------------------------------
            if ri < len(folded.requests):
                recorded = folded.requests[ri]
                if recorded["name"] != decision.name:
                    # An integrity guard, not an everyday path: because planner
                    # decisions are themselves journalled and replayed verbatim,
                    # a redeployed agent cannot normally desync these two. This
                    # fires when the journal and the code genuinely disagree —
                    # a hand-edited log, a botched schema migration, or an event
                    # stream written by an incompatible version. Applying X's
                    # recorded result to Y would be silent corruption.
                    raise ReplayDivergence(
                        f"step {step}: journal recorded tool '{recorded['name']}' "
                        f"but the planner now asks for '{decision.name}'"
                    )
                call_id = recorded["call_id"]
                replayed = True
            else:
                # The request is persisted before lookup, approval, or handler
                # execution so a crash cannot make a resumed run invent a new
                # call_id for the same planned action.
                call_id = new_id("call")
                self._emit(
                    run_id,
                    EventType.TOOL_REQUESTED,
                    {
                        "call_id": call_id,
                        "name": decision.name,
                        "arguments": decision.arguments,
                    },
                )
            ri += 1

            # --- resolve the tool -------------------------------------------
            try:
                tool = self._registry.get(decision.name)
            except ToolNotFound as exc:
                # Not fatal: tell the agent so it can pick a different tool.
                transcript.append(
                    {"role": "tool", "name": decision.name, "error": str(exc)}
                )
                continue

            # --- approval gate ----------------------------------------------
            if tool.approval is Approval.REQUIRED and call_id not in folded.approvals:
                self._emit(
                    run_id,
                    EventType.APPROVAL_REQUESTED,
                    {
                        "call_id": call_id,
                        "name": decision.name,
                        "arguments": decision.arguments,
                    },
                )
                self._journal.set_status(run_id, RunStatus.AWAITING_APPROVAL, time.time())
                return RunState(
                    run_id=run_id,
                    status=RunStatus.AWAITING_APPROVAL,
                    pending_approval={
                        "call_id": call_id,
                        "name": decision.name,
                        "arguments": decision.arguments,
                    },
                    steps=step,
                    transcript=transcript,
                    replayed=replayed,
                )

            if call_id in folded.approvals:
                allowed, reason = folded.approvals[call_id]
                if not allowed:
                    # A denial is feedback, not a crash: the agent gets told and
                    # can choose another path.
                    transcript.append(
                        {
                            "role": "tool",
                            "name": decision.name,
                            "error": (
                                f"denied by human review: {reason or 'no reason given'}"
                            ),
                        }
                    )
                    continue

            # --- execute (or replay) ----------------------------------------
            if call_id in folded.outcomes:
                kind, value = folded.outcomes[call_id]
                replayed = True  # never re-invoke; may have had side effects
            elif call_id in folded.started and not tool.idempotent:
                # There is proof execution began but no durable proof of its
                # outcome. The external effect may already exist. Re-firing is
                # unsafe, regardless of how capable the planner is.
                if call_id not in folded.unknown:
                    self._emit(
                        run_id,
                        EventType.TOOL_OUTCOME_UNKNOWN,
                        {
                            "call_id": call_id,
                            "name": decision.name,
                            "reason": "execution_started_without_durable_outcome",
                        },
                    )
                return self._awaiting_recovery(
                    run_id,
                    call_id,
                    decision,
                    step,
                    transcript,
                    replayed=True,
                )
            else:
                # This is the only point where the outside world may be touched.
                # Once _execute journals an outcome, every later resume must use
                # that record instead of calling the handler again.
                kind, value = self._execute(run_id, call_id, tool, decision.arguments)

            if kind == "unknown":
                return self._awaiting_recovery(
                    run_id,
                    call_id,
                    decision,
                    step,
                    transcript,
                    replayed=replayed,
                    reason=value,
                )

            outcome_field = {"result": value} if kind == "ok" else {"error": value}
            transcript.append({"role": "tool", "name": decision.name, **outcome_field})
            self._journal.set_status(run_id, RunStatus.RUNNING, time.time())

        error = f"exceeded max_steps={self._max_steps} without finishing"
        self._emit(run_id, EventType.RUN_FAILED, {"error": error})
        self._journal.set_status(run_id, RunStatus.FAILED, time.time())
        raise MaxStepsExceeded(error)

    def _execute(
        self, run_id: str, call_id: str, tool, arguments: dict
    ) -> tuple[str, str]:
        last_error = ""
        max_attempts = self._max_retries + 1 if tool.idempotent else 1
        for attempt in range(max_attempts):
            self._emit(
                run_id,
                EventType.TOOL_EXECUTION_STARTED,
                {
                    "call_id": call_id,
                    "name": tool.name,
                    "attempt": attempt,
                    "idempotent": tool.idempotent,
                },
            )
            try:
                result = tool.handler(**arguments)
            except Exception as exc:  # noqa: BLE001 - tool code is untrusted
                last_error = f"{type(exc).__name__}: {exc}"
                if not tool.idempotent:
                    self._emit(
                        run_id,
                        EventType.TOOL_OUTCOME_UNKNOWN,
                        {
                            "call_id": call_id,
                            "name": tool.name,
                            "reason": "non_idempotent_handler_raised",
                            "error": last_error,
                        },
                    )
                    return "unknown", last_error
                final = attempt == max_attempts - 1
                self._emit(
                    run_id,
                    EventType.TOOL_FAILED,
                    {
                        "call_id": call_id,
                        "error": last_error,
                        "attempt": attempt,
                        "final": final,
                    },
                )
                if final:
                    return "error", last_error
                continue

            text = result if isinstance(result, str) else repr(result)
            self._emit(
                run_id,
                EventType.TOOL_SUCCEEDED,
                {"call_id": call_id, "result": text},
            )
            return "ok", text
        return "error", last_error  # pragma: no cover - loop always returns

    def _awaiting_recovery(
        self,
        run_id: str,
        call_id: str,
        decision: CallTool,
        step: int,
        transcript: list[dict[str, Any]],
        *,
        replayed: bool,
        reason: str = "outcome is unknown after interrupted execution",
    ) -> RunState:
        self._journal.set_status(run_id, RunStatus.AWAITING_RECOVERY, time.time())
        return RunState(
            run_id=run_id,
            status=RunStatus.AWAITING_RECOVERY,
            pending_recovery={
                "call_id": call_id,
                "name": decision.name,
                "arguments": decision.arguments,
                "reason": reason,
            },
            steps=step,
            transcript=transcript,
            replayed=replayed,
        )

    # -- helpers ------------------------------------------------------------

    def _rebuild_transcript(self, folded: _Folded) -> list[dict[str, Any]]:
        out = []
        for req in folded.requests:
            kind_value = folded.outcomes.get(req["call_id"])
            entry: dict[str, Any] = {"role": "tool", "name": req["name"]}
            if kind_value is not None:
                kind, value = kind_value
                entry["result" if kind == "ok" else "error"] = value
            out.append(entry)
        return out

    def _emit(self, run_id: str, etype: EventType, payload: dict) -> Event:
        return self._journal.append(Event(run_id=run_id, type=etype, payload=payload))


__all__ = ["AgentRuntime", "RunState", "ToolExecutionError"]
