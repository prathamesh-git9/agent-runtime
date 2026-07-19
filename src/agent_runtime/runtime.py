"""The durable agent loop.

The loop is a pure function of the journal. Starting a run and resuming a run
after a crash execute the *same* code path: both fold the existing events into
state, then continue. There is no separate "recovery" branch to drift out of
sync with the happy path — recovery correctness is exercised by every normal run.

Two rules make that work:

1. Every non-deterministic outcome (a planner decision, a tool result, a human
   approval) is journalled before it is acted on.
2. On replay, those recorded outcomes are *consumed* instead of re-produced. The
   planner is not re-asked and effectful tools are not re-executed — which is
   the entire point. Re-running `charge_card` to rebuild state would be a bug
   that costs real money.

When the journal runs out of recorded outcomes, the loop transparently goes
live. That boundary is the only difference between replay and fresh execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.errors import (
    MaxStepsExceeded,
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
                self._journal.set_status(
                    run_id, RunStatus.AWAITING_APPROVAL, time.time()
                )
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
                                "denied by human review: "
                                f"{reason or 'no reason given'}"
                            ),
                        }
                    )
                    continue

            # --- execute (or replay) ----------------------------------------
            if call_id in folded.outcomes:
                kind, value = folded.outcomes[call_id]
                replayed = True  # never re-invoke; may have had side effects
            else:
                kind, value = self._execute(run_id, call_id, tool, decision.arguments)

            outcome_field = {"result": value} if kind == "ok" else {"error": value}
            transcript.append(
                {"role": "tool", "name": decision.name, **outcome_field}
            )
            self._journal.set_status(run_id, RunStatus.RUNNING, time.time())

        error = f"exceeded max_steps={self._max_steps} without finishing"
        self._emit(run_id, EventType.RUN_FAILED, {"error": error})
        self._journal.set_status(run_id, RunStatus.FAILED, time.time())
        raise MaxStepsExceeded(error)

    def _execute(
        self, run_id: str, call_id: str, tool, arguments: dict
    ) -> tuple[str, str]:
        last_error = ""
        for attempt in range(self._max_retries + 1):
            try:
                result = tool.handler(**arguments)
            except Exception as exc:  # noqa: BLE001 - tool code is untrusted
                last_error = f"{type(exc).__name__}: {exc}"
                final = attempt == self._max_retries
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
