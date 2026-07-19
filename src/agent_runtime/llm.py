"""Planner interface: the component that decides the agent's next move.

Kept deliberately narrow — one method returning one decision — because the
runtime's durability guarantees depend on treating the planner as a
non-deterministic oracle whose every answer is journalled. A wider interface
would leak un-journalled decisions into the loop and break replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class CallTool:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "call_tool", "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class Finish:
    output: str

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "finish", "output": self.output}


Decision = CallTool | Finish


def decision_from_dict(d: dict[str, Any]) -> Decision:
    if d["kind"] == "finish":
        return Finish(output=d["output"])
    return CallTool(name=d["name"], arguments=d.get("arguments", {}))


@runtime_checkable
class Planner(Protocol):
    def decide(
        self, goal: str, transcript: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Decision: ...


class ScriptedPlanner:
    """A planner that replays a fixed list of decisions.

    Every test in this repo runs against this. Durable execution is fundamentally
    about *what happens between* the model calls, so testing it against a real
    model would add cost and flakiness while proving nothing extra — and would
    make crash-recovery tests impossible to write deterministically.

    It indexes off `len(transcript)`, not off its own call count, so that it is a
    pure function of the conversation exactly like a real planner. Counting its
    own invocations would desync on resume: replay serves recorded decisions
    without consulting the planner, so a self-counting fake would re-issue a
    decision the run had already executed.
    """

    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = list(decisions)
        # Invocation count, for asserting a resume did not re-bill the planner.
        self.calls = 0

    def decide(self, goal, transcript, tools) -> Decision:
        self.calls += 1
        index = len(transcript)
        if index >= len(self._decisions):
            return Finish(output="script exhausted")
        return self._decisions[index]
