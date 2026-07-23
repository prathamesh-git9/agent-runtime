"""The event vocabulary of a run.

Everything the runtime does that is *not* deterministically recomputable gets
an event. That is the whole contract: if a decision came from a model, a clock,
or the outside world, it is journalled, and on replay the journal answers
instead of the outside world.

Events are append-only and never mutated. A run's state is always the fold of
its events, which is what makes crash recovery a replay rather than a guess.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    LLM_RESPONDED = "llm_responded"
    TOOL_REQUESTED = "tool_requested"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    TOOL_EXECUTION_STARTED = "tool_execution_started"
    TOOL_OUTCOME_UNKNOWN = "tool_outcome_unknown"
    TOOL_SUCCEEDED = "tool_succeeded"
    TOOL_FAILED = "tool_failed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class RunStatus(StrEnum):
    RUNNING = "running"
    # Suspended is a first-class resting state, not an error: a run parked on an
    # approval gate may sit here for days across process restarts.
    AWAITING_APPROVAL = "awaiting_approval"
    # A non-idempotent handler started but no durable outcome exists. Retrying
    # could duplicate an external effect, so only evidence-backed resolution
    # may advance the run.
    AWAITING_RECOVERY = "awaiting_recovery"
    COMPLETED = "completed"
    FAILED = "failed"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class Event:
    run_id: str
    type: EventType
    # Monotonic per run, assigned by the journal on append. -1 means unassigned.
    seq: int = -1
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_row(self) -> tuple[str, int, str, str, float]:
        return (
            self.run_id,
            self.seq,
            str(self.type),
            json.dumps(self.payload, sort_keys=True),
            self.created_at,
        )

    @staticmethod
    def from_row(row: tuple) -> Event:
        run_id, seq, etype, payload, created_at = row
        return Event(
            run_id=run_id,
            seq=seq,
            type=EventType(etype),
            payload=json.loads(payload),
            created_at=created_at,
        )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = str(self.type)
        return d
