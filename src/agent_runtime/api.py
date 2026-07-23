"""HTTP control plane.

The API is intentionally thin: it owns no run state of its own. Every endpoint
reads or appends to the journal, which means an operator can kill this process
mid-run, start it somewhere else, and drive the same runs to completion.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_runtime.errors import (
    ConcurrentAppend,
    MaxStepsExceeded,
    OutcomeResolutionError,
    ReplayDivergence,
    RunNotFound,
)
from agent_runtime.events import RunStatus
from agent_runtime.journal import Journal
from agent_runtime.runtime import AgentRuntime
from agent_runtime.tools import ToolRegistry


class StartRun(BaseModel):
    goal: str = Field(min_length=1)


class ApprovalDecision(BaseModel):
    call_id: str
    allowed: bool
    reason: str = ""


class ToolOutcomeDecision(BaseModel):
    call_id: str
    succeeded: bool
    actor: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(min_length=1)
    result: str = ""
    error: str = ""


def _state_payload(state) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "status": str(state.status),
        "output": state.output,
        "error": state.error,
        "pending_approval": state.pending_approval,
        "pending_recovery": state.pending_recovery,
        "steps": state.steps,
        "replayed": state.replayed,
        "transcript": state.transcript,
    }


def create_app(journal: Journal, planner, registry: ToolRegistry, **kwargs) -> FastAPI:
    app = FastAPI(title="agent-runtime", version="0.1.0")
    runtime = AgentRuntime(journal, planner, registry, **kwargs)

    @app.exception_handler(RunNotFound)
    async def _not_found(_, exc: RunNotFound):
        raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tools")
    async def tools() -> list[dict]:
        return registry.specs()

    @app.post("/runs", status_code=201)
    async def start_run(body: StartRun) -> dict:
        return _state_payload(runtime.start(body.goal))

    @app.get("/runs")
    async def list_runs(status: str | None = None) -> list[dict]:
        return journal.list_runs(RunStatus(status) if status else None)

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        run = journal.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
        return run

    @app.get("/runs/{run_id}/events")
    async def get_events(run_id: str, after_seq: int = -1) -> list[dict]:
        if journal.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
        return [e.as_dict() for e in journal.read(run_id, after_seq=after_seq)]

    @app.post("/runs/{run_id}/advance")
    async def advance(run_id: str) -> dict:
        try:
            return _state_payload(runtime.advance(run_id))
        except RunNotFound:
            raise HTTPException(
                status_code=404, detail=f"no such run: {run_id}"
            ) from None
        except ReplayDivergence as exc:
            # 409: the stored run and the deployed code are incompatible. This
            # needs a human, not a retry.
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except MaxStepsExceeded as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @app.post("/runs/{run_id}/approvals")
    async def decide(run_id: str, body: ApprovalDecision) -> dict:
        if journal.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
        try:
            state = runtime.approve(
                run_id, body.call_id, allowed=body.allowed, reason=body.reason
            )
        except MaxStepsExceeded as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return _state_payload(state)

    @app.post("/runs/{run_id}/tool-outcomes")
    async def resolve_tool_outcome(run_id: str, body: ToolOutcomeDecision) -> dict:
        try:
            state = runtime.resolve_tool_outcome(
                run_id,
                body.call_id,
                succeeded=body.succeeded,
                actor=body.actor,
                evidence=body.evidence,
                result=body.result,
                error=body.error,
            )
        except RunNotFound:
            raise HTTPException(
                status_code=404, detail=f"no such run: {run_id}"
            ) from None
        except (OutcomeResolutionError, ConcurrentAppend) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return _state_payload(state)

    return app
