from __future__ import annotations

from fastapi.testclient import TestClient

from agent_runtime import Approval, CallTool, Finish, Journal, ScriptedPlanner, Tool
from agent_runtime.api import create_app
from agent_runtime.tools import ToolRegistry


def test_healthz():
    client = TestClient(create_app(Journal(), ScriptedPlanner([]), ToolRegistry()))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tools_lists_registered_tools():
    registry = ToolRegistry()
    registry.register(Tool(name="lookup", description="Lookup.", handler=lambda: "ok"))
    client = TestClient(create_app(Journal(), ScriptedPlanner([]), registry))

    response = client.get("/tools")

    assert response.status_code == 200
    assert response.json()[0]["name"] == "lookup"


def test_post_runs_starts_and_completes_finish_only_script():
    journal = Journal()
    client = TestClient(
        create_app(journal, ScriptedPlanner([Finish(output="done")]), ToolRegistry())
    )

    response = client.post("/runs", json={"goal": "finish"})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["output"] == "done"


def test_get_run_returns_run_and_unknown_run_returns_404():
    journal = Journal()
    client = TestClient(
        create_app(journal, ScriptedPlanner([Finish(output="done")]), ToolRegistry())
    )
    started = client.post("/runs", json={"goal": "finish"}).json()

    response = client.get(f"/runs/{started['run_id']}")
    missing = client.get("/runs/missing")

    assert response.status_code == 200
    assert response.json()["run_id"] == started["run_id"]
    assert missing.status_code == 404


def test_get_run_events_returns_event_list():
    journal = Journal()
    client = TestClient(
        create_app(journal, ScriptedPlanner([Finish(output="done")]), ToolRegistry())
    )
    started = client.post("/runs", json={"goal": "finish"}).json()

    response = client.get(f"/runs/{started['run_id']}/events")

    assert response.status_code == 200
    assert [event["type"] for event in response.json()] == [
        "run_started",
        "llm_responded",
        "run_completed",
    ]


def test_get_runs_lists_runs():
    journal = Journal()
    client = TestClient(
        create_app(journal, ScriptedPlanner([Finish(output="done")]), ToolRegistry())
    )
    started = client.post("/runs", json={"goal": "finish"}).json()

    response = client.get("/runs")

    assert response.status_code == 200
    assert response.json()[0]["run_id"] == started["run_id"]


def test_approval_gated_run_can_be_approved():
    journal = Journal()
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="refund",
            description="Issue refund.",
            handler=lambda: "refunded",
            approval=Approval.REQUIRED,
            idempotent=False,
        )
    )
    client = TestClient(
        create_app(
            journal,
            ScriptedPlanner([CallTool("refund"), Finish(output="done")]),
            registry,
        )
    )

    started = client.post("/runs", json={"goal": "refund customer"})

    assert started.status_code == 201
    body = started.json()
    assert body["status"] == "awaiting_approval"
    approved = client.post(
        f"/runs/{body['run_id']}/approvals",
        json={"call_id": body["pending_approval"]["call_id"], "allowed": True},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "completed"


def test_approval_for_unknown_run_returns_404():
    client = TestClient(create_app(Journal(), ScriptedPlanner([]), ToolRegistry()))

    response = client.post(
        "/runs/missing/approvals",
        json={"call_id": "call_missing", "allowed": True},
    )

    assert response.status_code == 404
