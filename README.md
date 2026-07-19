# agent-runtime

Agent runs are long and flaky. A crash in the middle of a run can lose all
state, re-spend tokens, and re-fire side effects like refunds, emails, or ticket
updates.

`agent-runtime` applies Temporal-style durable execution to tool-calling agent
loops. Every non-deterministic outcome, including planner decisions, tool
results, and human approvals, is journalled before it is acted on. On resume,
those outcomes are replayed rather than produced again.

```text
          +------------------+
          |   run journal    |
          +--------+---------+
                   ^
                   | append before acting
                   |
+------+    +------+------+    +-------+    +-----------+
| goal | -> | fold events | -> | plan  | -> | tool/gate |
+------+    +------+------+    +---+---+    +-----+-----+
                   ^               |              |
                   |               v              v
                   +--------- replayed outcomes <-+
```

## Why It Exists

Most agent loops keep critical state in memory. That works until the process is
killed, a deploy restarts the worker, a network call times out, or an operator
needs to resume a run after an approval. Reconstructing state by re-running the
agent is unsafe because planner calls are non-deterministic and tools can affect
the outside world.

This project makes the event log the source of truth. The runtime can stop after
any appended event and later continue from the same journal.

## How It Works

The runtime stores a sequence of events per run in SQLite. Starting or advancing
a run folds those events into a `RunState`, then continues the loop from the
first missing outcome.

Planner decisions are recorded as `LLM_RESPONDED` before the chosen action is
used. Tool calls are recorded as `TOOL_REQUESTED`, then their success or failure
is recorded. Human decisions are recorded as `APPROVAL_DECIDED`.

Tools have an `idempotent` flag. Idempotent tools are safe to call repeatedly in
principle. Effectful tools, such as refunds and emails, should set
`idempotent=False`; their recorded result is replayed on resume rather than
running the handler again.

## Approval Gates

Tools can declare `approval=Approval.REQUIRED`. When the planner requests such a
tool, the runtime records an approval request, sets the run status to
`awaiting_approval`, and returns without executing the tool. A later approval
records the human decision and advances the run. Denial becomes tool feedback in
the transcript rather than a runtime crash.

## Architecture

| Module | Responsibility |
| --- | --- |
| `agent_runtime.events` | Event types, run statuses, and event serialization. |
| `agent_runtime.journal` | SQLite-backed append-only journal and run metadata. |
| `agent_runtime.llm` | Narrow planner protocol and scripted test planner. |
| `agent_runtime.tools` | Tool registry, tool specs, and approval policy. |
| `agent_runtime.runtime` | Durable event-sourced agent loop. |
| `agent_runtime.api` | FastAPI control plane over the runtime and journal. |

## Quickstart

```bash
pip install -e .[dev]
python examples/support_triage.py
python -m uvicorn agent_runtime.__main__:app --host 0.0.0.0 --port 8000
```

The module entrypoint starts a demo FastAPI app with an in-memory journal,
demo tools, and a scripted planner so the Docker image has a concrete app
factory target.

## HTTP API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/healthz` | Health check. |
| `GET` | `/tools` | List registered tool specs. |
| `POST` | `/runs` | Start a run from a JSON body with `goal`. |
| `GET` | `/runs` | List runs, optionally filtered by status. |
| `GET` | `/runs/{id}` | Fetch run metadata. |
| `GET` | `/runs/{id}/events` | Fetch journal events for a run. |
| `POST` | `/runs/{id}/advance` | Drive an existing run forward. |
| `POST` | `/runs/{id}/approvals` | Record an approval decision and resume. |

## Testing

```bash
ruff check .
python -m pytest -q
```

The tests use `ScriptedPlanner` instead of a real model so retry, approval, and
replay behavior stays deterministic.

## Design Notes

SQLite is used because durability is the point. An in-memory journal can test
the interface, but a durable local database exercises the real failure model and
keeps deployment simple. The journal boundary is narrow enough that replacing it
with another store does not change the runtime loop.

The planner interface is intentionally small: one method receives the goal,
transcript, and tool specs, then returns either `CallTool` or `Finish`. A wider
planner surface would make it easier for un-journalled model decisions to leak
into execution.

`ReplayDivergence` protects suspended runs from incompatible code changes. If
the journal says a run called one tool but the current planner path now expects
another, the runtime refuses to continue rather than applying an old result to a
new action.
