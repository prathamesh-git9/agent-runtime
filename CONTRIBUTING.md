# Contributing

## Development Setup

This project targets Python 3.11 or newer.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .[dev]
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
```

On POSIX shells, use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.

## Adding a Tool

Tools live behind `ToolRegistry` in `agent_runtime.tools`. Register a `Tool`
directly or use `@registry.tool(...)`.

When adding a tool:

- Give it a stable name; journal replay matches recorded tool requests by name.
- Set `approval=Approval.REQUIRED` for dangerous, privileged, expensive,
  irreversible, or externally visible actions.
- Set `idempotent=False` for side-effecting tools such as refunds, emails,
  payments, account updates, and ticket writes.
- Keep handlers focused on one external action so approval and replay behavior
  are easy to reason about.
- Add tests that prove resume behavior does not re-run effectful work.

## Adding a Journal Event Type

Event types are defined in `agent_runtime.events.EventType` and folded in
`agent_runtime.runtime._fold`.

When adding an event:

- Add the enum value and serialization payload shape.
- Update `_fold` so replay can reconstruct state from the journal alone.
- Append the event before any later action depends on its outcome.
- Add tests for normal execution, resume from disk, and incompatible replay if
  the event affects tool-call ordering or outcomes.
- Preserve append-only semantics; do not mutate existing events to model a new
  state transition.

## House Style

- Use `from __future__ import annotations` in Python modules.
- Prefer WHY-focused docstrings and comments over restating the code.
- Keep changes scoped; avoid rewriting working code to make unrelated style
  improvements.
- Follow the Ruff configuration in `pyproject.toml`; line length is 90.
- Tests should use deterministic planners and local tools, not live model calls.
