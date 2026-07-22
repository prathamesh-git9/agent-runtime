# Security Policy

## Threat Model

`agent-runtime` is designed for durable, resumable execution of tool-calling AI
agents. The primary security and correctness guarantee is exactly-once side
effects across crashes: once an effectful tool call has committed and its result
is in the journal, replaying the run must never execute that side effect again.

The runtime treats the journal as the source of truth. Planner decisions, tool
requests, tool outcomes, and human approval decisions are appended before later
steps depend on them. On resume, recorded outcomes are consumed from the journal
instead of being produced again. Replaying a committed `send_email`,
`charge_card`, `issue_refund`, or similar effectful tool would be a critical
bug, because reconstructing state must not mutate the outside world.

Important assets and invariants:

- The per-run event stream is append-only and ordered by sequence number.
- A tool result recorded as `TOOL_SUCCEEDED` or final `TOOL_FAILED` is terminal
  for that call during replay.
- Human approval events are durable decisions, not in-memory flags.
- `ReplayDivergence` must stop execution when stored events and live code no
  longer agree.
- Concurrent workers must not silently interleave writes to the same run.

## Human Approval Gates

Tools that can perform dangerous, expensive, irreversible, privileged, or
externally visible work should require human approval. Examples include
refunds, payments, account changes, emails to customers, ticket updates, and
administrative actions.

A required approval must suspend the run before the tool handler executes. The
human decision is then recorded in the journal. Denied approvals become tool
feedback to the agent rather than a runtime crash, allowing the agent to choose
a safer path without losing the audit trail.

## Journal Integrity

The SQLite journal is append-only at the runtime boundary. Event sequence
numbers are monotonic per run, and the database schema enforces
`PRIMARY KEY (run_id, seq)` so competing writers cannot both append the same
logical step.

Any change that mutates, deletes, reorders, or rewrites committed events can
break replay safety. Schema migrations and alternate journal implementations
must preserve append-only semantics, stable event ordering, and the ability to
fold a complete run state from the event stream alone.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately using a GitHub Security
Advisory for this repository. Do not open a public issue for security reports.

Include:

- A description of the issue and affected invariant.
- Steps to reproduce, if available.
- Whether the issue could re-fire a committed side effect, bypass approval, or
  corrupt journal ordering.
- Any relevant logs, journal events, or proof-of-concept code.

We will acknowledge the report, investigate privately, and coordinate a fix and
disclosure timeline before public discussion.
