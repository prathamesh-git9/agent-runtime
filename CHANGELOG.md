# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-21

### Added

- Event-sourced run journal backed by SQLite, with per-run ordered events and
  run metadata.
- Durable agent runtime that folds the journal into run state before advancing.
- Deterministic replay of recorded planner decisions, tool requests, tool
  results, approval decisions, and terminal run outcomes.
- Crash recovery semantics that replay committed tool outcomes instead of
  re-executing effectful tool handlers.
- Tool registry with tool metadata, idempotency markers, and approval policy.
- Human approval gates that suspend runs, persist decisions, and resume after
  approval or denial.
- Replay divergence detection when journaled tool requests and replayed planner
  decisions disagree.
- Tool retry recording with final failure surfaced as tool feedback to the
  agent.
- FastAPI control plane for health checks, tools, runs, journal events,
  advancing runs, and approval decisions.
- Deterministic `ScriptedPlanner` and tests covering journal behavior, replay,
  approvals, retries, tools, and the HTTP API.
