"""Append-only event journal backed by SQLite.

SQLite is a deliberate choice over an in-memory store: durability is the entire
point of this project, and a runtime whose journal dies with the process would
be a simulation of durability rather than the real thing. Swapping in Postgres
is a matter of replacing this class — nothing above it knows the storage engine.

Concurrency safety rests on a UNIQUE(run_id, seq) constraint. Two workers that
both believe they own a run cannot both append at the same sequence number; the
loser gets an IntegrityError, which surfaces as ConcurrentAppend rather than a
silently interleaved journal.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from agent_runtime.errors import ConcurrentAppend
from agent_runtime.events import Event, EventType, RunStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    run_id     TEXT    NOT NULL,
    seq        INTEGER NOT NULL,
    type       TEXT    NOT NULL,
    payload    TEXT    NOT NULL,
    created_at REAL    NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    goal       TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class Journal:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        # check_same_thread=False + an explicit lock: FastAPI serves requests on
        # a threadpool, and a connection pinned to one thread would fail there.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    # -- runs ---------------------------------------------------------------

    def create_run(self, run_id: str, goal: str, now: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                (run_id, goal, str(RunStatus.RUNNING), now, now),
            )
            self._conn.commit()

    def set_status(self, run_id: str, status: RunStatus, now: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (str(status), now, run_id),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        cur = self._conn.execute(
            "SELECT run_id, goal, status, created_at, updated_at "
            "FROM runs WHERE run_id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "goal": row[1],
            "status": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }

    def list_runs(self, status: RunStatus | None = None) -> list[dict]:
        sql = "SELECT run_id, goal, status, created_at, updated_at FROM runs"
        params: tuple = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (str(status),)
        sql += " ORDER BY created_at DESC"
        return [
            {
                "run_id": r[0],
                "goal": r[1],
                "status": r[2],
                "created_at": r[3],
                "updated_at": r[4],
            }
            for r in self._conn.execute(sql, params).fetchall()
        ]

    # -- events -------------------------------------------------------------

    def next_seq(self, run_id: str) -> int:
        cur = self._conn.execute(
            "SELECT COALESCE(MAX(seq), -1) FROM events WHERE run_id = ?", (run_id,)
        )
        return cur.fetchone()[0] + 1

    def append(self, event: Event) -> Event:
        with self._lock:
            seq = event.seq
            if seq < 0:
                cur = self._conn.execute(
                    "SELECT COALESCE(MAX(seq), -1) FROM events WHERE run_id = ?",
                    (event.run_id,),
                )
                seq = cur.fetchone()[0] + 1
            stamped = Event(
                run_id=event.run_id,
                type=event.type,
                seq=seq,
                payload=event.payload,
                created_at=event.created_at,
            )
            try:
                self._conn.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?)", stamped.to_row()
                )
            except sqlite3.IntegrityError as exc:
                raise ConcurrentAppend(
                    f"seq {seq} already exists for run {event.run_id}; "
                    "another worker is advancing this run"
                ) from exc
            self._conn.commit()
            return stamped

    def read(self, run_id: str, *, after_seq: int = -1) -> list[Event]:
        rows = self._conn.execute(
            "SELECT run_id, seq, type, payload, created_at FROM events "
            "WHERE run_id = ? AND seq > ? ORDER BY seq",
            (run_id, after_seq),
        ).fetchall()
        return [Event.from_row(r) for r in rows]

    def find_last(self, run_id: str, etype: EventType) -> Event | None:
        rows = self._conn.execute(
            "SELECT run_id, seq, type, payload, created_at FROM events "
            "WHERE run_id = ? AND type = ? ORDER BY seq DESC LIMIT 1",
            (run_id, str(etype)),
        ).fetchall()
        return Event.from_row(rows[0]) if rows else None
