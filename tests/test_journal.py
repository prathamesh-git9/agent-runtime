from __future__ import annotations

import pytest

from agent_runtime.errors import ConcurrentAppend
from agent_runtime.events import Event, EventType, RunStatus
from agent_runtime.journal import Journal


@pytest.fixture
def journal(tmp_path):
    j = Journal(tmp_path / "journal.db")
    yield j
    j.close()


def test_sequence_numbers_increment_from_zero_per_run(journal):
    first = journal.append(Event(run_id="run_a", type=EventType.RUN_STARTED))
    second = journal.append(Event(run_id="run_a", type=EventType.LLM_RESPONDED))
    other = journal.append(Event(run_id="run_b", type=EventType.RUN_STARTED))

    assert first.seq == 0
    assert second.seq == 1
    assert other.seq == 0
    assert journal.next_seq("run_a") == 2


def test_events_are_isolated_by_run_id(journal):
    journal.append(Event(run_id="run_a", type=EventType.RUN_STARTED))
    journal.append(Event(run_id="run_b", type=EventType.RUN_COMPLETED))

    events = journal.read("run_a")

    assert len(events) == 1
    assert events[0].run_id == "run_a"
    assert events[0].type is EventType.RUN_STARTED


def test_read_after_seq_returns_only_later_events(journal):
    journal.append(Event(run_id="run_a", type=EventType.RUN_STARTED))
    journal.append(Event(run_id="run_a", type=EventType.TOOL_REQUESTED))
    journal.append(Event(run_id="run_a", type=EventType.TOOL_SUCCEEDED))

    later = journal.read("run_a", after_seq=0)

    assert [event.seq for event in later] == [1, 2]


def test_appending_duplicate_seq_raises_concurrent_append(journal):
    event = Event(run_id="run_a", type=EventType.RUN_STARTED, seq=0)
    journal.append(event)

    with pytest.raises(ConcurrentAppend):
        journal.append(event)


def test_run_metadata_round_trips_and_filters_by_status(journal):
    journal.create_run("run_a", "first goal", 10.0)
    journal.create_run("run_b", "second goal", 20.0)
    journal.set_status("run_a", RunStatus.COMPLETED, 30.0)

    assert journal.get_run("run_a") == {
        "run_id": "run_a",
        "goal": "first goal",
        "status": "completed",
        "created_at": 10.0,
        "updated_at": 30.0,
    }
    assert journal.get_run("missing") is None
    assert [run["run_id"] for run in journal.list_runs()] == ["run_b", "run_a"]
    assert [run["run_id"] for run in journal.list_runs(RunStatus.COMPLETED)] == [
        "run_a"
    ]


def test_find_last_returns_most_recent_event_of_a_type(journal):
    journal.append(
        Event(
            run_id="run_a",
            type=EventType.TOOL_FAILED,
            payload={"error": "first"},
        )
    )
    journal.append(Event(run_id="run_a", type=EventType.TOOL_SUCCEEDED))
    journal.append(
        Event(
            run_id="run_a",
            type=EventType.TOOL_FAILED,
            payload={"error": "second"},
        )
    )

    event = journal.find_last("run_a", EventType.TOOL_FAILED)

    assert event is not None
    assert event.payload["error"] == "second"
    assert journal.find_last("run_a", EventType.RUN_FAILED) is None
