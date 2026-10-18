import sqlite3
from pathlib import Path

import pytest

from retail_agent.reports import ReportError, ReportStore
from retail_agent.agent import parse_report_command


def save(store: ReportStore, request: str, owner: str, conversation: str, body: str) -> str:
    return store.save_report(
        request_id=request,
        owner=owner,
        conversation_id=conversation,
        title=f"Report {request}",
        body=body,
        scope=["Calvin Klein"],
    )


def test_deletion_requires_owned_exact_confirmation_and_is_idempotent(tmp_path: Path) -> None:
    store = ReportStore(tmp_path / "reports.sqlite3")
    own = save(store, "r1", "alice", "conversation", "Client X result")
    save(store, "r2", "bob", "conversation", "Client X result")

    pending = store.prepare_delete(
        owner="alice", conversation_id="conversation", contains="Client X"
    )
    assert pending is not None
    assert [report.report_id for report in pending.reports] == [own]

    with pytest.raises(ReportError):
        store.confirm_delete(
            owner="bob",
            conversation_id="conversation",
            action_id=pending.action_id,
            token=pending.token,
        )
    with pytest.raises(ReportError):
        store.confirm_delete(
            owner="alice",
            conversation_id="conversation",
            action_id=pending.action_id,
            token="wrong",
        )
    assert len(store.list_reports("alice")) == 1

    receipt = store.confirm_delete(
        owner="alice",
        conversation_id="conversation",
        action_id=pending.action_id,
        token=pending.token,
    )
    replay = store.confirm_delete(
        owner="alice",
        conversation_id="conversation",
        action_id=pending.action_id,
        token=pending.token,
    )
    assert receipt == replay
    assert receipt.report_ids == [own]
    assert store.list_reports("alice") == []
    assert len(store.list_reports("bob")) == 1
    store.close()


def test_conversation_delete_does_not_include_other_conversations(tmp_path: Path) -> None:
    store = ReportStore(tmp_path / "reports.sqlite3")
    report = save(store, "r1", "alice", "current", "one")
    save(store, "r2", "alice", "other", "two")
    pending = store.prepare_delete(
        owner="alice", conversation_id="current", current_conversation_only=True
    )
    assert pending is not None
    assert [item.report_id for item in pending.reports] == [report]
    assert store.cancel_delete(
        owner="alice", conversation_id="current", action_id=pending.action_id
    )
    assert len(store.list_reports("alice")) == 2
    store.close()


def test_plain_yes_is_never_a_deletion_confirmation() -> None:
    assert parse_report_command("yes") is None
    assert parse_report_command("confirm") is None


def test_confirmation_is_bound_to_the_preparing_conversation(tmp_path: Path) -> None:
    store = ReportStore(tmp_path / "reports.sqlite3")
    save(store, "r1", "alice", "original", "Client X result")
    pending = store.prepare_delete(
        owner="alice", conversation_id="original", contains="Client X"
    )
    assert pending is not None

    with pytest.raises(ReportError):
        store.confirm_delete(
            owner="alice",
            conversation_id="other",
            action_id=pending.action_id,
            token=pending.token,
        )
    assert not store.cancel_delete(
        owner="alice", conversation_id="other", action_id=pending.action_id
    )
    assert len(store.list_reports("alice")) == 1
    store.close()


def test_legacy_unused_digest_column_is_removed(tmp_path: Path) -> None:
    database = tmp_path / "reports.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        """CREATE TABLE pending_deletions (
            action_id TEXT PRIMARY KEY, owner TEXT NOT NULL,
            conversation_id TEXT NOT NULL, token_hash TEXT NOT NULL,
            targets_json TEXT NOT NULL, digest TEXT NOT NULL,
            expires_at TEXT NOT NULL, status TEXT NOT NULL, receipt_json TEXT
        )"""
    )
    connection.commit()
    connection.close()

    store = ReportStore(database)
    columns = {
        row[1]
        for row in store.connection.execute(
            "PRAGMA table_info(pending_deletions)"
        ).fetchall()
    }
    assert "digest" not in columns
    store.close()
