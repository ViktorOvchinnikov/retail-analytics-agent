from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import DeletionReceipt, PendingDeletion, ReportSummary


class ReportError(RuntimeError):
    pass


class ReportStore:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                owner TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                scope_json TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE INDEX IF NOT EXISTS reports_owner_active
                ON reports(owner, deleted_at);
            CREATE TABLE IF NOT EXISTS pending_deletions (
                action_id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                targets_json TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL,
                receipt_json TEXT
            );
            CREATE TABLE IF NOT EXISTS report_audit (
                event_id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                action_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                report_ids_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        columns = {
            row[1]
            for row in self.connection.execute(
                "PRAGMA table_info(pending_deletions)"
            ).fetchall()
        }
        if "digest" in columns:
            self.connection.execute(
                "ALTER TABLE pending_deletions DROP COLUMN digest"
            )
        self.connection.commit()

    def save_report(
        self,
        *,
        request_id: str,
        owner: str,
        conversation_id: str,
        title: str,
        body: str,
        scope: list[str] | None,
    ) -> str:
        report_id = str(uuid.uuid4())
        now = _now().isoformat()
        with self.connection:
            self.connection.execute(
                """INSERT OR IGNORE INTO reports
                   (report_id, request_id, owner, conversation_id, title, body,
                    scope_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report_id,
                    request_id,
                    owner,
                    conversation_id,
                    title,
                    body,
                    json.dumps(scope),
                    now,
                ),
            )
            row = self.connection.execute(
                "SELECT report_id FROM reports WHERE request_id = ? AND owner = ?",
                (request_id, owner),
            ).fetchone()
        if row is None:
            raise ReportError("report could not be saved")
        return str(row["report_id"])

    def list_reports(self, owner: str) -> list[ReportSummary]:
        rows = self.connection.execute(
            """SELECT report_id, title, created_at FROM reports
               WHERE owner = ? AND deleted_at IS NULL
               ORDER BY created_at DESC""",
            (owner,),
        ).fetchall()
        return [ReportSummary(**dict(row)) for row in rows]

    def prepare_delete(
        self,
        *,
        owner: str,
        conversation_id: str,
        contains: str | None = None,
        current_conversation_only: bool = False,
        ttl_minutes: int = 10,
    ) -> PendingDeletion | None:
        clauses = ["owner = ?", "deleted_at IS NULL"]
        values: list[object] = [owner]
        if current_conversation_only:
            clauses.append("conversation_id = ?")
            values.append(conversation_id)
        if contains is not None:
            phrase = contains.strip()
            if not phrase:
                raise ReportError("deletion phrase cannot be empty")
            clauses.append("instr(lower(title || char(10) || body), lower(?)) > 0")
            values.append(phrase)

        rows = self.connection.execute(
            f"""SELECT report_id, title, created_at, version FROM reports
                WHERE {' AND '.join(clauses)} ORDER BY created_at""",
            values,
        ).fetchall()
        if not rows:
            return None

        targets = [
            {"report_id": row["report_id"], "version": row["version"]}
            for row in rows
        ]
        target_json = json.dumps(targets, sort_keys=True, separators=(",", ":"))
        action_id = str(uuid.uuid4())
        token = secrets.token_urlsafe(18)
        expires_at = _now() + timedelta(minutes=ttl_minutes)
        with self.connection:
            self.connection.execute(
                """INSERT INTO pending_deletions
                   (action_id, owner, conversation_id, token_hash, targets_json,
                    expires_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending')""",
                (
                    action_id,
                    owner,
                    conversation_id,
                    _token_hash(token),
                    target_json,
                    expires_at.isoformat(),
                ),
            )
        return PendingDeletion(
            action_id=action_id,
            token=token,
            reports=[
                ReportSummary(
                    report_id=row["report_id"],
                    title=row["title"],
                    created_at=row["created_at"],
                )
                for row in rows
            ],
            expires_at=expires_at.isoformat(),
        )

    def confirm_delete(
        self,
        *,
        owner: str,
        conversation_id: str,
        action_id: str,
        token: str,
    ) -> DeletionReceipt:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            action = self.connection.execute(
                """SELECT * FROM pending_deletions
                   WHERE action_id = ? AND owner = ? AND conversation_id = ?""",
                (action_id, owner, conversation_id),
            ).fetchone()
            if action is None or not secrets.compare_digest(
                action["token_hash"], _token_hash(token)
            ):
                raise ReportError("invalid deletion action or confirmation token")
            if action["status"] == "confirmed" and action["receipt_json"]:
                self.connection.rollback()
                return DeletionReceipt.model_validate_json(action["receipt_json"])
            if action["status"] != "pending":
                raise ReportError(f"deletion action is {action['status']}")
            if datetime.fromisoformat(action["expires_at"]) <= _now():
                self.connection.execute(
                    "UPDATE pending_deletions SET status = 'expired' WHERE action_id = ?",
                    (action_id,),
                )
                self.connection.commit()
                raise ReportError("deletion action expired; request a new preview")

            targets = json.loads(action["targets_json"])
            report_ids: list[str] = []
            for target in targets:
                row = self.connection.execute(
                    """SELECT version FROM reports
                       WHERE report_id = ? AND owner = ? AND deleted_at IS NULL""",
                    (target["report_id"], owner),
                ).fetchone()
                if row is None or row["version"] != target["version"]:
                    raise ReportError("report set changed; request a new preview")
                report_ids.append(target["report_id"])

            committed_at = _now().isoformat()
            placeholders = ",".join("?" for _ in report_ids)
            self.connection.execute(
                f"""UPDATE reports SET deleted_at = ?, version = version + 1
                    WHERE owner = ? AND report_id IN ({placeholders})""",
                [committed_at, owner, *report_ids],
            )
            receipt = DeletionReceipt(
                action_id=action_id,
                deleted_count=len(report_ids),
                report_ids=report_ids,
                committed_at=committed_at,
            )
            self.connection.execute(
                """UPDATE pending_deletions
                   SET status = 'confirmed', receipt_json = ? WHERE action_id = ?""",
                (receipt.model_dump_json(), action_id),
            )
            self.connection.execute(
                """INSERT INTO report_audit
                   (event_id, owner, action_id, event_type, report_ids_json, created_at)
                   VALUES (?, ?, ?, 'delete', ?, ?)""",
                (
                    str(uuid.uuid4()),
                    owner,
                    action_id,
                    json.dumps(report_ids),
                    committed_at,
                ),
            )
            self.connection.commit()
            return receipt
        except Exception:
            if self.connection.in_transaction:
                self.connection.rollback()
            raise

    def cancel_delete(
        self, *, owner: str, conversation_id: str, action_id: str
    ) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                """UPDATE pending_deletions SET status = 'cancelled'
                   WHERE action_id = ? AND owner = ? AND conversation_id = ?
                   AND status = 'pending'""",
                (action_id, owner, conversation_id),
            )
        return cursor.rowcount == 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
