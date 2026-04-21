"""SQLite-backed task store for async API jobs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset_agent.domain.ports import TaskRepositoryPort, TaskStatus


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteTaskRepository(TaskRepositoryPort):
    def __init__(self, db_path: Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    result_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create_task(self, payload: dict[str, Any]) -> str:
        tid = str(uuid.uuid4())
        now = _utc_now()
        with self._connect() as c:
            c.execute(
                """INSERT INTO tasks (id, status, payload, result_path, error, created_at, updated_at)
                   VALUES (?, ?, ?, NULL, NULL, ?, ?)""",
                (tid, TaskStatus.PENDING, json.dumps(payload), now, now),
            )
        return tid

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as c:
            row = c.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return None
        return _row_to_dict(row)

    def list_tasks(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as c:
            rows = c.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        result_path: str | None = None,
        error: str | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if result_path is not None:
            fields.append("result_path = ?")
            values.append(result_path)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        fields.append("updated_at = ?")
        values.append(_utc_now())
        values.append(task_id)
        sql = f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?"
        with self._connect() as c:
            c.execute(sql, values)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["payload"] = json.loads(d["payload"])
    except (json.JSONDecodeError, TypeError):
        pass
    return d
