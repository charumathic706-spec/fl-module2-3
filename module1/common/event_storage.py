from __future__ import annotations

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, List

try:
    from module1.common.event_schema import RoundEventRecord
except ImportError:
    from common.event_schema import RoundEventRecord


class EventStorageBackend(ABC):
    @abstractmethod
    def append_round_event(self, event: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def read_all_events(self) -> List[Dict[str, Any]]:
        pass


class JsonlEventStorage(EventStorageBackend):
    def __init__(self, file_path: str):
        self.file_path = file_path
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    def append_round_event(self, event: Dict[str, Any]) -> None:
        record = RoundEventRecord.from_payload(event).to_dict()
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")

    def read_all_events(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            return []
        rows: List[Dict[str, Any]] = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(RoundEventRecord.from_payload(json.loads(line)).to_dict())
        return rows


class SqliteEventStorage(EventStorageBackend):
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS round_events (
                    round INTEGER PRIMARY KEY,
                    schema_name TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def append_round_event(self, event: Dict[str, Any]) -> None:
        record = RoundEventRecord.from_payload(event).to_dict()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO round_events(round, schema_name, schema_version, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    int(record["round"]),
                    str(record["schema_name"]),
                    int(record["schema_version"]),
                    json.dumps(record, sort_keys=True),
                ),
            )
            conn.commit()

    def read_all_events(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM round_events ORDER BY round ASC"
            ).fetchall()
        return [RoundEventRecord.from_payload(json.loads(r[0])).to_dict() for r in rows]


def create_event_storage(backend: str, log_dir: str) -> EventStorageBackend:
    b = str(backend or "jsonl").strip().lower()
    if b == "jsonl":
        return JsonlEventStorage(os.path.join(log_dir, "round_events.jsonl"))
    if b == "sqlite":
        return SqliteEventStorage(os.path.join(log_dir, "round_events.db"))
    raise ValueError(f"Unsupported event storage backend: {backend}")
