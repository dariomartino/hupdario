"""Memoria persistente (SQLite in /data): compiti, regole, obiettivi, diario e chat."""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
from dataclasses import dataclass

OPEN_STATUSES = ("aperto", "da_verificare")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    room TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    next_action_at REAL NOT NULL,
    reminders INTEGER NOT NULL DEFAULT 0,
    channel TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS goals (
    room TEXT PRIMARY KEY,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    room TEXT,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class Task:
    id: str
    room: str
    text: str
    status: str
    created_at: float
    updated_at: float
    next_action_at: float
    reminders: int
    channel: str | None
    note: str | None

    def as_dict(self) -> dict:
        return {"task_id": self.id, "stanza": self.room, "compito": self.text, "stato": self.status,
                "promemoria_inviati": self.reminders, "nota": self.note or ""}


class Memory:
    def __init__(self, path: str, clock=time.time):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.clock = clock

    # ---------- compiti ----------
    def add_task(self, room: str, text: str, next_action_at: float) -> Task:
        now = self.clock()
        tid = secrets.token_hex(3)
        self.db.execute(
            "INSERT INTO tasks (id, room, text, status, created_at, updated_at, next_action_at) "
            "VALUES (?, ?, ?, 'aperto', ?, ?, ?)",
            (tid, room, text, now, now, next_action_at),
        )
        self.db.commit()
        return self.task(tid)

    def task(self, tid: str) -> Task | None:
        row = self.db.execute("SELECT * FROM tasks WHERE id = ?", (tid.strip().lower(),)).fetchone()
        return Task(**dict(row)) if row else None

    def update_task(self, tid: str, **fields) -> Task | None:
        if not fields:
            return self.task(tid)
        fields["updated_at"] = self.clock()
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE tasks SET {cols} WHERE id = ?", (*fields.values(), tid))
        self.db.commit()
        return self.task(tid)

    def open_tasks(self, room: str | None = None) -> list[Task]:
        q = f"SELECT * FROM tasks WHERE status IN {OPEN_STATUSES}"
        args: tuple = ()
        if room:
            q += " AND room = ?"
            args = (room,)
        return [Task(**dict(r)) for r in self.db.execute(q + " ORDER BY created_at", args)]

    def recent_closed(self, room: str, hours: float = 24, limit: int = 10) -> list[Task]:
        since = self.clock() - hours * 3600
        rows = self.db.execute(
            f"SELECT * FROM tasks WHERE room = ? AND status NOT IN {OPEN_STATUSES} AND updated_at >= ? "
            "ORDER BY updated_at DESC LIMIT ?", (room, since, limit))
        return [Task(**dict(r)) for r in rows]

    def tasks_page(self, limit: int = 30) -> list[Task]:
        rows = self.db.execute("SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [Task(**dict(r)) for r in rows]

    # ---------- regole e obiettivi ----------
    def add_rule(self, room: str, text: str) -> int:
        cur = self.db.execute("INSERT INTO rules (room, text, created_at) VALUES (?, ?, ?)",
                              (room, text.strip(), self.clock()))
        self.db.commit()
        return int(cur.lastrowid)

    def remove_rule(self, rule_id: int) -> bool:
        cur = self.db.execute("DELETE FROM rules WHERE id = ?", (int(rule_id),))
        self.db.commit()
        return cur.rowcount > 0

    def rules(self, room: str | None = None) -> list[dict]:
        if room:
            rows = self.db.execute("SELECT * FROM rules WHERE room = ? ORDER BY id", (room,))
        else:
            rows = self.db.execute("SELECT * FROM rules ORDER BY room, id")
        return [dict(r) for r in rows]

    def goal(self, room: str, default: str) -> str:
        row = self.db.execute("SELECT text FROM goals WHERE room = ?", (room,)).fetchone()
        return row["text"] if row else default

    def set_goal(self, room: str, text: str) -> None:
        self.db.execute("INSERT INTO goals (room, text) VALUES (?, ?) "
                        "ON CONFLICT(room) DO UPDATE SET text = excluded.text", (room, text.strip()))
        self.db.commit()

    # ---------- diario ----------
    def log(self, kind: str, text: str, room: str | None = None) -> None:
        self.db.execute("INSERT INTO events (ts, kind, room, text) VALUES (?, ?, ?, ?)",
                        (self.clock(), kind, room, text))
        self.db.execute("DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT 2000)")
        self.db.commit()

    def events(self, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))]

    # ---------- chat ----------
    def add_chat(self, role: str, text: str) -> None:
        self.db.execute("INSERT INTO chat (ts, role, text) VALUES (?, ?, ?)", (self.clock(), role, text))
        self.db.execute("DELETE FROM chat WHERE id NOT IN (SELECT id FROM chat ORDER BY id DESC LIMIT 200)")
        self.db.commit()

    def chat_history(self, limit: int = 20, max_age_hours: float = 12) -> list[dict]:
        since = self.clock() - max_age_hours * 3600
        rows = self.db.execute("SELECT role, text FROM chat WHERE ts >= ? ORDER BY id DESC LIMIT ?",
                               (since, limit)).fetchall()
        history = [{"role": r["role"], "text": r["text"]} for r in reversed(rows)]
        while history and history[0]["role"] != "user":  # la conversazione deve iniziare dall'utente
            history.pop(0)
        return history

    # ---------- chiave/valore ----------
    def get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value) -> None:
        self.db.execute("INSERT INTO kv (key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, json.dumps(value)))
        self.db.commit()
