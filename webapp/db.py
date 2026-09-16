"""SQLite persistence for conversations, chat history, and per-stage telemetry.

Local, single-user, zero-setup: the schema is created on first import of
``connect()``'s target file. Nothing here knows about the LLM — the runner owns
pipeline logic, this module owns rows.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from chat_common.services.llm_service import INPUT_COST_PER_1K, OUTPUT_COST_PER_1K

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "conversations.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id                 TEXT PRIMARY KEY,
    title              TEXT    NOT NULL DEFAULT 'New chat',
    created_at         TEXT    NOT NULL,
    updated_at         TEXT    NOT NULL,
    stage              TEXT    NOT NULL DEFAULT 'awaiting_idea',
    raw_idea           TEXT    NOT NULL DEFAULT '',
    built_prompt       TEXT    NOT NULL DEFAULT '',
    brief              TEXT    NOT NULL DEFAULT '',
    plan               TEXT    NOT NULL DEFAULT '',
    optimized_prompt   TEXT    NOT NULL DEFAULT '',
    final_prompt       TEXT    NOT NULL DEFAULT '',
    audience           TEXT    NOT NULL DEFAULT 'a general-purpose LLM',
    target_model       TEXT    NOT NULL DEFAULT '',
    model_notes        TEXT    NOT NULL DEFAULT '',
    humanize           INTEGER NOT NULL DEFAULT 0,
    clarify_rounds     INTEGER NOT NULL DEFAULT 0,
    max_clarify_rounds INTEGER NOT NULL DEFAULT 3,
    output_path        TEXT    NOT NULL DEFAULT '',
    start_stage        TEXT    NOT NULL DEFAULT 'build_prompt'
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'text',
    content         TEXT NOT NULL DEFAULT '',
    payload         TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, id);

CREATE TABLE IF NOT EXISTS stage_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id   TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    stage             TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    latency_seconds   REAL    NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stage_runs_conversation ON stage_runs (conversation_id, id);
"""

_CONN: Optional[sqlite3.Connection] = None
# Reentrant: every write below takes the lock and then calls connect(), which
# takes it again. A plain Lock deadlocks on the first insert.
_LOCK = threading.RLock()


def _now() -> str:
    """UTC ISO-8601 with microseconds.

    Microseconds matter: two conversations created in the same second would
    otherwise tie in the sidebar ordering and fall back to an arbitrary id sort.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Return the process-wide connection, creating the file and schema if needed."""
    global _CONN
    with _LOCK:
        if _CONN is None:
            target = Path(path or DB_PATH)
            target.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: FastAPI runs handlers on a worker thread pool
            # and every write below is serialized through _LOCK anyway.
            _CONN = sqlite3.connect(target, check_same_thread=False)
            _CONN.row_factory = sqlite3.Row
            _CONN.execute("PRAGMA foreign_keys = ON")
            _CONN.execute("PRAGMA journal_mode = WAL")
            _CONN.executescript(_SCHEMA)
            _migrate(_CONN)
            _CONN.commit()
        return _CONN


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database file already exists.

    ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so new
    columns need an explicit, idempotent ``ALTER TABLE`` here.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    if "start_stage" not in existing:
        conn.execute("ALTER TABLE conversations ADD COLUMN start_stage TEXT NOT NULL DEFAULT 'build_prompt'")


def close() -> None:
    """Drop the cached connection (tests, or swapping DB files)."""
    global _CONN
    with _LOCK:
        if _CONN is not None:
            _CONN.close()
            _CONN = None


# --------------------------------------------------------------------------- conversations


def create_conversation(title: str = "New chat", **fields: Any) -> str:
    """Insert a fresh conversation and return its generated id."""
    conversation_id = uuid.uuid4().hex[:12]
    now = _now()
    columns = ["id", "title", "created_at", "updated_at", *fields.keys()]
    values = [conversation_id, title, now, now, *fields.values()]
    placeholders = ", ".join("?" for _ in columns)
    with _LOCK:
        conn = connect()
        conn.execute(f"INSERT INTO conversations ({', '.join(columns)}) VALUES ({placeholders})", values)
        conn.commit()
    return conversation_id


def get_conversation(conversation_id: str) -> Optional[dict[str, Any]]:
    row = connect().execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    return dict(row) if row else None


def list_conversations(limit: int = 200) -> list[dict[str, Any]]:
    """Newest-first sidebar listing."""
    rows = connect().execute(
        # Lexicographic sort on the raw string, not datetime(): SQLite's datetime()
        # truncates to whole seconds, discarding the precision that breaks ties.
        # Safe because every timestamp is written as fixed-offset (+00:00) UTC.
        "SELECT c.id, c.title, c.created_at, c.updated_at, c.stage, c.target_model,"
        " COALESCE(u.prompt_tokens, 0) AS prompt_tokens, COALESCE(u.completion_tokens, 0) AS completion_tokens"
        " FROM conversations c"
        " LEFT JOIN ("
        "   SELECT conversation_id, SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens"
        "   FROM stage_runs GROUP BY conversation_id"
        " ) u ON u.conversation_id = c.id"
        " ORDER BY c.updated_at DESC, c.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conversations = []
    for row in rows:
        conversation = dict(row)
        conversation["estimated_cost"] = round(_estimate_cost(conversation.pop("prompt_tokens"), conversation.pop("completion_tokens")), 6)
        conversations.append(conversation)
    return conversations


def update_conversation(conversation_id: str, **fields: Any) -> None:
    """Patch named columns and bump ``updated_at``."""
    if not fields:
        return
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with _LOCK:
        conn = connect()
        conn.execute(
            f"UPDATE conversations SET {assignments}, updated_at = ? WHERE id = ?",
            [*fields.values(), _now(), conversation_id],
        )
        conn.commit()


def delete_conversation(conversation_id: str) -> bool:
    """Remove a conversation and everything hanging off it. Returns False if it was already gone.

    Children are deleted explicitly rather than relying on ``ON DELETE CASCADE``:
    the cascade only fires while ``PRAGMA foreign_keys`` is on, and a database
    file opened by anything else would silently leave orphaned rows behind.
    """
    with _LOCK:
        conn = connect()
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM stage_runs WHERE conversation_id = ?", (conversation_id,))
        cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        return cursor.rowcount > 0


# --------------------------------------------------------------------------- messages


def add_message(
    conversation_id: str,
    role: str,
    content: str = "",
    kind: str = "text",
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Append one chat turn and return it in the shape the frontend consumes."""
    now = _now()
    with _LOCK:
        conn = connect()
        cursor = conn.execute(
            "INSERT INTO messages (conversation_id, role, kind, content, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, role, kind, content, json.dumps(payload) if payload else None, now),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
        conn.commit()
        message_id = cursor.lastrowid
    return {
        "id": message_id,
        "conversation_id": conversation_id,
        "role": role,
        "kind": kind,
        "content": content,
        "payload": payload,
        "created_at": now,
    }


def list_messages(conversation_id: str) -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
    ).fetchall()
    messages = []
    for row in rows:
        message = dict(row)
        message["payload"] = json.loads(message["payload"]) if message["payload"] else None
        messages.append(message)
    return messages


def supersede_question_messages(
    conversation_id: str, answers: Optional[list[dict[str, Any]]] = None
) -> None:
    """Mark every outstanding question card answered so the UI stops accepting input.

    Without this, switching away and back to a conversation would re-render an
    already-answered MCQ card as live. ``answers`` is folded into the card's
    payload so a reopened conversation still shows which options were chosen —
    otherwise the card comes back with every radio blank.
    """
    with _LOCK:
        conn = connect()
        rows = conn.execute(
            "SELECT id, payload FROM messages WHERE conversation_id = ? AND kind = 'questions'",
            (conversation_id,),
        ).fetchall()
        for row in rows:
            payload = json.loads(row["payload"]) if row["payload"] else {}
            if answers is not None:
                payload["answers"] = answers
            conn.execute(
                "UPDATE messages SET kind = 'questions_answered', payload = ? WHERE id = ?",
                (json.dumps(payload), row["id"]),
            )
        conn.commit()


# --------------------------------------------------------------------------- telemetry


def record_stage_run(conversation_id: str, stage: str, completion: Any) -> None:
    """Persist token/latency figures that ``runtime.PromptCompletion`` already captures."""
    with _LOCK:
        conn = connect()
        conn.execute(
            "INSERT INTO stage_runs (conversation_id, stage, prompt_tokens, completion_tokens,"
            " total_tokens, latency_seconds, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                stage,
                getattr(completion, "prompt_tokens", 0),
                getattr(completion, "completion_tokens", 0),
                getattr(completion, "total_tokens", 0),
                getattr(completion, "latency_seconds", 0.0),
                _now(),
            ),
        )
        conn.commit()


def completed_stages(conversation_id: str) -> set[str]:
    """Stages that have produced at least one LLM call for this conversation.

    Derived from ``stage_runs`` rather than tracked separately, so the progress
    panel stays correct across reloads and conversation switches.
    """
    rows = connect().execute(
        "SELECT DISTINCT stage FROM stage_runs WHERE conversation_id = ?", (conversation_id,)
    ).fetchall()
    return {row["stage"] for row in rows}


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Recurring per-token API cost, using the same rates the LLM service logs against."""
    return (prompt_tokens / 1000) * INPUT_COST_PER_1K + (completion_tokens / 1000) * OUTPUT_COST_PER_1K


def usage_summary(conversation_id: str) -> dict[str, Any]:
    row = connect().execute(
        "SELECT COUNT(*) AS calls, COALESCE(SUM(total_tokens), 0) AS total_tokens,"
        " COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, COALESCE(SUM(completion_tokens), 0) AS completion_tokens,"
        " COALESCE(SUM(latency_seconds), 0) AS latency_seconds FROM stage_runs WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    summary = dict(row) if row else {
        "calls": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_seconds": 0,
    }
    summary["estimated_cost"] = round(_estimate_cost(summary["prompt_tokens"], summary["completion_tokens"]), 6)
    return summary