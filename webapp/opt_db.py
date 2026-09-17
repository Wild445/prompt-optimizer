"""SQLite persistence for prompt-optimization sessions.

Shares the connection and the write lock with :mod:`webapp.db` — same file, same
single-writer discipline — but keeps its tables and queries here, because the
optimization loop's row shapes (criteria, test cases, per-iteration results) have
nothing to do with the chat pipeline's.

Table map:

``optimizations``
    One session: the prompt being optimized, its template dialect, the generated
    judge prompt, the current iteration, and the resume point.
``optimization_criteria``
    The judge's success criteria. Rows are replaced wholesale on each
    consolidation, since ids are positional (see ``service.assign_ids``).
``optimization_prompts``
    Every version of the prompt under test, with the change notes that produced it.
``optimization_cases``
    The uploaded test data, one row per ``message_id``.
``optimization_results``
    One row per test case per iteration: the response, the judge's verdict, and
    the user's agreement or dissent.
``optimization_messages``
    The session transcript, in the same shape ``webapp.db.messages`` uses so the
    frontend can render both workspaces with one code path.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from webapp.db import _LOCK, _estimate_cost, _now, connect

_SCHEMA = """
CREATE TABLE IF NOT EXISTS optimizations (
    id             TEXT PRIMARY KEY,
    title          TEXT    NOT NULL DEFAULT 'New optimization',
    created_at     TEXT    NOT NULL,
    updated_at     TEXT    NOT NULL,
    stage          TEXT    NOT NULL DEFAULT 'awaiting_prompt',
    template_kind  TEXT    NOT NULL DEFAULT 'jinja2',
    base_prompt    TEXT    NOT NULL DEFAULT '',
    current_prompt TEXT    NOT NULL DEFAULT '',
    judge_scaffold TEXT    NOT NULL DEFAULT '',
    judge_prompt   TEXT    NOT NULL DEFAULT '',
    observations   TEXT    NOT NULL DEFAULT '',
    iteration      INTEGER NOT NULL DEFAULT 0,
    prompt_version INTEGER NOT NULL DEFAULT 1,
    dataset_name   TEXT    NOT NULL DEFAULT '',
    output_path    TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS optimization_criteria (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_id TEXT NOT NULL REFERENCES optimizations(id) ON DELETE CASCADE,
    criterion_id    TEXT NOT NULL DEFAULT '',
    title           TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    origin          TEXT NOT NULL DEFAULT 'agent',
    position        INTEGER NOT NULL DEFAULT 0,
    added_iteration INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_opt_criteria ON optimization_criteria (optimization_id, position);

CREATE TABLE IF NOT EXISTS optimization_prompts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_id TEXT    NOT NULL REFERENCES optimizations(id) ON DELETE CASCADE,
    version         INTEGER NOT NULL,
    prompt          TEXT    NOT NULL DEFAULT '',
    change_notes    TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_opt_prompts ON optimization_prompts (optimization_id, version);

CREATE TABLE IF NOT EXISTS optimization_cases (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_id    TEXT    NOT NULL REFERENCES optimizations(id) ON DELETE CASCADE,
    message_id         TEXT    NOT NULL,
    input_payload      TEXT    NOT NULL DEFAULT '',
    other_input_params TEXT    NOT NULL DEFAULT '{}',
    position           INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_opt_cases ON optimization_cases (optimization_id, position);

CREATE TABLE IF NOT EXISTS optimization_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_id TEXT    NOT NULL REFERENCES optimizations(id) ON DELETE CASCADE,
    iteration       INTEGER NOT NULL,
    case_id         INTEGER NOT NULL,
    message_id      TEXT    NOT NULL,
    prompt_version  INTEGER NOT NULL DEFAULT 1,
    response        TEXT    NOT NULL DEFAULT '',
    verdict         INTEGER,
    result_label    TEXT    NOT NULL DEFAULT '',
    failed_criteria TEXT    NOT NULL DEFAULT '[]',
    failures        TEXT    NOT NULL DEFAULT '[]',
    judge_raw       TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'pending',
    error           TEXT    NOT NULL DEFAULT '',
    user_agrees     INTEGER NOT NULL DEFAULT 1,
    user_reason     TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_opt_results_unique
    ON optimization_results (optimization_id, iteration, message_id);

CREATE TABLE IF NOT EXISTS optimization_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_id TEXT NOT NULL REFERENCES optimizations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'text',
    content         TEXT NOT NULL DEFAULT '',
    payload         TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_opt_messages ON optimization_messages (optimization_id, id);

CREATE TABLE IF NOT EXISTS optimization_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_id   TEXT NOT NULL REFERENCES optimizations(id) ON DELETE CASCADE,
    step              TEXT NOT NULL,
    iteration         INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    latency_seconds   REAL    NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_opt_runs ON optimization_runs (optimization_id, id);
"""

_CHILD_TABLES = (
    "optimization_criteria",
    "optimization_prompts",
    "optimization_cases",
    "optimization_results",
    "optimization_messages",
    "optimization_runs",
)

_ready = False


def ensure_schema() -> None:
    """Create the optimization tables once per process.

    Separate from ``db.connect()`` so the chat pipeline keeps working untouched
    if anything here fails; called from the app's startup hook.
    """
    global _ready
    with _LOCK:
        if _ready:
            return
        conn = connect()
        conn.executescript(_SCHEMA)
        conn.commit()
        _ready = True


# --------------------------------------------------------------------------- sessions


def create_optimization(title: str = "New optimization", **fields: Any) -> str:
    """Insert a fresh optimization session and return its generated id."""
    ensure_schema()
    optimization_id = uuid.uuid4().hex[:12]
    now = _now()
    columns = ["id", "title", "created_at", "updated_at", *fields.keys()]
    values = [optimization_id, title, now, now, *fields.values()]
    placeholders = ", ".join("?" for _ in columns)
    with _LOCK:
        conn = connect()
        conn.execute(f"INSERT INTO optimizations ({', '.join(columns)}) VALUES ({placeholders})", values)
        conn.commit()
    return optimization_id


def get_optimization(optimization_id: str) -> Optional[dict[str, Any]]:
    ensure_schema()
    row = connect().execute("SELECT * FROM optimizations WHERE id = ?", (optimization_id,)).fetchone()
    return dict(row) if row else None


def list_optimizations(limit: int = 200) -> list[dict[str, Any]]:
    """Newest-first sidebar listing."""
    ensure_schema()
    rows = connect().execute(
        # Same lexicographic ordering as db.list_conversations; see the note there.
        "SELECT o.id, o.title, o.created_at, o.updated_at, o.stage, o.iteration,"
        " COALESCE(u.prompt_tokens, 0) AS prompt_tokens, COALESCE(u.completion_tokens, 0) AS completion_tokens"
        " FROM optimizations o"
        " LEFT JOIN ("
        "   SELECT optimization_id, SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens"
        "   FROM optimization_runs GROUP BY optimization_id"
        " ) u ON u.optimization_id = o.id"
        " ORDER BY o.updated_at DESC, o.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    optimizations = []
    for row in rows:
        optimization = dict(row)
        optimization["estimated_cost"] = round(
            _estimate_cost(optimization.pop("prompt_tokens"), optimization.pop("completion_tokens")), 6
        )
        optimizations.append(optimization)
    return optimizations


def update_optimization(optimization_id: str, **fields: Any) -> None:
    """Patch named columns and bump ``updated_at``."""
    if not fields:
        return
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with _LOCK:
        conn = connect()
        conn.execute(
            f"UPDATE optimizations SET {assignments}, updated_at = ? WHERE id = ?",
            [*fields.values(), _now(), optimization_id],
        )
        conn.commit()


def delete_optimization(optimization_id: str) -> bool:
    """Remove a session and every row hanging off it.

    Children go explicitly rather than by cascade, for the reason given in
    ``db.delete_conversation``.
    """
    ensure_schema()
    with _LOCK:
        conn = connect()
        for table in _CHILD_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE optimization_id = ?", (optimization_id,))
        cursor = conn.execute("DELETE FROM optimizations WHERE id = ?", (optimization_id,))
        conn.commit()
        return cursor.rowcount > 0


# --------------------------------------------------------------------------- criteria


def replace_criteria(optimization_id: str, criteria: list[dict[str, Any]], iteration: int = 0) -> None:
    """Swap in a whole criteria list, preserving each entry's original iteration.

    Wholesale replacement rather than a diff because ``C1..Cn`` are positional:
    re-running the consolidator can legitimately merge two entries into one, and
    trying to keep per-row identity across that would produce ids that no longer
    match the list the judge is reading.
    """
    with _LOCK:
        conn = connect()
        previous = {
            row["title"]: row["added_iteration"]
            for row in conn.execute(
                "SELECT title, added_iteration FROM optimization_criteria WHERE optimization_id = ?",
                (optimization_id,),
            ).fetchall()
        }
        conn.execute("DELETE FROM optimization_criteria WHERE optimization_id = ?", (optimization_id,))
        for position, criterion in enumerate(criteria):
            title = str(criterion.get("title") or "")
            conn.execute(
                "INSERT INTO optimization_criteria (optimization_id, criterion_id, title, description,"
                " origin, position, added_iteration) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    optimization_id,
                    str(criterion.get("id") or f"C{position + 1}"),
                    title,
                    str(criterion.get("description") or ""),
                    str(criterion.get("origin") or "agent"),
                    position,
                    int(previous.get(title, criterion.get("added_iteration", iteration))),
                ),
            )
        conn.commit()


def list_criteria(optimization_id: str) -> list[dict[str, Any]]:
    ensure_schema()
    rows = connect().execute(
        "SELECT criterion_id AS id, title, description, origin, added_iteration FROM optimization_criteria"
        " WHERE optimization_id = ? ORDER BY position",
        (optimization_id,),
    ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- prompt versions


def add_prompt_version(optimization_id: str, version: int, prompt: str, change_notes: str = "") -> None:
    with _LOCK:
        conn = connect()
        conn.execute(
            "INSERT INTO optimization_prompts (optimization_id, version, prompt, change_notes, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (optimization_id, version, prompt, change_notes, _now()),
        )
        conn.commit()


def list_prompt_versions(optimization_id: str) -> list[dict[str, Any]]:
    ensure_schema()
    rows = connect().execute(
        "SELECT version, prompt, change_notes, created_at FROM optimization_prompts"
        " WHERE optimization_id = ? ORDER BY version",
        (optimization_id,),
    ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- test cases


def replace_cases(optimization_id: str, cases: list[dict[str, Any]]) -> None:
    """Swap in a freshly uploaded dataset, dropping every result that graded the old one."""
    with _LOCK:
        conn = connect()
        conn.execute("DELETE FROM optimization_results WHERE optimization_id = ?", (optimization_id,))
        conn.execute("DELETE FROM optimization_cases WHERE optimization_id = ?", (optimization_id,))
        for position, case in enumerate(cases):
            conn.execute(
                "INSERT INTO optimization_cases (optimization_id, message_id, input_payload,"
                " other_input_params, position) VALUES (?, ?, ?, ?, ?)",
                (
                    optimization_id,
                    case["message_id"],
                    case["input_payload"],
                    json.dumps(case.get("other_input_params") or {}),
                    position,
                ),
            )
        conn.commit()


def list_cases(optimization_id: str) -> list[dict[str, Any]]:
    ensure_schema()
    rows = connect().execute(
        "SELECT id, message_id, input_payload, other_input_params FROM optimization_cases"
        " WHERE optimization_id = ? ORDER BY position",
        (optimization_id,),
    ).fetchall()
    cases = []
    for row in rows:
        case = dict(row)
        case["other_input_params"] = json.loads(case["other_input_params"] or "{}")
        cases.append(case)
    return cases


# --------------------------------------------------------------------------- results


def start_result(optimization_id: str, iteration: int, case: dict[str, Any], prompt_version: int) -> int:
    """Create (or reset) the row for one test case in this iteration.

    ``ON CONFLICT`` rather than a plain insert so re-running an iteration after a
    mid-run failure overwrites the partial row instead of tripping the unique
    index on ``(optimization_id, iteration, message_id)``.
    """
    with _LOCK:
        conn = connect()
        conn.execute(
            "INSERT INTO optimization_results (optimization_id, iteration, case_id, message_id,"
            " prompt_version, status, created_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)"
            " ON CONFLICT (optimization_id, iteration, message_id) DO UPDATE SET"
            " case_id = excluded.case_id, prompt_version = excluded.prompt_version, status = 'pending',"
            " response = '', verdict = NULL, result_label = '', failed_criteria = '[]', failures = '[]',"
            " judge_raw = '', error = '', user_agrees = 1, user_reason = ''",
            (optimization_id, iteration, case["id"], case["message_id"], prompt_version, _now()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM optimization_results WHERE optimization_id = ? AND iteration = ? AND message_id = ?",
            (optimization_id, iteration, case["message_id"]),
        ).fetchone()
        return int(row["id"])


def update_result(result_id: int, **fields: Any) -> None:
    if not fields:
        return
    for key in ("failed_criteria", "failures"):
        if key in fields and not isinstance(fields[key], str):
            fields[key] = json.dumps(fields[key])
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with _LOCK:
        conn = connect()
        conn.execute(
            f"UPDATE optimization_results SET {assignments} WHERE id = ?", [*fields.values(), result_id]
        )
        conn.commit()


def list_results(optimization_id: str, iteration: Optional[int] = None) -> list[dict[str, Any]]:
    """Results for one iteration (the latest by default), joined onto their test case."""
    ensure_schema()
    if iteration is None:
        row = connect().execute(
            "SELECT MAX(iteration) AS iteration FROM optimization_results WHERE optimization_id = ?",
            (optimization_id,),
        ).fetchone()
        iteration = row["iteration"] if row and row["iteration"] is not None else 0
    rows = connect().execute(
        "SELECT r.*, c.input_payload, c.other_input_params, c.position FROM optimization_results r"
        " JOIN optimization_cases c ON c.id = r.case_id"
        " WHERE r.optimization_id = ? AND r.iteration = ? ORDER BY c.position",
        (optimization_id, iteration),
    ).fetchall()
    results = []
    for row in rows:
        result = dict(row)
        result["other_input_params"] = json.loads(result["other_input_params"] or "{}")
        result["failed_criteria"] = json.loads(result["failed_criteria"] or "[]")
        result["failures"] = json.loads(result["failures"] or "[]")
        results.append(result)
    return results


def iteration_scoreboard(optimization_id: str, iteration: int) -> dict[str, int]:
    """Live pass/fail/error tallies for the progress panel."""
    ensure_schema()
    row = connect().execute(
        "SELECT COUNT(*) AS total,"
        " SUM(CASE WHEN verdict = 1 THEN 1 ELSE 0 END) AS passed,"
        " SUM(CASE WHEN verdict = 0 THEN 1 ELSE 0 END) AS failed,"
        " SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errored"
        " FROM optimization_results WHERE optimization_id = ? AND iteration = ?",
        (optimization_id, iteration),
    ).fetchone()
    return {key: int(row[key] or 0) for key in ("total", "passed", "failed", "errored")} if row else {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errored": 0,
    }


def set_user_feedback(optimization_id: str, iteration: int, feedback: list[dict[str, Any]]) -> None:
    """Record, per test case, whether the user agreed with the judge and why not."""
    with _LOCK:
        conn = connect()
        for item in feedback:
            conn.execute(
                "UPDATE optimization_results SET user_agrees = ?, user_reason = ?"
                " WHERE optimization_id = ? AND iteration = ? AND message_id = ?",
                (
                    1 if item.get("agrees", True) else 0,
                    str(item.get("reason") or "").strip(),
                    optimization_id,
                    iteration,
                    str(item.get("message_id") or ""),
                ),
            )
        conn.commit()


# --------------------------------------------------------------------------- transcript and telemetry


def add_message(
    optimization_id: str,
    role: str,
    content: str = "",
    kind: str = "text",
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Append one transcript entry and return it in the shape the frontend consumes."""
    ensure_schema()
    now = _now()
    with _LOCK:
        conn = connect()
        cursor = conn.execute(
            "INSERT INTO optimization_messages (optimization_id, role, kind, content, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (optimization_id, role, kind, content, json.dumps(payload) if payload else None, now),
        )
        conn.execute("UPDATE optimizations SET updated_at = ? WHERE id = ?", (now, optimization_id))
        conn.commit()
        message_id = cursor.lastrowid
    return {
        "id": message_id,
        "optimization_id": optimization_id,
        "role": role,
        "kind": kind,
        "content": content,
        "payload": payload,
        "created_at": now,
    }


def list_messages(optimization_id: str) -> list[dict[str, Any]]:
    ensure_schema()
    rows = connect().execute(
        "SELECT * FROM optimization_messages WHERE optimization_id = ? ORDER BY id", (optimization_id,)
    ).fetchall()
    messages = []
    for row in rows:
        message = dict(row)
        message["payload"] = json.loads(message["payload"]) if message["payload"] else None
        messages.append(message)
    return messages


def latest_message(optimization_id: str, kind: str) -> Optional[dict[str, Any]]:
    """The newest transcript entry of one kind, payload already decoded.

    Used to read a pause's own card back when the user acts on it — the proposed
    changes the approval card is waiting on live in its payload and nowhere else,
    so this is what makes that step survive a reload.
    """
    ensure_schema()
    row = connect().execute(
        "SELECT * FROM optimization_messages WHERE optimization_id = ? AND kind = ? ORDER BY id DESC LIMIT 1",
        (optimization_id, kind),
    ).fetchone()
    if row is None:
        return None
    message = dict(row)
    message["payload"] = json.loads(message["payload"]) if message["payload"] else None
    return message


def supersede_kind(optimization_id: str, kind: str, suffix: str = "_done", **payload_updates: Any) -> None:
    """Retire every open interactive card of one kind so it stops accepting input.

    Same purpose as ``db.supersede_question_messages``: without it, reopening a
    session re-renders an already-submitted form as live.
    """
    with _LOCK:
        conn = connect()
        rows = conn.execute(
            "SELECT id, payload FROM optimization_messages WHERE optimization_id = ? AND kind = ?",
            (optimization_id, kind),
        ).fetchall()
        for row in rows:
            payload = json.loads(row["payload"]) if row["payload"] else {}
            payload.update(payload_updates)
            conn.execute(
                "UPDATE optimization_messages SET kind = ?, payload = ? WHERE id = ?",
                (f"{kind}{suffix}", json.dumps(payload), row["id"]),
            )
        conn.commit()


def record_run(optimization_id: str, step: str, completion: Any, iteration: int = 0) -> None:
    """Persist token/latency figures for one LLM call in the loop."""
    if completion is None:
        return
    with _LOCK:
        conn = connect()
        conn.execute(
            "INSERT INTO optimization_runs (optimization_id, step, iteration, prompt_tokens,"
            " completion_tokens, total_tokens, latency_seconds, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                optimization_id,
                step,
                iteration,
                getattr(completion, "prompt_tokens", 0),
                getattr(completion, "completion_tokens", 0),
                getattr(completion, "total_tokens", 0),
                getattr(completion, "latency_seconds", 0.0),
                _now(),
            ),
        )
        conn.commit()


def completed_steps(optimization_id: str) -> set[str]:
    """Steps that have produced at least one LLM call, for the progress panel."""
    ensure_schema()
    rows = connect().execute(
        "SELECT DISTINCT step FROM optimization_runs WHERE optimization_id = ?", (optimization_id,)
    ).fetchall()
    return {row["step"] for row in rows}


def usage_summary(optimization_id: str) -> dict[str, Any]:
    ensure_schema()
    row = connect().execute(
        "SELECT COUNT(*) AS calls, COALESCE(SUM(total_tokens), 0) AS total_tokens,"
        " COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, COALESCE(SUM(completion_tokens), 0) AS completion_tokens,"
        " COALESCE(SUM(latency_seconds), 0) AS latency_seconds FROM optimization_runs"
        " WHERE optimization_id = ?",
        (optimization_id,),
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
