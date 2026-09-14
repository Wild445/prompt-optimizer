"""FastAPI app serving the chat UI and the conversation API.

Run it with ``python run_ui.py`` (or ``uvicorn webapp.app:app --reload``).
Everything is local: SQLite under ``data/``, generated prompts under
``outputs/<conversation_id>/``.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chat_common.common.logging import logger
from webapp import db, runner

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="NIQ Prompt Creator", docs_url="/api/docs")


@app.on_event("startup")
def _startup() -> None:
    db.connect()
    runner.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("Prompt Creator UI ready", extra={"db_path": str(db.DB_PATH)})


# --------------------------------------------------------------------------- request models


class NewConversation(BaseModel):
    title: str = "New chat"
    audience: str = "a general-purpose LLM"
    target_model: str = ""
    model_notes: str = ""
    humanize: bool = False
    max_clarify_rounds: int = Field(default=3, ge=0, le=6)


class ConversationSettings(BaseModel):
    title: Optional[str] = None
    audience: Optional[str] = None
    target_model: Optional[str] = None
    model_notes: Optional[str] = None
    humanize: Optional[bool] = None
    max_clarify_rounds: Optional[int] = Field(default=None, ge=0, le=6)


class UserMessage(BaseModel):
    content: str


class Answer(BaseModel):
    question: str
    selected: list[str] = []
    other: str = ""


class AnswerSubmission(BaseModel):
    answers: list[Answer]


# --------------------------------------------------------------------------- helpers


def _require(conversation_id: str) -> dict[str, Any]:
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"No conversation {conversation_id}")
    return conversation


async def _guarded(conversation_id: str, coro) -> dict[str, Any]:
    """Run a pipeline step, turning any LLM/runtime failure into a chat-visible error.

    A stack trace in the server log is no use to someone staring at a silent UI,
    so the failure is written into the transcript and the conversation is left
    resumable rather than wedged mid-stage.
    """
    try:
        return await coro
    except Exception as error:  # noqa: BLE001 - surfaced to the user, logged in full
        logger.error(
            "Pipeline stage failed",
            extra={"conversation_id": conversation_id, "error": str(error)},
        )
        traceback.print_exc()
        db.add_message(
            conversation_id,
            role="assistant",
            kind="error",
            content=f"{type(error).__name__}: {error}",
        )
        conversation = db.get_conversation(conversation_id)
        # Roll back to a stage the user can act from instead of leaving 'running'.
        resume = "clarifying" if conversation and conversation["clarify_rounds"] else "awaiting_idea"
        db.update_conversation(conversation_id, stage=resume)
        return runner._state(conversation_id)


# --------------------------------------------------------------------------- API


@app.get("/api/conversations")
def get_conversations() -> dict[str, Any]:
    return {"conversations": db.list_conversations()}


@app.post("/api/conversations")
def post_conversation(body: NewConversation) -> dict[str, Any]:
    conversation_id = db.create_conversation(
        title=body.title,
        audience=body.audience,
        target_model=body.target_model,
        model_notes=body.model_notes,
        humanize=int(body.humanize),
        max_clarify_rounds=body.max_clarify_rounds,
    )
    return runner._state(conversation_id)


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, Any]:
    _require(conversation_id)
    return runner._state(conversation_id)


@app.patch("/api/conversations/{conversation_id}")
def patch_conversation(conversation_id: str, body: ConversationSettings) -> dict[str, Any]:
    _require(conversation_id)
    fields = {key: value for key, value in body.model_dump(exclude_none=True).items()}
    if "humanize" in fields:
        fields["humanize"] = int(fields["humanize"])
    db.update_conversation(conversation_id, **fields)
    return runner._state(conversation_id)


@app.delete("/api/conversations/{conversation_id}")
def remove_conversation(conversation_id: str) -> dict[str, Any]:
    _require(conversation_id)
    db.delete_conversation(conversation_id)
    return {"deleted": conversation_id}


@app.post("/api/conversations/{conversation_id}/messages")
async def post_message(conversation_id: str, body: UserMessage) -> dict[str, Any]:
    conversation = _require(conversation_id)
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message is empty")
    if conversation["stage"] == "clarifying":
        # Free-text sent while an MCQ card is open counts as the answer to it.
        return await _guarded(
            conversation_id,
            runner.submit_answers(conversation_id, [{"question": "Additional detail", "selected": [], "other": content}]),
        )
    return await _guarded(conversation_id, runner.start(conversation_id, content))


@app.post("/api/conversations/{conversation_id}/answers")
async def post_answers(conversation_id: str, body: AnswerSubmission) -> dict[str, Any]:
    _require(conversation_id)
    answers = [answer.model_dump() for answer in body.answers]
    return await _guarded(conversation_id, runner.submit_answers(conversation_id, answers))


@app.post("/api/conversations/{conversation_id}/skip")
async def post_skip(conversation_id: str) -> dict[str, Any]:
    _require(conversation_id)
    return await _guarded(conversation_id, runner.skip_clarification(conversation_id))


@app.get("/api/conversations/{conversation_id}/output")
def get_output(conversation_id: str):
    """Download the saved prompt file for this conversation."""
    conversation = _require(conversation_id)
    path = Path(conversation["output_path"] or "")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No output saved for this conversation yet")
    return FileResponse(path, media_type="text/markdown", filename=path.name)


@app.exception_handler(KeyError)
def _key_error_handler(_request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


# --------------------------------------------------------------------------- static UI

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
