"""FastAPI app serving the chat UI and the conversation API.

Run it with ``python run_ui.py`` (or ``uvicorn webapp.app:app --reload``).
Everything is local: SQLite under ``data/``, generated prompts under
``outputs/<conversation_id>/``.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chat_common.common.logging import logger
from config import ENV_PATH, missing_env_vars
from webapp import db, runner

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="NIQ Prompt Creator", docs_url="/api/docs")


@app.on_event("startup")
def _startup() -> None:
    db.connect()
    runner.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    missing = missing_env_vars()
    if missing:
        logger.warning(
            "Missing environment variables; Prompty loads and LLM calls will fail",
            extra={"missing": ", ".join(missing), "env_file": str(ENV_PATH)},
        )
    logger.info("Prompt Creator UI ready", extra={"db_path": str(db.DB_PATH)})


# --------------------------------------------------------------------------- request models


class NewConversation(BaseModel):
    title: str = "New chat"
    audience: str = "a general-purpose LLM"
    target_model: str = ""
    model_notes: str = ""
    humanize: bool = False
    max_clarify_rounds: int = Field(default=3, ge=0, le=6)
    start_stage: str = "build_prompt"


class ConversationSettings(BaseModel):
    title: Optional[str] = None
    audience: Optional[str] = None
    target_model: Optional[str] = None
    model_notes: Optional[str] = None
    humanize: Optional[bool] = None
    max_clarify_rounds: Optional[int] = Field(default=None, ge=0, le=6)
    start_stage: Optional[str] = None


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


def _validate_start_stage(start_stage: str, target_model: str) -> None:
    if start_stage not in runner.STAGE_ORDER:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline step: {start_stage}")
    if start_stage == "tune_for_model" and not target_model.strip():
        raise HTTPException(
            status_code=400,
            detail="Set a target model before starting from the 'Tune for model' step",
        )


def _record_failure(conversation_id: str, error: Exception) -> None:
    """Turn an LLM/runtime failure into a chat-visible error card.

    A stack trace in the server log is no use to someone staring at a silent UI,
    so the failure is written into the transcript and the conversation is left
    resumable rather than wedged mid-stage.
    """
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


async def _guarded(conversation_id: str, coro) -> dict[str, Any]:
    """Run a pipeline step, turning any failure into a chat-visible error card."""
    try:
        return await coro
    except Exception as error:  # noqa: BLE001 - surfaced to the user, logged in full
        _record_failure(conversation_id, error)
        return runner._state(conversation_id)


def _stream_response(conversation_id: str, events: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
    """Relay pipeline events to the browser as newline-delimited JSON.

    NDJSON rather than SSE because the UI drives these with ``fetch`` POSTs;
    ``EventSource`` only speaks GET. The terminating ``state`` event carries the
    same payload the blocking endpoints return, so the client re-renders from a
    single authoritative snapshot once the run settles.
    """

    async def body() -> AsyncIterator[bytes]:
        try:
            async for event in events:
                yield (json.dumps(event) + "\n").encode("utf-8")
        except Exception as error:  # noqa: BLE001 - surfaced to the user, logged in full
            _record_failure(conversation_id, error)
            yield (json.dumps({"type": "error", "detail": f"{type(error).__name__}: {error}"}) + "\n").encode("utf-8")
        yield (json.dumps({"type": "state", **runner._state(conversation_id)}) + "\n").encode("utf-8")

    return StreamingResponse(
        body(),
        media_type="application/x-ndjson",
        # Without these a proxy (or the browser) may sit on the response until it
        # completes, which is exactly the wait streaming exists to remove.
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- API


@app.get("/api/stages")
def get_stages() -> dict[str, Any]:
    """Describe every pipeline step, in order, for the sidebar's step picker."""
    return {
        "stages": [{"key": key, **runner.STAGE_INFO[key]} for key in runner.STAGE_ORDER],
    }


@app.get("/api/conversations")
def get_conversations() -> dict[str, Any]:
    return {"conversations": db.list_conversations()}


@app.post("/api/conversations")
def post_conversation(body: NewConversation) -> dict[str, Any]:
    _validate_start_stage(body.start_stage, body.target_model)
    conversation_id = db.create_conversation(
        title=body.title,
        audience=body.audience,
        target_model=body.target_model,
        model_notes=body.model_notes,
        humanize=int(body.humanize),
        max_clarify_rounds=body.max_clarify_rounds,
        start_stage=body.start_stage,
    )
    return runner._state(conversation_id)


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, Any]:
    _require(conversation_id)
    return runner._state(conversation_id)


@app.patch("/api/conversations/{conversation_id}")
def patch_conversation(conversation_id: str, body: ConversationSettings) -> dict[str, Any]:
    conversation = _require(conversation_id)
    fields = {key: value for key, value in body.model_dump(exclude_none=True).items()}
    if "humanize" in fields:
        fields["humanize"] = int(fields["humanize"])
    if "start_stage" in fields:
        if conversation["stage"] != "awaiting_idea":
            raise HTTPException(status_code=400, detail="Can't change the starting step after the pipeline has run")
        _validate_start_stage(fields["start_stage"], fields.get("target_model", conversation["target_model"]))
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


# --------------------------------------------------------------------------- streaming API
# Same three actions as above, but reporting progress as they run. The UI uses
# these; the blocking variants stay for scripted/API callers.


@app.post("/api/conversations/{conversation_id}/messages/stream")
def post_message_stream(conversation_id: str, body: UserMessage) -> StreamingResponse:
    conversation = _require(conversation_id)
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message is empty")
    if conversation["stage"] == "clarifying":
        # Free-text sent while an MCQ card is open counts as the answer to it.
        events = runner.submit_answers_stream(
            conversation_id, [{"question": "Additional detail", "selected": [], "other": content}]
        )
    else:
        events = runner.start_stream(conversation_id, content)
    return _stream_response(conversation_id, events)


@app.post("/api/conversations/{conversation_id}/answers/stream")
def post_answers_stream(conversation_id: str, body: AnswerSubmission) -> StreamingResponse:
    _require(conversation_id)
    answers = [answer.model_dump() for answer in body.answers]
    return _stream_response(conversation_id, runner.submit_answers_stream(conversation_id, answers))


@app.post("/api/conversations/{conversation_id}/skip/stream")
def post_skip_stream(conversation_id: str) -> StreamingResponse:
    _require(conversation_id)
    return _stream_response(conversation_id, runner.skip_clarification_stream(conversation_id))


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


@app.get("/favicon.ico", include_in_schema=False)
def get_favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


class NoCacheStaticFiles(StaticFiles):
    """Serve the UI with caching disabled.

    Browsers otherwise keep serving a stale ``app.js``/``styles.css`` after an
    edit, which looks like a broken feature rather than a cache hit. This is a
    local single-user app, so there is nothing to gain from caching anyway.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:  # noqa: D102 - see class docstring
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


app.mount("/", NoCacheStaticFiles(directory=STATIC_DIR, html=True), name="static")