"""FastAPI app serving the chat UI and the conversation API.

Run it with ``python run_ui.py`` (or ``uvicorn webapp.app:app --reload``).
Everything is local: SQLite under ``data/``, generated prompts under
``outputs/<conversation_id>/``.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chat_common.common.logging import logger
from config import ENV_PATH, missing_env_vars
from optimization import dataset
from webapp import db, opt_db, optimizer, runner

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="NIQ Prompt Creator", docs_url="/api/docs")


@app.on_event("startup")
def _startup() -> None:
    db.connect()
    opt_db.ensure_schema()
    runner.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    optimizer.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if os.getenv("PROMPT_CREATOR_FAKE_LLM") == "1":
        # Lets the real chat UI be driven end to end with no LLM calls, for
        # manual walkthroughs of the pipeline. See fake_llm.py.
        for key in ("cis_llm_endpoint", "cis_llm_apikey", "cis_llm_apiversion", "CIS_LLM_4_DOT_1_DEPLOYMENT"):
            os.environ.setdefault(key, "fake")

        import runtime
        from fake_llm import FakeOpenAIClient

        fake_client = FakeOpenAIClient()
        runtime.get_openai_client = lambda: fake_client
        logger.warning("PROMPT_CREATOR_FAKE_LLM=1: all LLM calls are faked, no real completions will be made")
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


# --------------------------------------------------------------------------- optimization API
# The second workspace: build an LLM-as-a-judge from the user's own failure
# observations, grade a fixed test set with it, and loop the prompt until it
# passes. Mirrors the conversation API above — same NDJSON streaming, same
# "return the whole state" contract — against webapp/optimizer.py.


class NewOptimization(BaseModel):
    title: str = "New optimization"
    template_kind: str = "jinja2"


class OptimizationSettings(BaseModel):
    title: Optional[str] = None
    template_kind: Optional[str] = None


class PromptSubmission(BaseModel):
    prompt: str
    template_kind: str = "jinja2"


class ObservationSubmission(BaseModel):
    observations: str


class CriterionInput(BaseModel):
    id: str = ""
    title: str = ""
    description: str = ""
    origin: str = "agent"


class CriteriaSubmission(BaseModel):
    criteria: list[CriterionInput] = []
    additions: str = ""


class CaseFeedback(BaseModel):
    message_id: str
    agrees: bool = True
    reason: str = ""


class ReviewSubmission(BaseModel):
    feedback: list[CaseFeedback] = []


def _require_optimization(optimization_id: str) -> dict[str, Any]:
    optimization = opt_db.get_optimization(optimization_id)
    if optimization is None:
        raise HTTPException(status_code=404, detail=f"No optimization {optimization_id}")
    return optimization


def _record_optimizer_failure(optimization_id: str, error: Exception) -> None:
    """Turn a failed step into a card in the session, the way _record_failure does for chat.

    The resume stage is read back from the transcript rather than guessed: each
    interactive card is superseded only once its step succeeds, so the newest
    open form is exactly where the user should land.
    """
    logger.error(
        "Optimization step failed",
        extra={"optimization_id": optimization_id, "error": str(error)},
    )
    if not isinstance(error, optimizer.OptimizerError):
        traceback.print_exc()
    opt_db.add_message(
        optimization_id,
        role="assistant",
        kind="error",
        content=f"{type(error).__name__}: {error}" if not isinstance(error, optimizer.OptimizerError) else str(error),
    )
    open_forms = {
        "observations_form": "awaiting_observations",
        "criteria_form": "reviewing_criteria",
        "dataset_form": "awaiting_dataset",
        "run_form": "ready_to_run",
        "results_form": "reviewing_results",
        "iteration_form": "awaiting_iteration",
    }
    resume = "awaiting_prompt"
    for message in opt_db.list_messages(optimization_id):
        if message["kind"] in open_forms:
            resume = open_forms[message["kind"]]
    opt_db.update_optimization(optimization_id, stage=resume)


def _optimizer_stream(optimization_id: str, events: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
    """Relay optimizer events as NDJSON, closing with an authoritative state snapshot."""

    async def body() -> AsyncIterator[bytes]:
        try:
            async for event in events:
                yield (json.dumps(event) + "\n").encode("utf-8")
        except Exception as error:  # noqa: BLE001 - surfaced to the user, logged in full
            _record_optimizer_failure(optimization_id, error)
            detail = str(error) if isinstance(error, optimizer.OptimizerError) else f"{type(error).__name__}: {error}"
            yield (json.dumps({"type": "error", "detail": detail}) + "\n").encode("utf-8")
        yield (json.dumps({"type": "state", **optimizer._state(optimization_id)}) + "\n").encode("utf-8")

    return StreamingResponse(
        body(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/optimizer/steps")
def get_optimizer_steps() -> dict[str, Any]:
    """Describe every step of the optimization loop, in order, for the sidebar."""
    return {
        "steps": [{"key": key, **optimizer.STEP_INFO[key]} for key in optimizer.STEP_ORDER],
        "template_kinds": list(dataset.TEMPLATE_KINDS),
        "column_help": dataset.COLUMN_HELP,
    }


@app.get("/api/optimizations")
def get_optimizations() -> dict[str, Any]:
    return {"optimizations": opt_db.list_optimizations()}


@app.post("/api/optimizations")
def post_optimization(body: NewOptimization) -> dict[str, Any]:
    if body.template_kind not in dataset.TEMPLATE_KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown template type: {body.template_kind}")
    optimization_id = opt_db.create_optimization(title=body.title, template_kind=body.template_kind)
    return optimizer._state(optimization_id)


@app.get("/api/optimizations/{optimization_id}")
def get_optimization(optimization_id: str) -> dict[str, Any]:
    _require_optimization(optimization_id)
    return optimizer._state(optimization_id)


@app.patch("/api/optimizations/{optimization_id}")
def patch_optimization(optimization_id: str, body: OptimizationSettings) -> dict[str, Any]:
    _require_optimization(optimization_id)
    fields = body.model_dump(exclude_none=True)
    if "template_kind" in fields and fields["template_kind"] not in dataset.TEMPLATE_KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown template type: {fields['template_kind']}")
    opt_db.update_optimization(optimization_id, **fields)
    return optimizer._state(optimization_id)


@app.delete("/api/optimizations/{optimization_id}")
def remove_optimization(optimization_id: str) -> dict[str, Any]:
    _require_optimization(optimization_id)
    opt_db.delete_optimization(optimization_id)
    return {"deleted": optimization_id}


@app.post("/api/optimizations/{optimization_id}/prompt/stream")
def post_optimizer_prompt(optimization_id: str, body: PromptSubmission) -> StreamingResponse:
    _require_optimization(optimization_id)
    return _optimizer_stream(
        optimization_id, optimizer.submit_prompt_stream(optimization_id, body.prompt, body.template_kind)
    )


@app.post("/api/optimizations/{optimization_id}/observations/stream")
def post_optimizer_observations(optimization_id: str, body: ObservationSubmission) -> StreamingResponse:
    _require_optimization(optimization_id)
    return _optimizer_stream(
        optimization_id, optimizer.submit_observations_stream(optimization_id, body.observations)
    )


@app.post("/api/optimizations/{optimization_id}/criteria/stream")
def post_optimizer_criteria(optimization_id: str, body: CriteriaSubmission) -> StreamingResponse:
    _require_optimization(optimization_id)
    criteria = [criterion.model_dump() for criterion in body.criteria]
    return _optimizer_stream(
        optimization_id, optimizer.submit_criteria_stream(optimization_id, criteria, body.additions)
    )


@app.get("/api/optimizer/dataset-template")
def get_dataset_template() -> Response:
    """The starter workbook, so nobody has to guess the column names."""
    filename = dataset.sample_filename()
    media_type = (
        "text/csv"
        if filename.endswith(".csv")
        else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    return Response(
        content=dataset.sample_workbook(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/optimizations/{optimization_id}/dataset")
async def post_optimizer_dataset(optimization_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """Parse and store the uploaded test cases.

    A :class:`dataset.DatasetError` is a spreadsheet the user needs to fix, so it
    comes back as a 400 with the sentence to fix it — not as a card in the
    transcript, because nothing has happened to the session yet.
    """
    _require_optimization(optimization_id)
    raw = await file.read()
    try:
        cases = dataset.parse_dataset(raw, file.filename or "")
    except dataset.DatasetError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    rows = [
        {
            "message_id": case.message_id,
            "input_payload": case.input_payload,
            "other_input_params": case.other_input_params,
        }
        for case in cases
    ]
    return optimizer.store_dataset(optimization_id, rows, file.filename or "uploaded file")


@app.post("/api/optimizations/{optimization_id}/run/stream")
def post_optimizer_run(optimization_id: str) -> StreamingResponse:
    _require_optimization(optimization_id)
    return _optimizer_stream(optimization_id, optimizer.run_iteration_stream(optimization_id))


@app.post("/api/optimizations/{optimization_id}/review/stream")
def post_optimizer_review(optimization_id: str, body: ReviewSubmission) -> StreamingResponse:
    _require_optimization(optimization_id)
    feedback = [item.model_dump() for item in body.feedback]
    return _optimizer_stream(optimization_id, optimizer.submit_review_stream(optimization_id, feedback))


@app.post("/api/optimizations/{optimization_id}/stop")
def post_optimizer_stop(optimization_id: str) -> dict[str, Any]:
    """Close a session off after an iteration instead of running another one.

    Only meaningful while a "run it again?" card is open, so it retires that card
    rather than touching anything the loop produced — the prompt, the criteria,
    and every graded run stay exactly as they are.
    """
    _require_optimization(optimization_id)
    opt_db.supersede_kind(optimization_id, "iteration_form", stopped=True)
    opt_db.update_optimization(optimization_id, stage="complete")
    opt_db.add_message(
        optimization_id,
        role="user",
        kind="text",
        content="Stopping here — keeping the current prompt version.",
    )
    return optimizer._state(optimization_id)


@app.get("/api/optimizations/{optimization_id}/results.csv")
def get_optimizer_results(optimization_id: str):
    """Download the graded table for the latest iteration."""
    optimization = _require_optimization(optimization_id)
    path = Path(optimization["output_path"] or "")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No iteration has been run for this optimization yet")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/optimizations/{optimization_id}/prompt")
def get_optimizer_prompt(optimization_id: str) -> Response:
    """Download the latest prompt version as markdown."""
    optimization = _require_optimization(optimization_id)
    prompt = optimization["current_prompt"]
    if not prompt.strip():
        raise HTTPException(status_code=404, detail="This optimization has no prompt yet")
    return Response(
        content=prompt,
        media_type="text/markdown",
        headers={
            "Content-Disposition":
                f'attachment; filename="prompt_v{optimization["prompt_version"]}_{optimization_id}.md"'
        },
    )


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