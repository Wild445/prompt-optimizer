"""Drives the six synthesis stages as a resumable, DB-backed state machine.

``synthesis.orchestrator.run_pipeline`` blocks on an ``answer_fn`` callback,
which suits a CLI but not a chat UI: the clarification pause has to survive
across HTTP requests, page reloads, and conversation switches. So this module
runs the same stages, persisting after each one, with ``conversations.stage`` as
the resume point.

Stages: ``awaiting_idea`` -> ``clarifying`` -> ``complete``.

Every entry point comes in two flavours. The ``*_stream`` generators yield
progress events (``stage_start`` / ``delta`` / ``stage_done``) as the pipeline
runs, which is what the UI consumes; the plain wrappers drain those generators
and return the final state, for callers that just want the result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from synthesis import service
from webapp import db

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"

#: One event emitted by the ``*_stream`` generators. ``type`` is one of
#: ``stage_start``, ``delta``, or ``stage_done``.
StreamEvent = dict[str, Any]

_STAGE_LABELS = {
    "build_prompt": "Built the initial prompt draft",
    "clarify": "Checked the draft for gaps",
    "plan_workflow": "Planned the workflow",
    "optimize_prompt": "Optimized the prompt",
    "tune_for_model": "Tuned for the target model",
    "humanize": "Humanized the wording",
}

# Canonical stage order, used both to decide what to skip when a conversation
# starts partway through the pipeline and to describe each step to the UI.
STAGE_ORDER = ["build_prompt", "clarify", "plan_workflow", "optimize_prompt", "tune_for_model", "humanize"]

STAGE_INFO = {
    "build_prompt": {
        "title": "Build prompt",
        "description": (
            "Turn a rough idea into a structured first-draft prompt. Start here if you're"
            " beginning from just a topic or a sentence or two."
        ),
    },
    "clarify": {
        "title": "Clarify",
        "description": (
            "Ask you clarifying questions to fill in missing detail before planning. Start"
            " here if you already have a draft prompt but want it interrogated for gaps."
        ),
    },
    "plan_workflow": {
        "title": "Plan workflow",
        "description": (
            "Break the brief into an ordered plan of steps the final prompt should cover."
            " Start here if your prompt/brief is already clear and doesn't need clarifying."
        ),
    },
    "optimize_prompt": {
        "title": "Optimize prompt",
        "description": (
            "Tighten the wording for stronger reasoning and cleaner output. Start here if"
            " you already have a solid prompt and just want it optimized."
        ),
    },
    "tune_for_model": {
        "title": "Tune for model",
        "description": (
            "Adapt the prompt to a specific target model's conventions and quirks. Requires"
            " a target model to be set in Settings. Start here if you already have an"
            " optimized prompt and only need it adapted for a particular model."
        ),
    },
    "humanize": {
        "title": "Humanize",
        "description": (
            "Strip AI-sounding phrasing and patterns from the final draft. Start here if you"
            " already have a finished prompt and just want the wording humanized."
        ),
    },
}


def _emit_stage(conversation_id: str, stage: str, content: str) -> dict[str, Any]:
    """Write one intermediate artifact into the transcript as a collapsible card."""
    return db.add_message(
        conversation_id,
        role="assistant",
        kind="stage",
        content=content,
        payload={"stage": stage, "label": _STAGE_LABELS.get(stage, stage)},
    )


# --------------------------------------------------------------------------- progress


def planned_stages(conversation: dict[str, Any]) -> list[str]:
    """The stages this conversation will actually run, in order.

    Mirrors the branching in :func:`_finalize_stream` so the progress panel never
    shows a step that is going to be skipped (no target model, humanizer off, or
    a start stage partway down the pipeline).
    """
    start_stage = conversation["start_stage"] or "build_prompt"
    start_index = STAGE_ORDER.index(start_stage) if start_stage in STAGE_ORDER else 0

    stages: list[str] = []
    for stage in STAGE_ORDER[start_index:]:
        if stage == "clarify" and conversation["max_clarify_rounds"] <= 0:
            continue
        if stage == "tune_for_model" and not (conversation["target_model"] or "").strip():
            continue
        if stage == "humanize" and not (conversation["humanize"] or start_stage == "humanize"):
            continue
        stages.append(stage)
    return stages


def steps_snapshot(conversation_id: str, active: Optional[str] = None) -> list[dict[str, Any]]:
    """Per-step progress for the UI panel: ``done`` / ``active`` / ``pending``.

    Completion is read back from ``stage_runs`` rather than tracked in memory, so
    the panel is still right after a reload or a conversation switch.
    """
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        return []
    completed = db.completed_stages(conversation_id)

    steps: list[dict[str, Any]] = []
    for stage in planned_stages(conversation):
        if stage == active or (stage == "clarify" and conversation["stage"] == "clarifying"):
            # Clarify stays 'active' while its questions are still unanswered.
            status = "active"
        elif stage in completed:
            status = "done"
        else:
            status = "pending"
        steps.append({"key": stage, "title": STAGE_INFO[stage]["title"], "status": status})
    return steps


def _progress(conversation_id: str, event_type: str, stage: str, **extra: Any) -> StreamEvent:
    """Build a progress event carrying a fresh snapshot of every step."""
    active = stage if event_type == "stage_start" else None
    return {
        "type": event_type,
        "stage": stage,
        "label": _STAGE_LABELS.get(stage, stage),
        "title": STAGE_INFO.get(stage, {}).get("title", stage),
        "steps": steps_snapshot(conversation_id, active=active),
        **extra,
    }


def format_answers(answers: list[dict[str, Any]]) -> str:
    """Render submitted MCQ answers as the text folded back into the brief."""
    lines: list[str] = []
    for answer in answers:
        question = str(answer.get("question") or "").strip()
        chosen = [str(option).strip() for option in (answer.get("selected") or []) if str(option).strip()]
        other = str(answer.get("other") or "").strip()
        if other:
            chosen.append(other)
        lines.append(f"Q: {question}\nA: {'; '.join(chosen) if chosen else 'No preference — use your best judgment.'}")
    return "\n\n".join(lines)


async def _run_stage(
    conversation_id: str,
    stage: str,
    text: str,
    outputs: dict[str, str],
    **options: Any,
) -> AsyncIterator[StreamEvent]:
    """Stream one stage, persist its artifact, and record what it produced.

    Yields ``stage_start``, a ``delta`` per token, then ``stage_done``. The
    stage's finished text lands in ``outputs[stage]`` for the caller to chain
    into the next stage — async generators can't return a value.
    """
    yield _progress(conversation_id, "stage_start", stage)

    stream = await service.stream_stage(service.stage_request(stage, text, **options))
    async for delta in stream:
        yield {"type": "delta", "stage": stage, "text": delta}

    completion = stream.completion
    db.record_stage_run(conversation_id, stage, completion)
    outputs[stage] = completion.content
    message = _emit_stage(conversation_id, stage, completion.content)
    yield _progress(conversation_id, "stage_done", stage, message=message)


async def _clarify_round_stream(conversation: dict[str, Any], brief: str) -> AsyncIterator[StreamEvent]:
    """Run one Clarify call; either post an MCQ card or move on to the final stages.

    Clarify is not token-streamed: it returns JSON the user never reads, so only
    its start/finish are reported for the progress panel.
    """
    conversation_id = conversation["id"]
    rounds_used = conversation["clarify_rounds"]
    max_rounds = conversation["max_clarify_rounds"]

    if rounds_used >= max_rounds:
        async for event in _finalize_stream(conversation, brief):
            yield event
        return

    yield _progress(conversation_id, "stage_start", "clarify")

    outcome = await service.clarify(brief)
    completion = outcome.get("completion")
    if completion is not None:
        db.record_stage_run(conversation_id, "clarify", completion)

    summary = outcome.get("clarified_summary") or brief
    questions = outcome.get("questions") or []
    rounds_used += 1
    db.update_conversation(conversation_id, clarify_rounds=rounds_used, brief=summary)

    if outcome.get("status") == "ready" or not questions:
        conversation = db.get_conversation(conversation_id) or conversation
        yield _progress(conversation_id, "stage_done", "clarify")
        async for event in _finalize_stream(conversation, summary):
            yield event
        return

    db.supersede_question_messages(conversation_id)
    db.add_message(
        conversation_id,
        role="assistant",
        kind="questions",
        content="A few things would change the output. Pick an option or write your own.",
        payload={"questions": questions, "round": rounds_used, "max_rounds": max_rounds},
    )
    db.update_conversation(conversation_id, stage="clarifying")
    yield _progress(conversation_id, "stage_done", "clarify")


async def _finalize_stream(
    conversation: dict[str, Any], brief: str, start_stage: str = "plan_workflow"
) -> AsyncIterator[StreamEvent]:
    """Run planner -> optimizer -> [model tuning] -> [humanizer], then save the result.

    ``start_stage`` lets the caller jump straight into the middle of this chain
    (see ``STAGE_ORDER``): any stage before it is skipped and its artifact falls
    back to ``brief`` untouched, rather than being generated.
    """
    conversation_id = conversation["id"]
    db.supersede_question_messages(conversation_id)
    start_index = STAGE_ORDER.index(start_stage) if start_stage in STAGE_ORDER else STAGE_ORDER.index("plan_workflow")
    outputs: dict[str, str] = {}

    plan = ""
    if start_index <= STAGE_ORDER.index("plan_workflow"):
        async for event in _run_stage(conversation_id, "plan_workflow", brief, outputs):
            yield event
        plan = outputs["plan_workflow"]

    optimized_prompt = brief
    if start_index <= STAGE_ORDER.index("optimize_prompt"):
        async for event in _run_stage(conversation_id, "optimize_prompt", brief, outputs):
            yield event
        optimized_prompt = outputs["optimize_prompt"]

    final_prompt = optimized_prompt
    target_model = (conversation["target_model"] or "").strip()
    if target_model and start_index <= STAGE_ORDER.index("tune_for_model"):
        async for event in _run_stage(
            conversation_id,
            "tune_for_model",
            final_prompt,
            outputs,
            target_model=target_model,
            model_notes=conversation["model_notes"] or "",
        ):
            yield event
        final_prompt = outputs["tune_for_model"]

    run_humanize = conversation["humanize"] or start_stage == "humanize"
    if run_humanize and start_index <= STAGE_ORDER.index("humanize"):
        async for event in _run_stage(conversation_id, "humanize", final_prompt, outputs):
            yield event
        final_prompt = outputs["humanize"]

    output_path = _save_output(conversation_id, brief, plan, final_prompt, target_model)
    db.update_conversation(
        conversation_id,
        stage="complete",
        brief=brief,
        plan=plan,
        optimized_prompt=optimized_prompt,
        final_prompt=final_prompt,
        output_path=str(output_path),
    )
    db.add_message(
        conversation_id,
        role="assistant",
        kind="final",
        content=final_prompt,
        payload={"output_path": str(output_path), "target_model": target_model},
    )
    _save_transcript(conversation_id)


def _save_output(
    conversation_id: str, brief: str, plan: str, final_prompt: str, target_model: str
) -> Path:
    """Write the finished prompt (plus its brief and plan) under ``outputs/<id>/``."""
    directory = OUTPUT_ROOT / conversation_id
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"final_prompt_{stamp}.md"
    path.write_text(
        "\n".join(
            [
                f"# Final prompt — conversation {conversation_id}",
                "",
                f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
                f"- Target model: {target_model or 'not specified (generic)'}",
                "",
                "## Prompt",
                "",
                final_prompt,
                "",
                "## Clarified brief",
                "",
                brief,
                "",
                "## Plan",
                "",
                plan,
                "",
            ]
        ),
        encoding="utf-8",
    )
    # `latest.md` gives a stable path to point tooling at; the stamped files are the history.
    (directory / "latest.md").write_text(final_prompt, encoding="utf-8")
    return path


def _save_transcript(conversation_id: str) -> None:
    """Snapshot the full chat history next to the prompt.

    Called after the final message is appended, not during ``_save_output`` —
    otherwise the snapshot is written one message short of the transcript the
    user is looking at.
    """
    directory = OUTPUT_ROOT / conversation_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "conversation.json").write_text(
        json.dumps(
            {
                "conversation_id": conversation_id,
                "conversation": db.get_conversation(conversation_id),
                "messages": db.list_messages(conversation_id),
                "usage": db.usage_summary(conversation_id),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _state(conversation_id: str) -> dict[str, Any]:
    """The payload every mutating endpoint returns: conversation + full transcript."""
    return {
        "conversation": db.get_conversation(conversation_id),
        "messages": db.list_messages(conversation_id),
        "usage": db.usage_summary(conversation_id),
        "steps": steps_snapshot(conversation_id),
    }


async def _drain(conversation_id: str, events: AsyncIterator[StreamEvent]) -> dict[str, Any]:
    """Run a ``*_stream`` generator to completion and return the resulting state."""
    async for _event in events:
        pass
    return _state(conversation_id)


# --------------------------------------------------------------------------- entry points


async def start_stream(conversation_id: str, raw_idea: str) -> AsyncIterator[StreamEvent]:
    """Handle the user's opening idea: build a draft, then start clarifying."""
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise KeyError(conversation_id)

    db.add_message(conversation_id, role="user", content=raw_idea)
    # First message doubles as the sidebar title.
    if conversation["title"] in ("", "New chat"):
        title = raw_idea.strip().splitlines()[0][:60] or "New chat"
        db.update_conversation(conversation_id, title=title)

    db.update_conversation(conversation_id, raw_idea=raw_idea, stage="running")

    start_stage = conversation["start_stage"] or "build_prompt"

    if start_stage == "build_prompt":
        outputs: dict[str, str] = {}
        async for event in _run_stage(
            conversation_id, "build_prompt", raw_idea, outputs, audience=conversation["audience"]
        ):
            yield event
        built_prompt = outputs["build_prompt"]
        db.update_conversation(conversation_id, built_prompt=built_prompt, brief=built_prompt)
        conversation = db.get_conversation(conversation_id) or conversation
        async for event in _clarify_round_stream(conversation, built_prompt):
            yield event
        return

    if start_stage == "clarify":
        # The user's message is already a draft prompt/brief, not a raw idea.
        db.update_conversation(conversation_id, built_prompt=raw_idea, brief=raw_idea)
        conversation = db.get_conversation(conversation_id) or conversation
        async for event in _clarify_round_stream(conversation, raw_idea):
            yield event
        return

    # plan_workflow / optimize_prompt / tune_for_model / humanize: everything
    # before the chosen stage is skipped and the message is treated as the
    # brief/prompt those later stages expect as input.
    db.update_conversation(
        conversation_id,
        built_prompt=raw_idea,
        brief=raw_idea,
        clarify_rounds=conversation["max_clarify_rounds"],
    )
    conversation = db.get_conversation(conversation_id) or conversation
    async for event in _finalize_stream(conversation, raw_idea, start_stage=start_stage):
        yield event


async def submit_answers_stream(
    conversation_id: str, answers: list[dict[str, Any]]
) -> AsyncIterator[StreamEvent]:
    """Fold MCQ answers into the brief and run the next Clarify round."""
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise KeyError(conversation_id)

    answer_text = format_answers(answers)
    db.supersede_question_messages(conversation_id, answers=answers)
    db.add_message(
        conversation_id, role="user", kind="answers", content=answer_text, payload={"answers": answers}
    )

    brief = f"{conversation['brief']}\n\nAdditional detail from user:\n{answer_text}"
    db.update_conversation(conversation_id, brief=brief, stage="running")
    conversation = db.get_conversation(conversation_id) or conversation
    async for event in _clarify_round_stream(conversation, brief):
        yield event


async def skip_clarification_stream(conversation_id: str) -> AsyncIterator[StreamEvent]:
    """User chose 'looks good, continue' — jump straight to the final stages."""
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise KeyError(conversation_id)
    db.supersede_question_messages(conversation_id)
    db.add_message(conversation_id, role="user", kind="text", content="Skip the questions — use your best judgment.")
    db.update_conversation(conversation_id, stage="running")
    conversation = db.get_conversation(conversation_id) or conversation
    async for event in _finalize_stream(conversation, conversation["brief"]):
        yield event


async def start(conversation_id: str, raw_idea: str) -> dict[str, Any]:
    """Blocking form of :func:`start_stream`."""
    return await _drain(conversation_id, start_stream(conversation_id, raw_idea))


async def submit_answers(conversation_id: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Blocking form of :func:`submit_answers_stream`."""
    return await _drain(conversation_id, submit_answers_stream(conversation_id, answers))


async def skip_clarification(conversation_id: str) -> dict[str, Any]:
    """Blocking form of :func:`skip_clarification_stream`."""
    return await _drain(conversation_id, skip_clarification_stream(conversation_id))