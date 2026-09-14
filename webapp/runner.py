"""Drives the six synthesis stages as a resumable, DB-backed state machine.

``synthesis.orchestrator.run_pipeline`` blocks on an ``answer_fn`` callback,
which suits a CLI but not a chat UI: the clarification pause has to survive
across HTTP requests, page reloads, and conversation switches. So this module
runs the same stages, persisting after each one, with ``conversations.stage`` as
the resume point.

Stages: ``awaiting_idea`` -> ``clarifying`` -> ``complete``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthesis import service
from webapp import db

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"

_STAGE_LABELS = {
    "build_prompt": "Built the initial prompt draft",
    "plan_workflow": "Planned the workflow",
    "optimize_prompt": "Optimized the prompt",
    "tune_for_model": "Tuned for the target model",
    "humanize": "Humanized the wording",
}


def _record(conversation_id: str, stage: str, completion: Any) -> str:
    """Persist telemetry for one stage and hand back its text."""
    db.record_stage_run(conversation_id, stage, completion)
    return completion.content


def _emit_stage(conversation_id: str, stage: str, content: str) -> dict[str, Any]:
    """Write one intermediate artifact into the transcript as a collapsible card."""
    return db.add_message(
        conversation_id,
        role="assistant",
        kind="stage",
        content=content,
        payload={"stage": stage, "label": _STAGE_LABELS.get(stage, stage)},
    )


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


async def _clarify_round(conversation: dict[str, Any], brief: str) -> dict[str, Any]:
    """Run one Clarify call; either post an MCQ card or move on to the final stages.

    Returns the frontend-visible state for the conversation.
    """
    conversation_id = conversation["id"]
    rounds_used = conversation["clarify_rounds"]
    max_rounds = conversation["max_clarify_rounds"]

    if rounds_used >= max_rounds:
        return await _finalize(conversation, brief)

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
        return await _finalize(conversation, summary)

    db.supersede_question_messages(conversation_id)
    db.add_message(
        conversation_id,
        role="assistant",
        kind="questions",
        content="A few things would change the output. Pick an option or write your own.",
        payload={"questions": questions, "round": rounds_used, "max_rounds": max_rounds},
    )
    db.update_conversation(conversation_id, stage="clarifying")
    return _state(conversation_id)


async def _finalize(conversation: dict[str, Any], brief: str) -> dict[str, Any]:
    """Run planner -> optimizer -> [model tuning] -> [humanizer], then save the result."""
    conversation_id = conversation["id"]
    db.supersede_question_messages(conversation_id)

    planned = await service.plan_workflow(brief)
    plan = _record(conversation_id, "plan_workflow", planned)
    _emit_stage(conversation_id, "plan_workflow", plan)

    optimized = await service.optimize_prompt(brief)
    optimized_prompt = _record(conversation_id, "optimize_prompt", optimized)
    _emit_stage(conversation_id, "optimize_prompt", optimized_prompt)

    final_prompt = optimized_prompt
    target_model = (conversation["target_model"] or "").strip()
    if target_model:
        tuned = await service.tune_for_model(
            final_prompt, target_model, model_notes=conversation["model_notes"] or ""
        )
        final_prompt = _record(conversation_id, "tune_for_model", tuned)
        _emit_stage(conversation_id, "tune_for_model", final_prompt)

    if conversation["humanize"]:
        humanized = await service.humanize(final_prompt)
        final_prompt = _record(conversation_id, "humanize", humanized)
        _emit_stage(conversation_id, "humanize", final_prompt)

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
    return _state(conversation_id)


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
    }


async def start(conversation_id: str, raw_idea: str) -> dict[str, Any]:
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

    built = await service.build_prompt(raw_idea, audience=conversation["audience"])
    built_prompt = _record(conversation_id, "build_prompt", built)
    _emit_stage(conversation_id, "build_prompt", built_prompt)
    db.update_conversation(conversation_id, built_prompt=built_prompt, brief=built_prompt)

    conversation = db.get_conversation(conversation_id) or conversation
    return await _clarify_round(conversation, built_prompt)


async def submit_answers(conversation_id: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
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
    return await _clarify_round(conversation, brief)


async def skip_clarification(conversation_id: str) -> dict[str, Any]:
    """User chose 'looks good, continue' — jump straight to the final stages."""
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise KeyError(conversation_id)
    db.supersede_question_messages(conversation_id)
    db.add_message(conversation_id, role="user", kind="text", content="Skip the questions — use your best judgment.")
    db.update_conversation(conversation_id, stage="running")
    conversation = db.get_conversation(conversation_id) or conversation
    return await _finalize(conversation, conversation["brief"])
