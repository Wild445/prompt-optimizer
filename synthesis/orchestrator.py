"""Chains the prompt-synthesis stages into one pipeline run.

Stages 1, 3, 4, 5, 6 are single-shot and chain automatically. Stage 2
(Clarify) is interactive by nature, so the pipeline pauses there and hands
control back to the caller via ``answer_fn`` — a callback that receives the
clarifying questions and returns the user's answers as free text. Pass
``max_clarify_rounds=0`` to skip clarification entirely.

This is the batch/CLI entry point: it blocks on ``answer_fn`` until the user
replies. The chat UI does not use it — ``webapp.runner`` drives the same stages
as a resumable state machine so each HTTP request advances one step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from synthesis import service

# Receives the clarifying questions as ``{"question", "options", "multi_select"}``
# dicts (see ``service.clarify``) and returns the user's answers as free text.
AnswerFn = Callable[[list[dict[str, Any]]], Awaitable[str]]


@dataclass
class PipelineResult:
    """Every intermediate artifact from one end-to-end run, for inspection/eval."""

    raw_idea: str
    built_prompt: str = ""
    clarified_brief: str = ""
    clarify_rounds: int = 0
    plan: str = ""
    optimized_prompt: str = ""
    model_tuned_prompt: str = ""
    humanized_output: str = ""
    target_model: Optional[str] = None
    trace: list[str] = field(default_factory=list)


async def run_pipeline(
    raw_idea: str,
    *,
    audience: str = "a general-purpose LLM",
    target_model: Optional[str] = None,
    model_notes: str = "",
    humanize_output: bool = False,
    answer_fn: Optional[AnswerFn] = None,
    max_clarify_rounds: int = 3,
) -> PipelineResult:
    """Run Builder -> Clarify -> Planner -> Optimizer -> Model-tuning [-> Humanizer].

    ``answer_fn`` is required if clarification may trigger (``max_clarify_rounds > 0``);
    it is awaited with the model's questions and must return the user's reply as text.
    Without it, clarification is skipped and the built prompt is used as-is.

    ``model_notes`` is passed to the model-tuning stage as authoritative reference
    material about ``target_model``; see ``service.tune_for_model``.
    """
    result = PipelineResult(raw_idea=raw_idea, target_model=target_model)

    built = await service.build_prompt(raw_idea, audience=audience)
    result.built_prompt = built.content
    result.trace.append("build_prompt")

    brief = built.content
    if max_clarify_rounds > 0 and answer_fn is not None:
        for _ in range(max_clarify_rounds):
            outcome = await service.clarify(brief)
            result.clarify_rounds += 1
            if outcome.get("status") == "ready" or not outcome.get("questions"):
                brief = outcome.get("clarified_summary") or brief
                break
            answers = await answer_fn(outcome["questions"])
            brief = f"{outcome.get('clarified_summary', brief)}\n\nAdditional detail from user:\n{answers}"
        result.trace.append(f"clarify x{result.clarify_rounds}")
    result.clarified_brief = brief

    planned = await service.plan_workflow(brief)
    result.plan = planned.content
    result.trace.append("plan_workflow")

    optimized = await service.optimize_prompt(brief)
    result.optimized_prompt = optimized.content
    result.trace.append("optimize_prompt")

    current = optimized.content
    if target_model:
        tuned = await service.tune_for_model(current, target_model, model_notes=model_notes)
        result.model_tuned_prompt = tuned.content
        current = tuned.content
        result.trace.append("tune_for_model")

    if humanize_output:
        humanized = await service.humanize(current)
        result.humanized_output = humanized.content
        result.trace.append("humanize")

    return result
