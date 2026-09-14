"""Independent, callable steps of the prompt-synthesis pipeline.

Each function is a thin wrapper around ``runtime.complete_prompt`` targeting
one ``.prompty`` file under ``prompts/``. They can be called on their own
(menu-style) or composed by ``synthesis.orchestrator``. No LLM-calling logic
lives here — that stays centralized in ``runtime.py``.

The user-supplied payload for every stage is passed as ``extra_messages``, not
as a templated ``.prompty`` input, per ``runtime.merge_history``: untrusted text
must never reach PromptyChatParser, which would otherwise treat a line like
``system:`` inside a pasted prompt as a new role block. Only short, trusted
knobs (``audience``, ``max_questions``, ``target_model``, ``model_notes``) are
templated into the system message.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from runtime import PromptCompletion, complete_prompt

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


async def build_prompt(raw_idea: str, audience: str = "a general-purpose LLM") -> PromptCompletion:
    """Stage 1 — turn a rough idea into a structured prompt draft."""
    return await complete_prompt(
        _PROMPTS_DIR / "01_prompt_builder.prompty",
        inputs={"audience": audience},
        extra_messages=[{"role": "user", "content": raw_idea}],
    )


def _normalize_questions(raw: Any) -> list[dict[str, Any]]:
    """Coerce whatever came back into ``[{question, options, multi_select}]``.

    The strict json_schema in ``02_clarify.prompty`` already guarantees this
    shape, but the schema is downgradeable to ``json_object`` on older API
    versions — and a bare string list is what that downgrade tends to produce.
    Normalizing here keeps every caller (CLI and UI) on one shape.
    """
    questions: list[dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, str):
            questions.append({"question": item, "options": [], "multi_select": False})
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("question") or item.get("text") or "").strip()
        if not text:
            continue
        options = [str(option).strip() for option in (item.get("options") or []) if str(option).strip()]
        questions.append(
            {"question": text, "options": options, "multi_select": bool(item.get("multi_select"))}
        )
    return questions


async def clarify(draft: str, max_questions: int = 4) -> dict[str, Any]:
    """Stage 2 — ask clarifying questions, or signal the draft is ready.

    Returns ``{"status", "questions", "clarified_summary"}``, where each question
    is ``{"question": str, "options": [str], "multi_select": bool}`` so a UI can
    render it as multiple choice. Call again with an updated ``draft`` (the
    summary plus the user's answers folded in) until ``status`` is ``"ready"``.
    """
    completion = await complete_prompt(
        _PROMPTS_DIR / "02_clarify.prompty",
        inputs={"max_questions": max_questions},
        extra_messages=[{"role": "user", "content": draft}],
    )
    try:
        body = json.loads(completion.content)
    except json.JSONDecodeError:
        # Fall back to a conservative "needs another pass" shape rather than raising,
        # so a caller looping on `status` doesn't crash on an off-spec response.
        return {
            "status": "needs_clarification",
            "questions": [],
            "clarified_summary": completion.content,
            "completion": completion,
        }
    questions = _normalize_questions(body.get("questions"))
    status = body.get("status") or ("needs_clarification" if questions else "ready")
    return {
        "status": status,
        "questions": questions,
        "clarified_summary": body.get("clarified_summary") or "",
        "completion": completion,
    }


async def plan_workflow(brief: str) -> PromptCompletion:
    """Stage 3 — break a clarified brief into an ordered plan of steps."""
    return await complete_prompt(
        _PROMPTS_DIR / "03_workflow_planner.prompty",
        extra_messages=[{"role": "user", "content": brief}],
    )


async def optimize_prompt(prompt_draft: str) -> PromptCompletion:
    """Stage 4 — tighten a prompt for stronger reasoning / cleaner output."""
    return await complete_prompt(
        _PROMPTS_DIR / "04_prompt_optimizer.prompty",
        extra_messages=[{"role": "user", "content": prompt_draft}],
    )


async def tune_for_model(prompt_draft: str, target_model: str, model_notes: str = "") -> PromptCompletion:
    """Stage 5 — adapt a prompt to one target model's conventions/quirks.

    ``model_notes`` is authoritative reference material about ``target_model``
    (its prompting guide, supported parameters, known quirks). Supply it for any
    model newer than the calling deployment's training cutoff — without it the
    stage stays deliberately generic rather than inventing conventions.
    """
    return await complete_prompt(
        _PROMPTS_DIR / "05_model_specific_prompting.prompty",
        inputs={"target_model": target_model, "model_notes": model_notes or "(none supplied)"},
        extra_messages=[{"role": "user", "content": prompt_draft}],
    )


async def humanize(content: str) -> PromptCompletion:
    """Stage 6 — strip AI-sounding phrasing/patterns from a draft."""
    return await complete_prompt(
        _PROMPTS_DIR / "06_ai_humanizer.prompty",
        extra_messages=[{"role": "user", "content": content}],
    )
