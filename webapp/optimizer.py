"""Drives the prompt-optimization loop as a resumable, DB-backed state machine.

Same shape as ``webapp/runner.py``, and for the same reason: every step here
pauses on the user (describe the failures, edit the criteria, upload the data,
disagree with the judge), and those pauses have to survive page reloads and
session switches. ``optimizations.stage`` is the resume point.

The loop::

    awaiting_prompt        user pastes the prompt to optimize -> judge scaffold
    awaiting_observations  user says how it fails             -> drafted criteria
    reviewing_criteria     user edits/adds criteria           -> judge prompt
    awaiting_dataset       user uploads the test cases
    ready_to_run           -> run the prompt, then the judge, over every case
    reviewing_results      user flags the verdicts they disagree with
                           -> change list
    reviewing_changes      user approves or rejects each proposed change
                           -> revised prompt built from the approved ones only
    awaiting_iteration     user decides whether to run again  -> ready_to_run

Every entry point is an async generator of progress events, consumed by the UI
over NDJSON. ``case_done`` events carry the running pass/fail tally so the panel
can count up while the run is still going.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from chat_common.common.logging import logger
from optimization import service
from optimization.dataset import TEMPLATE_KINDS, template_variables, to_csv
from webapp import opt_db

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "optimizations"

#: How many test cases are in flight at once. The judge call and the response
#: call are both network-bound, so a handful of them overlapping turns a
#: fifty-case run from minutes into seconds — but the deployment is shared, so
#: this stays low enough not to soak up someone else's rate limit.
CONCURRENCY = max(1, int(os.getenv("PROMPT_OPTIMIZER_CONCURRENCY", "4")))

StreamEvent = dict[str, Any]

STEP_ORDER = [
    "scaffold",
    "criteria",
    "judge",
    "dataset",
    "respond",
    "evaluate",
    "review",
    "analyze",
    "approve",
    "revise",
]

STEP_INFO = {
    "scaffold": {
        "title": "Judge scaffold",
        "description": "Builds the skeleton of the LLM-as-a-judge prompt from the prompt you want to optimize.",
    },
    "criteria": {
        "title": "Success criteria",
        "description": "Turns your observations about how the prompt fails into criteria a judge can score true or false.",
    },
    "judge": {
        "title": "Build the judge",
        "description": "Consolidates your criteria with the drafted ones and writes the finished judge prompt.",
    },
    "dataset": {
        "title": "Test cases",
        "description": "Your uploaded workbook of message_id / input_payload / other_input_params rows.",
    },
    "respond": {
        "title": "Generate responses",
        "description": "Runs the current prompt over every test case and stores what came back.",
    },
    "evaluate": {
        "title": "Judge the responses",
        "description": "Scores each response against every success criterion. One false makes the case fail.",
    },
    "review": {
        "title": "Your review",
        "description": "You flag the verdicts you disagree with and say why. Everything unflagged counts as agreed.",
    },
    "analyze": {
        "title": "Analyze failures",
        "description": "Reads the run and your remarks into a list of changes, plus any criteria the judge is missing.",
    },
    "approve": {
        "title": "Approve the changes",
        "description": "You approve or reject each proposed edit. Only the ones you approve reach the revision.",
    },
    "revise": {
        "title": "Revise the prompt",
        "description": "Applies the approved changes and produces the next version of the prompt.",
    },
}

_STEP_LABELS = {key: info["title"] for key, info in STEP_INFO.items()}

#: Stages where the loop is waiting on the user rather than on a model.
_USER_STAGES = {
    "awaiting_prompt": "prompt",
    "awaiting_observations": "criteria",
    "reviewing_criteria": "criteria",
    "awaiting_dataset": "dataset",
    "ready_to_run": "respond",
    "reviewing_results": "review",
    "reviewing_changes": "approve",
    "awaiting_iteration": "revise",
}


class OptimizerError(RuntimeError):
    """A step the user can act on, surfaced as a card rather than a stack trace."""


# --------------------------------------------------------------------------- progress


def steps_snapshot(optimization_id: str, active: Optional[str] = None) -> list[dict[str, Any]]:
    """Per-step progress for the right-hand panel: ``done`` / ``active`` / ``pending``.

    Completion comes from ``optimization_runs`` and from the rows that exist, not
    from anything held in memory, so the panel is still right after a reload. The
    loop is cyclic, so on iteration 2 and beyond the earlier steps stay ``done``
    while ``respond``/``evaluate`` go active again.
    """
    optimization = opt_db.get_optimization(optimization_id)
    if optimization is None:
        return []
    completed = opt_db.completed_steps(optimization_id)
    stage = optimization["stage"]
    waiting_step = _USER_STAGES.get(stage)
    has_cases = bool(opt_db.list_cases(optimization_id))
    scoreboard = opt_db.iteration_scoreboard(optimization_id, optimization["iteration"])

    steps: list[dict[str, Any]] = []
    for key in STEP_ORDER:
        if key == "dataset":
            done = has_cases
        elif key == "review":
            done = stage in ("reviewing_changes", "awaiting_iteration", "complete")
        elif key == "approve":
            # No LLM call stands behind this step, so completion is the stage
            # having moved past the approval card rather than a recorded run.
            done = stage in ("awaiting_iteration", "complete")
        elif key in ("respond", "evaluate"):
            done = scoreboard["total"] > 0 and scoreboard["total"] == scoreboard["passed"] + scoreboard["failed"] + scoreboard["errored"]
        else:
            done = key in completed

        waiting = key == waiting_step and not done and key != active
        if key == active or waiting:
            status = "active"
        elif done:
            status = "done"
        else:
            status = "pending"
        steps.append(
            {"key": key, "title": STEP_INFO[key]["title"], "status": status, "waiting": waiting}
        )
    return steps


def _progress(optimization_id: str, event_type: str, step: str, **extra: Any) -> StreamEvent:
    """Build a progress event carrying a fresh snapshot of every step."""
    active = step if event_type == "step_start" else None
    return {
        "type": event_type,
        "step": step,
        "label": _STEP_LABELS.get(step, step),
        "title": STEP_INFO.get(step, {}).get("title", step),
        "steps": steps_snapshot(optimization_id, active=active),
        **extra,
    }


def _emit(optimization_id: str, step: str, content: str, **payload: Any) -> dict[str, Any]:
    """Write one intermediate artifact into the transcript as a collapsible card."""
    return opt_db.add_message(
        optimization_id,
        role="assistant",
        kind="step",
        content=content,
        payload={"step": step, "label": _STEP_LABELS.get(step, step), **payload},
    )


def _state(optimization_id: str) -> dict[str, Any]:
    """The payload every mutating endpoint returns: session + transcript + results."""
    optimization = opt_db.get_optimization(optimization_id)
    iteration = optimization["iteration"] if optimization else 0
    return {
        "optimization": optimization,
        "messages": opt_db.list_messages(optimization_id),
        "criteria": opt_db.list_criteria(optimization_id),
        "cases": opt_db.list_cases(optimization_id),
        "results": opt_db.list_results(optimization_id, iteration),
        "scoreboard": opt_db.iteration_scoreboard(optimization_id, iteration),
        "prompt_versions": opt_db.list_prompt_versions(optimization_id),
        "usage": opt_db.usage_summary(optimization_id),
        "steps": steps_snapshot(optimization_id),
        "template_kinds": list(TEMPLATE_KINDS),
    }


def _require(optimization_id: str) -> dict[str, Any]:
    optimization = opt_db.get_optimization(optimization_id)
    if optimization is None:
        raise KeyError(optimization_id)
    return optimization


async def _drain(optimization_id: str, events: AsyncIterator[StreamEvent]) -> dict[str, Any]:
    """Run a ``*_stream`` generator to completion and return the resulting state."""
    async for _event in events:
        pass
    return _state(optimization_id)


# --------------------------------------------------------------------------- output files


def _output_dir(optimization_id: str) -> Path:
    directory = OUTPUT_ROOT / optimization_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write(optimization_id: str, name: str, text: str) -> Path:
    path = _output_dir(optimization_id) / name
    path.write_text(text, encoding="utf-8")
    return path


_RESULT_COLUMNS = [
    "message_id",
    "input_payload",
    "other_input_params",
    "prompt_version",
    "response",
    "judge_verdict",
    "failed_criteria",
    "reason_for_failure",
    "user_agrees",
    "user_reason",
    "status",
    "error",
]


def _result_rows(optimization_id: str, iteration: int) -> list[dict[str, Any]]:
    """Flatten stored results into the wide table the user reads in the output folder."""
    rows = []
    for result in opt_db.list_results(optimization_id, iteration):
        reasons = "; ".join(
            f"{failure.get('id', '?')}: {failure.get('why', '')}"
            f" [prompt: {failure.get('prompt_section', '')}]"
            for failure in result["failures"]
        )
        rows.append(
            {
                "message_id": result["message_id"],
                "input_payload": result["input_payload"],
                "other_input_params": json.dumps(result["other_input_params"]),
                "prompt_version": result["prompt_version"],
                "response": result["response"],
                "judge_verdict": "" if result["verdict"] is None else bool(result["verdict"]),
                "failed_criteria": ", ".join(result["failed_criteria"]),
                "reason_for_failure": reasons,
                "user_agrees": bool(result["user_agrees"]),
                "user_reason": result["user_reason"],
                "status": result["status"],
                "error": result["error"],
            }
        )
    return rows


def _save_iteration(optimization_id: str, iteration: int) -> Path:
    """Write this iteration's result table next to the prompt versions."""
    rows = _result_rows(optimization_id, iteration)
    return _write(optimization_id, f"iteration_{iteration}_results.csv", to_csv(rows, _RESULT_COLUMNS))


def _save_session(optimization_id: str) -> None:
    """Snapshot the whole session as JSON, the way the chat pipeline snapshots transcripts."""
    _write(
        optimization_id,
        "session.json",
        json.dumps(
            {
                "optimization_id": optimization_id,
                "optimization": opt_db.get_optimization(optimization_id),
                "criteria": opt_db.list_criteria(optimization_id),
                "prompt_versions": opt_db.list_prompt_versions(optimization_id),
                "cases": opt_db.list_cases(optimization_id),
                "results": opt_db.list_results(optimization_id),
                "usage": opt_db.usage_summary(optimization_id),
            },
            indent=2,
            default=str,
        ),
    )


# --------------------------------------------------------------------------- step 1-2: prompt -> scaffold


async def submit_prompt_stream(
    optimization_id: str, prompt: str, template_kind: str
) -> AsyncIterator[StreamEvent]:
    """Store the prompt to optimize, then build the judge scaffold around it."""
    optimization = _require(optimization_id)
    prompt = prompt.strip()
    if not prompt:
        raise OptimizerError("Paste the prompt you want to optimize first.")
    if template_kind not in TEMPLATE_KINDS:
        raise OptimizerError(f"Unknown template type {template_kind!r}. Pick one of: {', '.join(TEMPLATE_KINDS)}.")

    opt_db.add_message(optimization_id, role="user", kind="prompt", content=prompt)
    if optimization["title"] in ("", "New optimization"):
        opt_db.update_optimization(optimization_id, title=prompt.strip().splitlines()[0][:60] or "New optimization")
    opt_db.update_optimization(
        optimization_id,
        base_prompt=prompt,
        current_prompt=prompt,
        template_kind=template_kind,
        prompt_version=1,
        stage="running",
    )
    opt_db.add_prompt_version(optimization_id, 1, prompt, change_notes="Original prompt as supplied.")
    _write(optimization_id, "prompt_v1.md", prompt)

    yield _progress(optimization_id, "step_start", "scaffold")
    completion = await service.build_judge_scaffold(prompt)
    opt_db.record_run(optimization_id, "scaffold", completion)
    opt_db.update_optimization(optimization_id, judge_scaffold=completion.content)
    _write(optimization_id, "judge_scaffold.md", completion.content)
    _emit(optimization_id, "scaffold", completion.content)
    yield _progress(optimization_id, "step_done", "scaffold")

    variables = template_variables(prompt, template_kind)
    opt_db.supersede_kind(optimization_id, "observations_form")
    opt_db.add_message(
        optimization_id,
        role="assistant",
        kind="observations_form",
        content=(
            "The judge scaffold is ready. Now describe what the current prompt gets wrong —"
            " one problem per line is ideal. These become the success criteria the judge scores against."
        ),
        payload={"template_variables": variables},
    )
    opt_db.update_optimization(optimization_id, stage="awaiting_observations")


# --------------------------------------------------------------------------- step 3-4: observations -> criteria


async def submit_observations_stream(optimization_id: str, observations: str) -> AsyncIterator[StreamEvent]:
    """Turn the user's complaints into a draft criteria list for them to review."""
    optimization = _require(optimization_id)
    observations = observations.strip()
    if not observations:
        raise OptimizerError("Describe at least one problem with the current prompt.")

    opt_db.supersede_kind(optimization_id, "observations_form")
    opt_db.add_message(optimization_id, role="user", kind="text", content=observations)
    opt_db.update_optimization(optimization_id, observations=observations, stage="running")

    yield _progress(optimization_id, "step_start", "criteria")
    outcome = await service.draft_criteria(optimization["current_prompt"], observations)
    opt_db.record_run(optimization_id, "criteria", outcome.get("completion"))
    criteria = service.assign_ids(outcome["criteria"])
    if not criteria:
        logger.warning(
            "Criteria drafter returned nothing usable; falling back to a user-authored list",
            extra={"optimization_id": optimization_id},
        )
    opt_db.replace_criteria(optimization_id, criteria, iteration=optimization["iteration"])
    yield _progress(optimization_id, "step_done", "criteria")

    opt_db.supersede_kind(optimization_id, "criteria_form")
    opt_db.add_message(
        optimization_id,
        role="assistant",
        kind="criteria_form",
        content=(
            "These are the success criteria drawn from your observations. Edit or remove any of them,"
            " and add anything that is missing — the final list is what the judge will score against."
            if criteria
            else "Nothing usable came back from the drafter. Add the success criteria yourself below."
        ),
        payload={"criteria": criteria, "error": outcome.get("error", "")},
    )
    opt_db.update_optimization(optimization_id, stage="reviewing_criteria")


# --------------------------------------------------------------------------- step 5: criteria -> judge prompt


async def submit_criteria_stream(
    optimization_id: str, criteria: list[dict[str, Any]], additions: str = ""
) -> AsyncIterator[StreamEvent]:
    """Consolidate the criteria and assemble the finished judge prompt."""
    optimization = _require(optimization_id)

    kept = [
        {
            "title": str(item.get("title") or "").strip(),
            "description": str(item.get("description") or "").strip(),
            "origin": str(item.get("origin") or "agent"),
        }
        for item in criteria
        if str(item.get("title") or item.get("description") or "").strip()
    ]
    # Free-text additions are one criterion per line; the consolidator sharpens
    # them into checkable wording, so a rough phrasing here is fine.
    user_added = [
        {"title": line.strip()[:120], "description": line.strip(), "origin": "user"}
        for line in (additions or "").splitlines()
        if line.strip()
    ]
    if not kept and not user_added:
        raise OptimizerError("The judge needs at least one success criterion.")

    opt_db.supersede_kind(optimization_id, "criteria_form", criteria=kept, additions=additions)
    opt_db.update_optimization(optimization_id, stage="running")

    yield _progress(optimization_id, "step_start", "judge")
    merged = await service.consolidate_criteria(optimization["current_prompt"], kept, user_added)
    opt_db.record_run(optimization_id, "judge", merged.get("completion"))
    final_criteria = merged["criteria"]
    opt_db.replace_criteria(optimization_id, final_criteria, iteration=optimization["iteration"])

    # Substituted in Python, not written by a model: the criteria the judge
    # scores against are the consolidated list above, word for word.
    judge_prompt = service.build_judge_prompt(optimization["judge_scaffold"], final_criteria)
    opt_db.update_optimization(optimization_id, judge_prompt=judge_prompt)
    _write(optimization_id, "judge_prompt.md", judge_prompt)
    _write(optimization_id, "criteria.json", json.dumps(final_criteria, indent=2))
    _emit(optimization_id, "judge", judge_prompt, criteria=final_criteria)
    yield _progress(optimization_id, "step_done", "judge")

    _post_dataset_form(optimization_id)


def _post_dataset_form(optimization_id: str) -> None:
    """Ask for the test-case workbook, naming the variables the prompt needs filled."""
    optimization = _require(optimization_id)
    opt_db.supersede_kind(optimization_id, "dataset_form")
    opt_db.add_message(
        optimization_id,
        role="assistant",
        kind="dataset_form",
        content=(
            "The judge is ready. Upload the test cases as .xlsx or .csv with the columns"
            " message_id, input_payload, other_input_params."
        ),
        payload={
            "template_kind": optimization["template_kind"],
            "template_variables": template_variables(optimization["current_prompt"], optimization["template_kind"]),
        },
    )
    opt_db.update_optimization(optimization_id, stage="awaiting_dataset")


# --------------------------------------------------------------------------- step 6: dataset


def store_dataset(optimization_id: str, cases: list[dict[str, Any]], filename: str) -> dict[str, Any]:
    """Replace the test cases with a freshly uploaded workbook.

    Not a generator: parsing is synchronous and there is no model call to report
    progress on. Uploading again at any point is allowed and simply re-arms the
    loop at iteration 1 — old results graded data that no longer exists.
    """
    optimization = _require(optimization_id)
    opt_db.replace_cases(optimization_id, cases)
    opt_db.update_optimization(optimization_id, dataset_name=filename, iteration=0, stage="ready_to_run")
    opt_db.supersede_kind(optimization_id, "dataset_form", filename=filename, case_count=len(cases))
    opt_db.add_message(
        optimization_id,
        role="user",
        kind="dataset",
        content=f"Uploaded {len(cases)} test case(s) from {filename}.",
        payload={"filename": filename, "case_count": len(cases)},
    )
    missing = _missing_variables(optimization, cases)
    opt_db.add_message(
        optimization_id,
        role="assistant",
        kind="run_form",
        content=(
            f"{len(cases)} test cases loaded. Running the current prompt over all of them,"
            " then scoring each response against the success criteria."
        ),
        payload={"case_count": len(cases), "missing_variables": missing},
    )
    return _state(optimization_id)


def _missing_variables(optimization: dict[str, Any], cases: list[dict[str, Any]]) -> list[str]:
    """Template variables at least one row leaves out, so the user hears about it up front.

    Per-row rather than across the whole file: a half-filled column is the usual
    mistake, and under ``jinja2`` every row that omits a variable fails outright.
    Still a warning and not a rejection — ``langchain`` renders a missing value
    as empty, and the user may know something this check does not.
    """
    needed = set(template_variables(optimization["current_prompt"], optimization["template_kind"]))
    if not needed:
        return []
    missing: set[str] = set()
    for case in cases:
        missing.update(needed - set(case.get("other_input_params") or {}))
    return sorted(missing)


# --------------------------------------------------------------------------- steps 7-8: respond + judge


async def _process_case(
    optimization_id: str,
    case: dict[str, Any],
    iteration: int,
    prompt: str,
    template_kind: str,
    judge_prompt: str,
    prompt_version: int,
) -> dict[str, Any]:
    """Run one test case end to end: render, respond, judge, persist.

    Never raises: a case that blows up is recorded with ``status='error'`` and
    counted separately, because one malformed row must not abandon a run the
    user is watching count up.
    """
    result_id = opt_db.start_result(optimization_id, iteration, case, prompt_version)
    try:
        completion = await service.run_prompt_under_test(
            prompt, template_kind, case["input_payload"], case["other_input_params"]
        )
        opt_db.record_run(optimization_id, "respond", completion, iteration=iteration)
        opt_db.update_result(result_id, response=completion.content, status="responded")

        verdict = await service.judge_response(
            judge_prompt, prompt, case["input_payload"], completion.content
        )
        opt_db.record_run(optimization_id, "evaluate", verdict.get("completion"), iteration=iteration)
        opt_db.update_result(
            result_id,
            verdict=1 if verdict["passed"] else 0,
            result_label=verdict["result"],
            failed_criteria=verdict["failed_criteria"],
            failures=verdict["failures"],
            judge_raw=verdict["raw"],
            status="judged",
        )
        return {
            "message_id": case["message_id"],
            "passed": verdict["passed"],
            "failed_criteria": verdict["failed_criteria"],
            "status": "judged",
        }
    except Exception as error:  # noqa: BLE001 - recorded per case, logged in full
        logger.error(
            "Test case failed",
            extra={"optimization_id": optimization_id, "message_id": case["message_id"], "error": str(error)},
        )
        opt_db.update_result(result_id, status="error", error=f"{type(error).__name__}: {error}")
        return {"message_id": case["message_id"], "passed": None, "status": "error", "error": str(error)}


async def _run_cases(
    optimization_id: str,
    cases: list[dict[str, Any]],
    iteration: int,
    prompt: str,
    template_kind: str,
    judge_prompt: str,
    prompt_version: int,
) -> AsyncIterator[StreamEvent]:
    """Fan the test cases out over :data:`CONCURRENCY` workers, yielding as each lands.

    The workers push onto a queue and this generator drains it, which is what lets
    the panel tick up while the run is still going. If the consumer goes away
    (the browser disconnected), the ``finally`` cancels whatever is still in
    flight rather than leaving orphaned calls running against the deployment.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    done = object()

    async def worker(case: dict[str, Any]) -> None:
        async with semaphore:
            outcome = await _process_case(
                optimization_id, case, iteration, prompt, template_kind, judge_prompt, prompt_version
            )
        await queue.put(outcome)
        await queue.put(done)

    tasks = [asyncio.create_task(worker(case)) for case in cases]
    remaining = len(tasks)
    try:
        while remaining:
            item = await queue.get()
            if item is done:
                remaining -= 1
                continue
            yield _progress(
                optimization_id,
                "case_done",
                "evaluate",
                case=item,
                scoreboard=opt_db.iteration_scoreboard(optimization_id, iteration),
            )
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def run_iteration_stream(optimization_id: str) -> AsyncIterator[StreamEvent]:
    """Run the current prompt over every test case and score the results."""
    optimization = _require(optimization_id)
    cases = opt_db.list_cases(optimization_id)
    if not cases:
        raise OptimizerError("Upload the test cases before running an iteration.")
    if not optimization["judge_prompt"].strip():
        raise OptimizerError("The judge prompt has not been built yet.")

    iteration = optimization["iteration"] + 1
    prompt_version = optimization["prompt_version"]
    opt_db.update_optimization(optimization_id, iteration=iteration, stage="running")

    yield _progress(optimization_id, "step_start", "respond", iteration=iteration, total=len(cases))
    yield _progress(
        optimization_id,
        "case_start",
        "evaluate",
        iteration=iteration,
        total=len(cases),
        scoreboard={"total": 0, "passed": 0, "failed": 0, "errored": 0},
    )

    async for event in _run_cases(
        optimization_id,
        cases,
        iteration,
        optimization["current_prompt"],
        optimization["template_kind"],
        optimization["judge_prompt"],
        prompt_version,
    ):
        yield event

    scoreboard = opt_db.iteration_scoreboard(optimization_id, iteration)
    output_path = _save_iteration(optimization_id, iteration)
    opt_db.update_optimization(optimization_id, output_path=str(output_path), stage="reviewing_results")
    yield _progress(optimization_id, "step_done", "evaluate", scoreboard=scoreboard)

    # Retired only now that the run has landed: a run that dies half way leaves
    # its "run it" card open, so _record_optimizer_failure puts the user back on
    # the button they pressed instead of somewhere earlier in the loop.
    opt_db.supersede_kind(optimization_id, "run_form")
    opt_db.supersede_kind(optimization_id, "iteration_form")
    opt_db.supersede_kind(optimization_id, "results_form")
    opt_db.add_message(
        optimization_id,
        role="assistant",
        kind="results_form",
        content=(
            f"Iteration {iteration}: {scoreboard['passed']} of {scoreboard['total']} test cases passed."
            " Flag any verdict you disagree with and say why; everything you leave alone counts as agreed."
        ),
        payload={
            "iteration": iteration,
            "scoreboard": scoreboard,
            "output_path": str(output_path),
        },
    )
    _save_session(optimization_id)


# --------------------------------------------------------------------------- steps 9-11: review -> analyze -> revise


def _run_report(optimization_id: str, iteration: int) -> str:
    """Render the graded run for the analyst, user remarks included."""
    blocks: list[str] = []
    for result in opt_db.list_results(optimization_id, iteration):
        lines = [
            f"### Test case {result['message_id']}",
            f"Input: {result['input_payload']}",
            f"Response: {result['response'] or '(none)'}",
        ]
        if result["status"] == "error":
            lines.append(f"Judge verdict: not evaluated ({result['error']})")
        else:
            lines.append(f"Judge verdict: {result['result_label'] or ('Success' if result['verdict'] else 'Failed')}")
            if result["failed_criteria"]:
                lines.append(f"Failed criteria: {', '.join(result['failed_criteria'])}")
            for failure in result["failures"]:
                lines.append(
                    f"  - {failure.get('id', '?')}: {failure.get('why', '')}"
                    f" (prompt section: {failure.get('prompt_section', 'unknown')})"
                )
        if result["user_agrees"]:
            lines.append("User: agrees with this verdict.")
        else:
            lines.append(f"User DISAGREES with this verdict: {result['user_reason'] or '(no reason given)'}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def submit_review_stream(
    optimization_id: str, feedback: list[dict[str, Any]]
) -> AsyncIterator[StreamEvent]:
    """Record the user's agreement per case, then analyze and revise."""
    optimization = _require(optimization_id)
    iteration = optimization["iteration"]

    disputed = [item for item in feedback if not item.get("agrees", True)]
    unexplained = [item for item in disputed if not str(item.get("reason") or "").strip()]
    if unexplained:
        # The UI blocks this too, but a reason is the whole value of a disagreement:
        # without it the analyst has a vote it cannot act on.
        raise OptimizerError(
            "Flagged test cases need a reason: "
            + ", ".join(str(item.get("message_id")) for item in unexplained[:5])
        )

    opt_db.set_user_feedback(optimization_id, iteration, feedback)
    opt_db.supersede_kind(optimization_id, "results_form", feedback=feedback)
    opt_db.add_message(
        optimization_id,
        role="user",
        kind="review",
        content=(
            f"Reviewed iteration {iteration}: agreed with the judge on"
            f" {len(feedback) - len(disputed)} of {len(feedback)} test cases."
        ),
        payload={"feedback": feedback},
    )
    opt_db.update_optimization(optimization_id, stage="running")
    _save_iteration(optimization_id, iteration)

    criteria = opt_db.list_criteria(optimization_id)
    scoreboard = opt_db.iteration_scoreboard(optimization_id, iteration)

    yield _progress(optimization_id, "step_start", "analyze", iteration=iteration)
    analysis = await service.analyze_failures(
        optimization["current_prompt"], criteria, _run_report(optimization_id, iteration)
    )
    opt_db.record_run(optimization_id, "analyze", analysis.get("completion"), iteration=iteration)
    _emit(
        optimization_id,
        "analyze",
        analysis["summary"] or "(no summary returned)",
        changes=analysis["changes"],
        new_criteria=analysis["new_criteria"],
    )
    yield _progress(optimization_id, "step_done", "analyze")

    if analysis["changes"] or analysis["new_criteria"]:
        # The analyst proposed something, so the user gets the last word on it:
        # nothing touches the prompt or the judge until they have approved it.
        _post_changes_form(optimization_id, iteration, analysis, scoreboard)
        _save_session(optimization_id)
        return

    if scoreboard["failed"] == 0 and scoreboard["errored"] == 0:
        # Everything passed and the analyst found nothing to change: there is no
        # revision to make, so stop here rather than churning the prompt.
        opt_db.update_optimization(optimization_id, stage="complete")
        opt_db.add_message(
            optimization_id,
            role="assistant",
            kind="complete",
            content=f"All {scoreboard['total']} test cases passed and no changes were suggested. Nothing left to optimize.",
            payload={"scoreboard": scoreboard, "new_criteria": []},
        )
        _save_session(optimization_id)
        return

    # Cases are still failing but the analyst named no change to approve; hand the
    # revisor what there is rather than leaving the loop with nowhere to go.
    async for event in _revise_stream(optimization_id, iteration, [], 0, [], scoreboard):
        yield event


def _post_changes_form(
    optimization_id: str,
    iteration: int,
    analysis: dict[str, Any],
    scoreboard: dict[str, int],
) -> None:
    """Put the analyst's proposals in front of the user as an approve/reject card.

    The proposals live in this card's payload and nowhere else: the follow-up
    endpoint reads them back from here, so an approval survives a page reload
    exactly like every other pause in the loop.
    """
    changes = analysis["changes"]
    new_criteria = analysis["new_criteria"]
    opt_db.supersede_kind(optimization_id, "changes_form")
    opt_db.add_message(
        optimization_id,
        role="assistant",
        kind="changes_form",
        content=(
            f"{len(changes)} change{'' if len(changes) == 1 else 's'} to the prompt"
            + (
                f" and {len(new_criteria)} new success criteri{'on' if len(new_criteria) == 1 else 'a'}"
                if new_criteria
                else ""
            )
            + " came out of iteration "
            f"{iteration}. Reject anything you disagree with — only what you approve gets applied."
        ),
        payload={
            "iteration": iteration,
            "summary": analysis["summary"],
            "changes": changes,
            "new_criteria": new_criteria,
            "scoreboard": scoreboard,
        },
    )
    opt_db.update_optimization(optimization_id, stage="reviewing_changes")


# --------------------------------------------------------------------------- step 12: approve -> revise


async def submit_changes_stream(
    optimization_id: str,
    approved_changes: list[int],
    approved_criteria: list[int],
) -> AsyncIterator[StreamEvent]:
    """Apply only the proposals the user approved, then revise the prompt.

    Both arguments are positions into the lists stored on the open ``changes_form``
    card, which is where the proposals were persisted. Anything not named here is
    dropped: a rejected change never reaches the revisor, and a rejected criterion
    never reaches the judge.
    """
    optimization = _require(optimization_id)
    iteration = optimization["iteration"]

    pending = opt_db.latest_message(optimization_id, "changes_form")
    if pending is None:
        raise OptimizerError("There are no proposed changes waiting for your approval.")
    payload = pending["payload"] or {}
    proposed = payload.get("changes") or []
    proposed_criteria = payload.get("new_criteria") or []
    scoreboard = payload.get("scoreboard") or opt_db.iteration_scoreboard(optimization_id, iteration)

    change_picks = {index for index in approved_changes if 0 <= index < len(proposed)}
    criteria_picks = {index for index in approved_criteria if 0 <= index < len(proposed_criteria)}
    changes = [change for index, change in enumerate(proposed) if index in change_picks]
    new_criteria = [item for index, item in enumerate(proposed_criteria) if index in criteria_picks]
    rejected = len(proposed) - len(changes)

    opt_db.supersede_kind(
        optimization_id,
        "changes_form",
        approved_changes=sorted(change_picks),
        approved_criteria=sorted(criteria_picks),
    )
    opt_db.add_message(
        optimization_id,
        role="user",
        kind="approval",
        content=(
            f"Approved {len(changes)} of {len(proposed)} proposed change(s)"
            + (
                f" and {len(new_criteria)} of {len(proposed_criteria)} new criteri{'on' if len(proposed_criteria) == 1 else 'a'}"
                if proposed_criteria
                else ""
            )
            + "."
        ),
        payload={"approved_changes": changes, "rejected": rejected, "approved_criteria": new_criteria},
    )
    opt_db.update_optimization(optimization_id, stage="running")

    added = await _apply_new_criteria(optimization_id, new_criteria, iteration)

    if not changes:
        # Every change was rejected. The prompt is left exactly as it is — the
        # user has said the analyst was wrong — but the judge may have picked up
        # approved criteria, so another run against the same prompt is still
        # worth offering.
        _post_iteration_form(
            optimization_id,
            version=optimization["prompt_version"],
            iteration=iteration,
            scoreboard=scoreboard,
            added=added,
            content=(
                f"You rejected all {len(proposed)} proposed change(s), so the prompt stays at"
                f" v{optimization['prompt_version']}."
                + (" The criteria you approved have been added to the judge." if added else "")
                + " Run the test cases again?"
            ),
            changed=False,
        )
        _save_session(optimization_id)
        return

    async for event in _revise_stream(optimization_id, iteration, changes, rejected, added, scoreboard):
        yield event


async def _revise_stream(
    optimization_id: str,
    iteration: int,
    changes: list[dict[str, Any]],
    rejected: int,
    added: list[dict[str, Any]],
    scoreboard: dict[str, int],
) -> AsyncIterator[StreamEvent]:
    """Write the next prompt version from the approved change list."""
    optimization = _require(optimization_id)
    criteria = opt_db.list_criteria(optimization_id)

    yield _progress(optimization_id, "step_start", "revise", iteration=iteration)
    stream = await service.revise_prompt(optimization["current_prompt"], criteria, changes)
    async for delta in stream:
        yield {"type": "delta", "step": "revise", "text": delta}
    completion = stream.completion
    opt_db.record_run(optimization_id, "revise", completion, iteration=iteration)

    version = optimization["prompt_version"] + 1
    change_notes = service.changes_as_text(changes)
    if rejected:
        change_notes += f"\n\n({rejected} further change(s) were proposed and rejected by the reviewer.)"
    opt_db.add_prompt_version(optimization_id, version, completion.content, change_notes)
    opt_db.update_optimization(optimization_id, current_prompt=completion.content, prompt_version=version)
    _write(optimization_id, f"prompt_v{version}.md", completion.content)
    _emit(optimization_id, "revise", completion.content, version=version, change_notes=change_notes)
    yield _progress(optimization_id, "step_done", "revise")

    _post_iteration_form(
        optimization_id,
        version=version,
        iteration=iteration,
        scoreboard=scoreboard,
        added=added,
        content=(
            f"Prompt v{version} is ready."
            f" Iteration {iteration} left {scoreboard['failed']} of {scoreboard['total']} test cases failing."
            " Run the same test cases against the new prompt?"
        ),
    )
    _save_session(optimization_id)


def _post_iteration_form(
    optimization_id: str,
    version: int,
    iteration: int,
    scoreboard: dict[str, int],
    added: list[dict[str, Any]],
    content: str,
    changed: bool = True,
) -> None:
    """Offer another run against the current prompt version.

    ``changed`` is false when the user rejected every proposal, so the card can
    say the prompt is unchanged rather than announcing a new version.
    """
    opt_db.update_optimization(optimization_id, stage="awaiting_iteration")
    opt_db.supersede_kind(optimization_id, "iteration_form")
    opt_db.add_message(
        optimization_id,
        role="assistant",
        kind="iteration_form",
        content=content,
        payload={
            "version": version,
            "scoreboard": scoreboard,
            "new_criteria": added,
            "next_iteration": iteration + 1,
            "changed": changed,
        },
    )


async def _apply_new_criteria(
    optimization_id: str, new_criteria: list[dict[str, Any]], iteration: int
) -> list[dict[str, Any]]:
    """Append the criteria the user approved, then rebuild the judge prompt.

    A remark the judge had no criterion for is exactly the gap this loop exists to
    close, so the new criteria go in before the next run rather than being
    reported and forgotten — but only the ones that survived the approval card.
    Titles already present are skipped so repeated iterations do not stack
    near-duplicates.
    """
    if not new_criteria:
        return []
    optimization = _require(optimization_id)
    existing = opt_db.list_criteria(optimization_id)
    known = {criterion["title"].strip().lower() for criterion in existing}
    added = [
        {**criterion, "origin": "user", "added_iteration": iteration}
        for criterion in new_criteria
        if criterion.get("title", "").strip().lower() not in known
    ]
    if not added:
        return []

    combined = service.assign_ids([*existing, *added])
    opt_db.replace_criteria(optimization_id, combined, iteration=iteration)
    judge_prompt = service.build_judge_prompt(optimization["judge_scaffold"], combined)
    opt_db.update_optimization(optimization_id, judge_prompt=judge_prompt)
    _write(optimization_id, "judge_prompt.md", judge_prompt)
    _write(optimization_id, "criteria.json", json.dumps(combined, indent=2))
    opt_db.add_message(
        optimization_id,
        role="assistant",
        kind="criteria_added",
        content=f"Added {len(added)} success criteri{'on' if len(added) == 1 else 'a'} from your remarks.",
        payload={"added": added, "criteria": combined},
    )
    return added


# --------------------------------------------------------------------------- blocking wrappers


async def submit_prompt(optimization_id: str, prompt: str, template_kind: str) -> dict[str, Any]:
    """Blocking form of :func:`submit_prompt_stream`."""
    return await _drain(optimization_id, submit_prompt_stream(optimization_id, prompt, template_kind))


async def submit_observations(optimization_id: str, observations: str) -> dict[str, Any]:
    """Blocking form of :func:`submit_observations_stream`."""
    return await _drain(optimization_id, submit_observations_stream(optimization_id, observations))


async def submit_criteria(
    optimization_id: str, criteria: list[dict[str, Any]], additions: str = ""
) -> dict[str, Any]:
    """Blocking form of :func:`submit_criteria_stream`."""
    return await _drain(optimization_id, submit_criteria_stream(optimization_id, criteria, additions))


async def run_iteration(optimization_id: str) -> dict[str, Any]:
    """Blocking form of :func:`run_iteration_stream`."""
    return await _drain(optimization_id, run_iteration_stream(optimization_id))


async def submit_review(optimization_id: str, feedback: list[dict[str, Any]]) -> dict[str, Any]:
    """Blocking form of :func:`submit_review_stream`."""
    return await _drain(optimization_id, submit_review_stream(optimization_id, feedback))


async def submit_changes(
    optimization_id: str, approved_changes: list[int], approved_criteria: list[int]
) -> dict[str, Any]:
    """Blocking form of :func:`submit_changes_stream`."""
    return await _drain(
        optimization_id, submit_changes_stream(optimization_id, approved_changes, approved_criteria)
    )
