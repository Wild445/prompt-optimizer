"""Independent, callable steps of the prompt-optimization loop.

Mirrors ``synthesis/service.py``: each function wraps one ``.prompty`` file under
``prompts/`` (10-15) or one ad-hoc completion, and holds no pipeline logic — that
belongs to ``webapp/optimizer.py``.

Two of the calls here have no ``.prompty`` file behind them, because the prompt
only exists in the database: :func:`run_prompt_under_test` runs the user's own
prompt, and :func:`judge_response` runs the generated LLM-as-a-judge prompt. Both
go through ``runtime.complete_messages`` so their tokens and latency are recorded
like every other stage.

As in the synthesis service, user-supplied text is passed as ``extra_messages``
rather than templated into the system message, so PromptyChatParser never sees a
line like ``system:`` inside a pasted prompt and turns it into a role block.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from optimization.dataset import render_prompt
from runtime import PromptCompletion, complete_messages, complete_prompt, stream_prompt

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_JUDGE_PLACEHOLDER = "{{SUCCESS_CRITERIA}}"

#: Wrapper for text that came from the user or from a model, so an agent reading
#: several blocks can tell where each one starts and that none of them are
#: instructions addressed to it.
_BLOCK = "<<<{name}\n{body}\n{name}>>>"


def _block(name: str, body: str) -> str:
    return _BLOCK.format(name=name, body=(body or "").strip() or "(empty)")


def parse_json_object(content: str) -> dict[str, Any]:
    """Read a JSON object out of a model response that may be wrapped in prose.

    The ``response_format`` in these ``.prompty`` files is only honoured on API
    versions that support it — and the installed prompty release does not map
    the snake_case frontmatter onto the SDK call at all (see ``fake_llm.py``), so
    in practice nothing enforces the shape. Strip a code fence, then fall back to
    the outermost braces, rather than failing a whole run on a stray "Here is".

    Raises:
        ValueError: Nothing object-shaped was found.
    """
    text = (content or "").strip()
    fenced = re.match(r"^```(?:json)?\s*\n(.*?)\n?```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("The model did not return a JSON object.")
        loaded = json.loads(text[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("The model returned JSON, but not an object.")
    return loaded


def _normalize_criteria(raw: Any) -> list[dict[str, str]]:
    """Coerce whatever came back into ``[{title, description, origin}]``.

    Accepts bare strings as well as objects: with the response format unenforced,
    a plain list of criteria strings is the most common off-spec shape, and it
    carries everything the next stage actually needs.
    """
    criteria: list[dict[str, str]] = []
    for item in raw or []:
        if isinstance(item, str):
            title = item.strip()
            if title:
                criteria.append({"id": "", "title": title[:120], "description": title, "origin": "agent"})
            continue
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("criterion") or item.get("name") or "").strip()
        description = str(item.get("description") or item.get("detail") or "").strip()
        if not title and not description:
            continue
        criteria.append(
            {
                "id": str(item.get("id") or "").strip(),
                "title": title or description[:120],
                "description": description or title,
                "origin": str(item.get("origin") or "agent").strip() or "agent",
            }
        )
    return criteria


def assign_ids(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stamp ``C1..Cn`` onto a criteria list, in order.

    Ids are positional and re-assigned whenever the list changes rather than
    being sticky per criterion: the judge is handed the list as text, and an
    ordered ``C1, C2, C3`` is what keeps its ``failed_criteria`` readable
    against it.
    """
    return [{**criterion, "id": f"C{index}"} for index, criterion in enumerate(criteria, start=1)]


def criteria_as_text(criteria: list[dict[str, Any]]) -> str:
    """Render criteria the way every agent downstream expects to read them."""
    return "\n".join(
        f"{criterion.get('id') or f'C{index}'} — {criterion.get('title', '')}\n    {criterion.get('description', '')}"
        for index, criterion in enumerate(criteria, start=1)
    )


def criteria_as_judge_text(criteria: list[dict[str, Any]]) -> str:
    """Render criteria for the judge prompt: a bold heading line, then the description.

    The judge walks this list one entry at a time and quotes the ids back in
    ``failed_criteria``, so ``**C1 — Title**`` keeps the boundary between one
    criterion and the next unmissable however long the list gets. The text of the
    title and description is untouched — only the markers around them are added.
    """
    return "\n\n".join(
        f"**{criterion.get('id') or f'C{index}'} — {criterion.get('title', '')}**"
        f"\n{criterion.get('description', '')}"
        for index, criterion in enumerate(criteria, start=1)
    )


# --------------------------------------------------------------------------- agents


async def build_judge_scaffold(prompt_under_test: str) -> PromptCompletion:
    """Step 2 — draft the judge skeleton, with the criteria left as a placeholder."""
    return await complete_prompt(
        _PROMPTS_DIR / "10_judge_scaffold.prompty",
        extra_messages=[{"role": "user", "content": _block("PROMPT_UNDER_TEST", prompt_under_test)}],
    )


async def draft_criteria(prompt_under_test: str, observations: str, max_criteria: int = 10) -> dict[str, Any]:
    """Step 3 — turn the user's complaints into checkable success criteria.

    Returns ``{"criteria": [...], "completion": PromptCompletion}``. On an
    unparseable response the criteria list comes back empty rather than raising,
    so the user can still add their own by hand and carry on.
    """
    completion = await complete_prompt(
        _PROMPTS_DIR / "11_criteria_drafter.prompty",
        inputs={"max_criteria": max_criteria},
        extra_messages=[
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        _block("PROMPT_UNDER_TEST", prompt_under_test),
                        _block("USER_OBSERVATIONS", observations),
                    ]
                ),
            }
        ],
    )
    try:
        body = parse_json_object(completion.content)
    except (ValueError, json.JSONDecodeError):
        return {"criteria": [], "completion": completion, "error": completion.content[:400]}
    return {"criteria": _normalize_criteria(body.get("criteria")), "completion": completion}


def consolidate_criteria(
    kept: list[dict[str, Any]],
    user_added: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Step 5a — the final list: what the user kept, then what they added.

    Deliberately not an LLM call. A model asked to consolidate this list merges
    entries the user wanted separate, re-words descriptions it finds loose, and
    re-introduces checks the user deleted because the prompt under test still
    implies them — so the judge ends up scoring a list nobody approved. The only
    thing done to the list here is stamping positional ids.
    """
    return assign_ids([*kept, *user_added])


def build_judge_prompt(scaffold: str, criteria: list[dict[str, Any]]) -> str:
    """Step 5b — fold the final criteria into the scaffold to get the judge prompt.

    Deliberately not an LLM call. This is a mechanical substitution, and a model
    asked to perform it re-words criteria on the way through, revives ones the
    user deleted, and adds its own — so the judge ends up scoring a list nobody
    approved. The list that comes in is the list that goes out, verbatim.

    The scaffold is expected to carry the ``{{SUCCESS_CRITERIA}}`` placeholder on
    a line of its own; if the scaffold stage dropped it, the ``# Success
    criteria`` section is rewritten in place (discarding anything the scaffold
    invented there), and only a scaffold missing that section too gets one
    appended.
    """
    block = criteria_as_judge_text(criteria)
    if _JUDGE_PLACEHOLDER in scaffold:
        return scaffold.replace(_JUDGE_PLACEHOLDER, block)
    section = re.search(
        r"^(#+\s*Success criteria[^\n]*\n)(.*?)(?=^#+\s|\Z)",
        scaffold,
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    if section:
        return scaffold[: section.start(2)] + f"{block}\n\n" + scaffold[section.end(2) :]
    return f"{scaffold.rstrip()}\n\n# Success criteria\n{block}\n"


async def analyze_failures(
    prompt_under_test: str,
    criteria: list[dict[str, Any]],
    run_report: str,
) -> dict[str, Any]:
    """Step 9 — read the graded run plus the user's remarks into a change list."""
    completion = await complete_prompt(
        _PROMPTS_DIR / "14_failure_analyst.prompty",
        extra_messages=[
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        _block("PROMPT_UNDER_TEST", prompt_under_test),
                        _block("SUCCESS_CRITERIA", criteria_as_text(criteria)),
                        _block("EVALUATION_RUN", run_report),
                    ]
                ),
            }
        ],
    )
    try:
        body = parse_json_object(completion.content)
    except (ValueError, json.JSONDecodeError):
        return {
            "summary": completion.content,
            "changes": [],
            "new_criteria": [],
            "completion": completion,
        }
    changes = [change for change in (body.get("changes") or []) if isinstance(change, dict)]
    return {
        "summary": str(body.get("summary") or "").strip(),
        "changes": changes,
        "new_criteria": _normalize_criteria(body.get("new_criteria")),
        "completion": completion,
    }


def changes_as_text(changes: list[dict[str, Any]]) -> str:
    """Render the analyst's change list for the revisor (and for the transcript)."""
    lines: list[str] = []
    for index, change in enumerate(changes, start=1):
        targets = ", ".join(str(item) for item in (change.get("criteria") or [])) or "user remark"
        lines.append(
            f"{index}. [{targets}] in `{change.get('prompt_section') or '(missing)'}`:"
            f" {change.get('change') or ''}"
            + (f"\n   Evidence: {change['evidence']}" if change.get("evidence") else "")
        )
    return "\n".join(lines)


async def revise_prompt(
    prompt_under_test: str,
    criteria: list[dict[str, Any]],
    changes: list[dict[str, Any]],
):
    """Step 10 — apply the change list and return the next prompt version as a stream."""
    return await stream_prompt(
        _PROMPTS_DIR / "15_prompt_revisor.prompty",
        extra_messages=[
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        _block("CURRENT_PROMPT", prompt_under_test),
                        _block("SUCCESS_CRITERIA", criteria_as_text(criteria)),
                        _block("PRESCRIBED_CHANGES", changes_as_text(changes)),
                    ]
                ),
            }
        ],
    )


# --------------------------------------------------------------------------- the loop's two ad-hoc calls


async def run_prompt_under_test(
    prompt: str,
    template_kind: str,
    input_payload: str,
    params: Optional[dict[str, Any]] = None,
) -> PromptCompletion:
    """Step 7 — render the prompt for one test case and get the response back.

    The rendered prompt becomes the system message and ``input_payload`` the user
    message, which is how the prompt is used in the system it came from.

    Raises:
        TemplateError: The row's variables do not satisfy the template. Raised
            per case so one bad row fails one test case, not the run.
    """
    rendered = render_prompt(prompt, template_kind, params)
    return await complete_messages(
        [
            {"role": "system", "content": rendered},
            {"role": "user", "content": input_payload},
        ],
        label="prompt_under_test",
    )


def _normalize_verdict(body: dict[str, Any]) -> dict[str, Any]:
    """Coerce a judge response into the shape the runner and the UI store.

    The overall verdict is recomputed from the per-criterion scores rather than
    trusted: "every criterion true, else false" is the rule the whole loop rests
    on, and it is cheaper to enforce here than to hope the judge applied it.
    """
    entries: list[dict[str, Any]] = []
    for item in body.get("criteria") or []:
        if not isinstance(item, dict):
            continue
        entries.append(
            {
                "id": str(item.get("id") or "").strip(),
                "passed": bool(item.get("passed")),
                "reason": str(item.get("reason") or "").strip(),
            }
        )

    failures = [failure for failure in (body.get("failures") or []) if isinstance(failure, dict)]
    failed_ids = [entry["id"] for entry in entries if not entry["passed"]]
    if not entries:
        # No per-criterion breakdown came back; fall back to whatever the judge
        # reported directly so the case is still scored rather than errored.
        failed_ids = [str(item).strip() for item in (body.get("failed_criteria") or []) if str(item).strip()]
        if not failed_ids:
            failed_ids = [str(failure.get("id") or "").strip() for failure in failures if failure.get("id")]
        passed = str(body.get("result") or "").lower() in ("success", "true", "pass", "passed") and not failed_ids
    else:
        passed = not failed_ids

    return {
        "passed": passed,
        "result": "Success" if passed else "Failed",
        "criteria": entries,
        "failed_criteria": failed_ids,
        "failures": failures,
    }


async def judge_response(
    judge_prompt: str,
    prompt_under_test: str,
    input_payload: str,
    response: str,
) -> dict[str, Any]:
    """Step 8 — score one response against the criteria baked into ``judge_prompt``.

    Returns the normalized verdict plus ``completion`` and the judge's raw text.

    Raises:
        ValueError: The judge did not return anything object-shaped. The runner
            records that against the one test case and keeps going.
    """
    completion = await complete_messages(
        [
            {"role": "system", "content": judge_prompt},
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        _block("PROMPT_UNDER_TEST", prompt_under_test),
                        _block("INPUT", input_payload),
                        _block("RESPONSE", response),
                    ]
                ),
            },
        ],
        label="llm_as_a_judge",
        response_format={"type": "json_object"},
    )
    verdict = _normalize_verdict(parse_json_object(completion.content))
    verdict["completion"] = completion
    verdict["raw"] = completion.content
    return verdict
