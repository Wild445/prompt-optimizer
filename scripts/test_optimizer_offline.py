"""Run the whole prompt-optimization loop against ``fake_llm.FakeOpenAIClient``.

Drives ``webapp/optimizer.py`` end to end — scaffold, criteria, judge, dataset,
two graded iterations with a user disagreement and a part-approved change list in
between — against a temporary SQLite file and a temporary output folder, with no
network and no API key.

Usage:
    python scripts/test_optimizer_offline.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Dummy values so Prompty's ${env:...} frontmatter resolution doesn't raise
# PromptyLoadError before we ever get to the (patched-out) network call.
os.environ.setdefault("cis_llm_endpoint", "https://fake.invalid")
os.environ.setdefault("cis_llm_apikey", "fake-key")
os.environ.setdefault("cis_llm_apiversion", "2024-08-01-preview")
os.environ.setdefault("CIS_LLM_4_DOT_1_DEPLOYMENT", "fake-deployment")

import runtime  # noqa: E402  (must import after env vars are set)
from fake_llm import FakeOpenAIClient  # noqa: E402
from optimization import dataset  # noqa: E402
from webapp import db, opt_db, optimizer  # noqa: E402

_fake_client = FakeOpenAIClient()
runtime.get_openai_client = lambda: _fake_client

PROMPT = """You are a retail analytics assistant for {{region}}.
Answer questions about {{reporting_period}} performance in three short bullets."""


async def drive(label: str, events) -> None:
    """Consume one optimizer stream, printing the progress events it emits."""
    print(f"\n== {label} ==")
    async for event in events:
        kind = event.get("type")
        if kind == "delta":
            continue
        if kind == "case_done":
            board = event["scoreboard"]
            case = event["case"]
            print(f"  case {case['message_id']}: {case['status']} passed={case['passed']}"
                  f"  ->  {board['passed']} passed / {board['failed']} failed of {board['total']}")
        else:
            print(f"  {kind}: {event.get('step')}")


async def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="optimizer-offline-"))
    db.DB_PATH = workspace / "test.db"
    db.connect(db.DB_PATH)
    opt_db.ensure_schema()
    optimizer.OUTPUT_ROOT = workspace / "outputs"

    optimization_id = opt_db.create_optimization(template_kind="jinja2")
    print(f"optimization {optimization_id} in {workspace}")

    await drive("prompt -> scaffold", optimizer.submit_prompt_stream(optimization_id, PROMPT, "jinja2"))
    await drive(
        "observations -> criteria",
        optimizer.submit_observations_stream(
            optimization_id,
            "It writes paragraphs instead of bullets.\nIt invents numbers that aren't in the data.",
        ),
    )

    criteria = opt_db.list_criteria(optimization_id)
    print(f"  drafted {len(criteria)} criteria: {[c['title'] for c in criteria]}")
    await drive(
        "criteria -> judge",
        optimizer.submit_criteria_stream(optimization_id, criteria, additions="Never exceeds three bullets"),
    )

    # The judge prompt is assembled in Python, so the criteria stored against the
    # session and the criteria the judge reads have to be the same text.
    final = opt_db.list_criteria(optimization_id)
    judge_prompt = opt_db.get_optimization(optimization_id)["judge_prompt"]
    print(f"  judge prompt: {len(judge_prompt)} chars")
    print(f"  final criteria: {[c['title'] for c in final]}")
    for criterion in final:
        assert criterion["title"] in judge_prompt, f"{criterion['id']} missing from the judge prompt"
        assert criterion["description"] in judge_prompt, f"{criterion['id']} re-worded in the judge prompt"
    assert "{{SUCCESS_CRITERIA}}" not in judge_prompt, "the placeholder was left in the judge prompt"
    assert judge_prompt.count(" — ") == len(final), "the judge prompt holds criteria that are not on the list"
    print("  verbatim check: the judge prompt holds exactly the stored criteria, word for word")

    cases = dataset.parse_dataset(dataset.sample_workbook(), dataset.sample_filename())
    optimizer.store_dataset(
        optimization_id,
        [
            {
                "message_id": case.message_id,
                "input_payload": case.input_payload,
                "other_input_params": case.other_input_params,
            }
            for case in cases
        ],
        "template.xlsx",
    )
    print(f"\n== dataset ==\n  stored {len(cases)} test cases")

    await drive("iteration 1", optimizer.run_iteration_stream(optimization_id))
    results = opt_db.list_results(optimization_id, 1)
    print(f"  results: {[(r['message_id'], r['result_label'], r['failed_criteria']) for r in results]}")

    feedback = [
        {"message_id": results[0]["message_id"], "agrees": True, "reason": ""},
        {
            "message_id": results[-1]["message_id"],
            "agrees": False,
            "reason": "This one was fine — the judge misread the third bullet.",
        },
    ]
    await drive("review -> analyze", optimizer.submit_review_stream(optimization_id, feedback))

    optimization = opt_db.get_optimization(optimization_id)
    print(f"  stage={optimization['stage']} (waiting on the user to approve the changes)")
    proposal = opt_db.latest_message(optimization_id, "changes_form")["payload"]
    for index, change in enumerate(proposal["changes"]):
        print(f"  proposed change {index}: {change['change']}")
    for index, criterion in enumerate(proposal["new_criteria"]):
        print(f"  proposed criterion {index}: {criterion['title']}")

    # Approve the first change and the new criterion, reject the rest: only the
    # approved ones should reach the revisor and the judge.
    await drive(
        "approve (1 of 2 changes) -> revise",
        optimizer.submit_changes_stream(optimization_id, approved_changes=[0], approved_criteria=[0]),
    )

    optimization = opt_db.get_optimization(optimization_id)
    print(f"  now at prompt v{optimization['prompt_version']}, stage={optimization['stage']}")
    notes = opt_db.list_prompt_versions(optimization_id)[-1]["change_notes"]
    print(f"  change notes recorded:\n    " + notes.replace("\n", "\n    "))
    print(f"  criteria now: {[c['title'] for c in opt_db.list_criteria(optimization_id)]}")

    if optimization["stage"] == "awaiting_iteration":
        await drive("iteration 2", optimizer.run_iteration_stream(optimization_id))
        print(f"  scoreboard: {opt_db.iteration_scoreboard(optimization_id, 2)}")

    print("\n== rejecting every change leaves the prompt alone ==")
    results_2 = opt_db.list_results(optimization_id, 2)
    if results_2:
        await drive(
            "review -> analyze",
            optimizer.submit_review_stream(
                optimization_id,
                [{"message_id": row["message_id"], "agrees": True, "reason": ""} for row in results_2],
            ),
        )
        before = opt_db.get_optimization(optimization_id)["prompt_version"]
        await drive(
            "approve (nothing) -> no revision",
            optimizer.submit_changes_stream(optimization_id, approved_changes=[], approved_criteria=[]),
        )
        after = opt_db.get_optimization(optimization_id)
        print(f"  prompt version {before} -> {after['prompt_version']}, stage={after['stage']}")

    print("\n== approving with nothing pending is rejected ==")
    try:
        await optimizer.submit_changes(optimization_id, [0], [])
    except optimizer.OptimizerError as error:
        print(f"  rejected as expected: {error}")

    print("\n== a bad review is rejected ==")
    try:
        await optimizer.submit_review(optimization_id, [{"message_id": "msg-001", "agrees": False, "reason": ""}])
    except optimizer.OptimizerError as error:
        print(f"  rejected as expected: {error}")

    print("\n== output folder ==")
    for path in sorted((optimizer.OUTPUT_ROOT / optimization_id).iterdir()):
        print(f"  {path.name} ({path.stat().st_size} bytes)")

    db.close()
    print(f"\nAll steps completed offline. Workspace: {workspace}")


if __name__ == "__main__":
    asyncio.run(main())
