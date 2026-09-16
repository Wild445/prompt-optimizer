"""Run the whole prompt-optimization loop against ``fake_llm.FakeOpenAIClient``.

Drives ``webapp/optimizer.py`` end to end — scaffold, criteria, judge, dataset,
two graded iterations with a user disagreement in between — against a temporary
SQLite file and a temporary output folder, with no network and no API key.

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
    print(f"  judge prompt: {len(opt_db.get_optimization(optimization_id)['judge_prompt'])} chars")

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
    await drive("review -> analyze -> revise", optimizer.submit_review_stream(optimization_id, feedback))

    optimization = opt_db.get_optimization(optimization_id)
    print(f"  now at prompt v{optimization['prompt_version']}, stage={optimization['stage']}")

    if optimization["stage"] == "awaiting_iteration":
        await drive("iteration 2", optimizer.run_iteration_stream(optimization_id))
        print(f"  scoreboard: {opt_db.iteration_scoreboard(optimization_id, 2)}")

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
