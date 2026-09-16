"""Run every synthesis stage against ``fake_llm.FakeOpenAIClient`` — no network, no API key.

Verifies the whole pipeline wiring (Prompty loading/rendering, stage
composition in ``synthesis/service.py``, and both the blocking and streaming
call paths in ``runtime.py``) without ever calling Azure OpenAI.

Usage:
    python scripts/test_pipeline_offline.py
"""

from __future__ import annotations

import asyncio
import os
import sys
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
from synthesis import service  # noqa: E402

_fake_client = FakeOpenAIClient()
runtime.get_openai_client = lambda: _fake_client


async def main() -> None:
    print("== build_prompt ==")
    build = await service.build_prompt("A tool that summarizes customer support tickets")
    print(build.content)

    print("\n== clarify ==")
    clarify_result = await service.clarify(build.content)
    print(clarify_result)

    print("\n== plan_workflow ==")
    plan = await service.plan_workflow(clarify_result["clarified_summary"] or build.content)
    print(plan.content)

    print("\n== optimize_prompt ==")
    optimized = await service.optimize_prompt(plan.content)
    print(optimized.content)

    print("\n== tune_for_model (streamed) ==")
    stream = await service.stream_stage(
        service.stage_request("tune_for_model", optimized.content, target_model="gpt-5")
    )
    async for delta in stream:
        print(delta, end="")
    print()
    print(f"(tokens: {stream.completion.total_tokens}, latency: {stream.completion.latency_seconds:.3f}s)")

    print("\n== humanize ==")
    humanized = await service.humanize(stream.completion.content)
    print(humanized.content)

    print("\nAll stages completed offline with no real LLM calls.")


if __name__ == "__main__":
    asyncio.run(main())
