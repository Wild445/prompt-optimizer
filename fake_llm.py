"""A fake OpenAI-shaped chat client for exercising the pipeline with no network calls.

Drop-in replacement for ``chat_common.services.llm_service.get_openai_client``:
``FakeOpenAIClient().chat.completions.create(...)`` mimics the subset of the
``openai`` SDK response shape that ``runtime.py`` reads (``choices[0].message
.content``, ``.usage``, and for streaming, ``choices[0].delta.content``).

Used by ``scripts/test_pipeline_offline.py`` to run every synthesis stage
without hitting Azure OpenAI.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

_CLARIFY_SCHEMA_NAME = "clarify_result"
_CLARIFY_MARKER = "Clarify stage of a prompt-synthesis pipeline"

#: Prose returned for any stage that isn't asking for the clarify JSON schema.
_DEFAULT_PROSE = "This is a canned offline completion standing in for the real LLM response."

#: One marker per optimization agent, matched against the system prompt. Same
#: sniffing approach as _wants_clarify_json below, and for the same reason: the
#: installed prompty release never forwards the `response_format:` frontmatter,
#: so there is no structured-output signal to key off.
_JUDGE_SCAFFOLD_MARKER = "Judge Scaffold stage of a prompt-optimization pipeline"
_CRITERIA_DRAFTER_MARKER = "Criteria Drafter stage of a prompt-optimization pipeline"
_CRITERIA_CONSOLIDATOR_MARKER = "Criteria Consolidator stage of a prompt-optimization pipeline"
_FAILURE_ANALYST_MARKER = "Failure Analyst stage of a prompt-optimization pipeline"
_PROMPT_REVISOR_MARKER = "Prompt Revisor stage of a prompt-optimization pipeline"

_FAKE_SCAFFOLD = """# Role
You are a strict evaluator of one response at a time (offline fake scaffold).

# What you receive
The prompt under test, the input it was given, and the response that came back.

# Success criteria
{{SUCCESS_CRITERIA}}

# How to evaluate
Score each criterion true or false, then compile. One false makes the whole
verdict false.

# Output
One JSON object with `criteria`, `result`, `failed_criteria`, and `failures`.
"""

_FAKE_CRITERIA = (
    ("Answers in the requested format", "The response uses exactly the structure the prompt asks for."),
    ("States a figure for every claim", "Every quantitative claim carries the number it is based on."),
    ("Stays within the requested scope", "The response does not answer questions that were not asked."),
)


def _fake_criteria_json(origin: str) -> str:
    return json.dumps(
        {
            "criteria": [
                {"title": title, "description": description, "origin": origin}
                for title, description in _FAKE_CRITERIA
            ]
        }
    )


def _fake_verdict_json(messages: list[dict[str, Any]]) -> str:
    """Pass or fail a case deterministically, so an offline run shows both outcomes.

    Keyed off the hashed user turn rather than a counter: the UI re-renders from
    the server after every run, and a stable verdict per test case is what makes
    that snapshot match what the live counters just showed.
    """
    payload = "".join(str(message.get("content") or "") for message in messages if message.get("role") == "user")
    passed = hashlib.sha1(payload.encode("utf-8")).digest()[0] % 3 != 0
    criteria = [
        {"id": f"C{index}", "passed": passed or index != 2, "reason": "Offline fake verdict."}
        for index in range(1, len(_FAKE_CRITERIA) + 1)
    ]
    failed = [entry["id"] for entry in criteria if not entry["passed"]]
    return json.dumps(
        {
            "criteria": criteria,
            "result": "Success" if not failed else "Failed",
            "failed_criteria": failed,
            "failures": [
                {
                    "id": entry_id,
                    "criterion": _FAKE_CRITERIA[int(entry_id[1:]) - 1][0],
                    "why": "The offline fake judge fails this criterion for a third of inputs.",
                    "prompt_section": "(offline fake)",
                }
                for entry_id in failed
            ],
        }
    )


def _wants_clarify_json(messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> bool:
    # The "correct" signal is response_format.json_schema.name, but the
    # installed prompty==2.0.1 only maps camelCase option keys (maxOutputTokens,
    # additionalProperties, ...) onto ModelOptions, so this repo's snake_case
    # `response_format:` in the .prompty frontmatter never actually reaches
    # `create()` today. Fall back to sniffing the system prompt so this fake
    # client still exercises the clarify JSON path end to end.
    response_format = kwargs.get("response_format") or {}
    schema = (response_format or {}).get("json_schema") or {}
    if schema.get("name") == _CLARIFY_SCHEMA_NAME:
        return True
    return any(_CLARIFY_MARKER in (message.get("content") or "") for message in messages)


def _system_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content") or "") for message in messages if message.get("role") == "system")


def _fake_content(messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
    if _wants_clarify_json(messages, kwargs):
        return json.dumps(
            {
                "status": "ready",
                "questions": [],
                "clarified_summary": "Offline fake summary: the draft is clear enough to proceed.",
            }
        )

    system = _system_text(messages)
    if _JUDGE_SCAFFOLD_MARKER in system:
        return _FAKE_SCAFFOLD
    if _CRITERIA_DRAFTER_MARKER in system:
        return _fake_criteria_json("observation")
    if _CRITERIA_CONSOLIDATOR_MARKER in system:
        return _fake_criteria_json("added")
    if _FAILURE_ANALYST_MARKER in system:
        return json.dumps(
            {
                "summary": "Offline fake analysis: the format criterion dominates the failures.",
                "changes": [
                    {
                        "criteria": ["C2"],
                        "prompt_section": "(missing)",
                        "change": "Add a rule requiring a figure for every quantitative claim.",
                        "evidence": "Offline fake evidence from 1 test case.",
                    },
                    {
                        "criteria": ["C1"],
                        "prompt_section": "Answer questions about ...",
                        "change": "State explicitly that the answer must be three bullets and nothing else.",
                        "evidence": "Offline fake evidence from 2 test cases.",
                    },
                ],
                "new_criteria": [
                    {
                        "title": "Cites the reporting period",
                        "description": "The response names the reporting period it is answering about.",
                    }
                ],
            }
        )
    if _PROMPT_REVISOR_MARKER in system:
        return "Offline fake revision of the prompt, with the prescribed changes applied."

    # The generated judge prompt has no .prompty marker to match - it only exists
    # in the database - so it is identified by the response_format the judge call
    # passes explicitly (see optimization.service.judge_response).
    if (kwargs.get("response_format") or {}).get("type") == "json_object":
        return _fake_verdict_json(messages)

    return _DEFAULT_PROSE


@dataclass
class _Usage:
    prompt_tokens: int = 12
    completion_tokens: int = 8
    total_tokens: int = 20


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: Optional[_Message] = None
    delta: Optional[_Message] = None


@dataclass
class _ChatCompletion:
    choices: list[_Choice] = field(default_factory=list)
    usage: Optional[_Usage] = None


def _stream_chunks(content: str) -> Iterator[_ChatCompletion]:
    """Yield the fake content one word at a time, like a real SSE stream would."""
    words = content.split(" ")
    for index, word in enumerate(words):
        piece = word if index == 0 else f" {word}"
        yield _ChatCompletion(choices=[_Choice(delta=_Message(content=piece))])
    # Final chunk carries usage, same as the real API with stream_options include_usage.
    yield _ChatCompletion(choices=[], usage=_Usage())


class _FakeCompletions:
    def create(self, *, model: str, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any):
        content = _fake_content(messages, kwargs)
        # Simulate a tiny bit of latency so latency_seconds isn't exactly zero.
        time.sleep(0.01)
        if stream:
            return _stream_chunks(content)
        return _ChatCompletion(
            choices=[_Choice(message=_Message(content=content))],
            usage=_Usage(),
        )


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class FakeOpenAIClient:
    """Stand-in for the AzureOpenAI client returned by ``get_openai_client()``."""

    def __init__(self) -> None:
        self.chat = _FakeChat()
