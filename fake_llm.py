"""A fake OpenAI-shaped chat client for exercising the pipeline with no network calls.

Drop-in replacement for ``chat_common.services.llm_service.get_openai_client``:
``FakeOpenAIClient().chat.completions.create(...)`` mimics the subset of the
``openai`` SDK response shape that ``runtime.py`` reads (``choices[0].message
.content``, ``.usage``, and for streaming, ``choices[0].delta.content``).

Used by ``scripts/test_pipeline_offline.py`` to run every synthesis stage
without hitting Azure OpenAI.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

_CLARIFY_SCHEMA_NAME = "clarify_result"
_CLARIFY_MARKER = "Clarify stage of a prompt-synthesis pipeline"

#: Prose returned for any stage that isn't asking for the clarify JSON schema.
_DEFAULT_PROSE = "This is a canned offline completion standing in for the real LLM response."


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


def _fake_content(messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
    if _wants_clarify_json(messages, kwargs):
        return json.dumps(
            {
                "status": "ready",
                "questions": [],
                "clarified_summary": "Offline fake summary: the draft is clear enough to proceed.",
            }
        )
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
