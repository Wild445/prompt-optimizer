"""Thin async wrappers around Prompty 2.x load/prepare plus the NIQ LLM client.

Two call styles are offered: ``complete_prompt`` blocks until the whole
completion is back, and ``stream_prompt`` hands out text deltas as they arrive
so a UI can render mid-generation. Both end up with the same
``PromptCompletion`` record, so telemetry is identical either way.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional

import prompty
from openai import BadRequestError

from chat_common.common.logging import logger
from chat_common.services.llm_service import get_openai_client
from config import load_env

# Prompty resolves ${env:...} frontmatter during load, so .env must land first.
load_env()

_PROMPT_CACHE: dict[str, Any] = {}

_COMPLETION_PARAM_KEYS = ("temperature", "max_tokens", "response_format", "top_p", "timeout")


@dataclass(frozen=True)
class PreparedPrompt:
    """Rendered chat messages plus model settings from a Prompty 2.x file."""

    name: str
    path: Path
    messages: list[dict[str, Any]]
    parameters: dict[str, Any]
    connection: dict[str, Any]
    model: str


@dataclass(frozen=True)
class PromptCompletion:
    """LLM completion plus usage and latency for evals and agent metadata."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    messages: list[dict[str, Any]]
    model: str


def clear_prompt_cache() -> None:
    """Drop cached Prompty objects (used by tests when env vars change)."""
    _PROMPT_CACHE.clear()


def merge_history(
    messages: list[dict[str, Any]],
    extra_messages: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Append extra chat turns, or insert them before a trailing Prompty user block.

    Untrusted turns (history and the current user message) must be passed as
    ``extra_messages`` so PromptyChatParser never sees them. System-only
    ``.prompty`` files therefore just append; a trailing ``user:`` block in the
    file still keeps history in front of that last user message.
    """
    extra = [{"role": msg["role"], "content": msg["content"]} for msg in extra_messages or [] if msg.get("role") and msg.get("content") and msg.get("role") != "system"]
    if not extra:
        return list(messages)
    merged = list(messages)
    if merged and merged[-1].get("role") == "user":
        return merged[:-1] + extra + merged[-1:]
    return merged + extra


def _completion_kwargs(parameters: dict[str, Any], extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Keep only kwargs accepted by ``chat.completions.create``."""
    kwargs = {key: value for key, value in parameters.items() if key in _COMPLETION_PARAM_KEYS and value is not None}
    if extra:
        kwargs.update(extra)
    return kwargs


def _parameters_from_prompt(prompt: Any) -> dict[str, Any]:
    """Map Prompty 2.x ``model.options`` onto OpenAI ``chat.completions.create`` kwargs."""
    options = getattr(getattr(prompt, "model", None), "options", None)
    if options is None:
        return {}
    parameters: dict[str, Any] = {}
    if options.temperature is not None:
        parameters["temperature"] = options.temperature
    if options.max_output_tokens is not None:
        parameters["max_completion_tokens"] = options.max_output_tokens
    if options.top_p is not None:
        parameters["top_p"] = options.top_p
    extra = options.additional_properties or {}
    parameters.update(extra)
    return parameters


def _connection_from_prompt(prompt: Any) -> dict[str, Any]:
    """Expose the Prompty 2.x key connection as a plain dict."""
    connection = getattr(getattr(prompt, "model", None), "connection", None)
    if connection is None:
        return {}
    data: dict[str, Any] = {}
    kind = getattr(connection, "kind", None)
    if kind:
        data["kind"] = kind
    endpoint = getattr(connection, "endpoint", None)
    if endpoint:
        data["endpoint"] = endpoint
    api_key = getattr(connection, "api_key", None)
    if api_key:
        data["api_key"] = api_key
    return data


def _as_chat_messages(rendered: Any) -> list[dict[str, Any]]:
    """Convert Prompty 2.x ``Message`` objects into OpenAI chat dicts."""
    messages: list[dict[str, Any]] = []
    for item in rendered:
        role = getattr(item, "role", None)
        content = getattr(item, "text", None)
        if content is None and hasattr(item, "to_text_content"):
            content = item.to_text_content()
        if not role:
            continue
        messages.append({"role": role, "content": content or ""})
    return messages


async def prepare_prompt(
    path: Path | str,
    inputs: Optional[dict[str, Any]] = None,
    extra_messages: Optional[list[dict[str, Any]]] = None,
) -> PreparedPrompt:
    """Load a version-controlled ``.prompty`` file and render it with typed inputs.

    Uses ``prompty.load_async`` / ``prompty.prepare_async``. Secrets in
    ``${env:VAR:default}`` frontmatter are resolved by the Prompty 2.x loader.
    """
    resolved = Path(path).resolve()
    cache_key = str(resolved)
    prompt = _PROMPT_CACHE.get(cache_key)
    if prompt is None:
        prompt = await prompty.load_async(cache_key)
        _PROMPT_CACHE[cache_key] = prompt

    rendered = await prompty.prepare_async(prompt, inputs or {})
    model_id = getattr(getattr(prompt, "model", None), "id", None) or os.getenv("CIS_LLM_4_DOT_1_DEPLOYMENT") or ""
    return PreparedPrompt(
        name=prompt.name or resolved.stem,
        path=resolved,
        messages=merge_history(_as_chat_messages(rendered), extra_messages),
        parameters=_parameters_from_prompt(prompt),
        connection=_connection_from_prompt(prompt),
        model=model_id,
    )


async def complete_prompt(
    path: Path | str,
    inputs: Optional[dict[str, Any]] = None,
    extra_messages: Optional[list[dict[str, Any]]] = None,
    client: Any = None,
    extra_create_kwargs: Optional[dict[str, Any]] = None,
) -> PromptCompletion:
    """Prepare a Prompty file and complete it through the existing Azure OpenAI client.

    ``prompty.invoke`` is intentionally not used: this service must keep the
    NIQ CIS consumer headers from ``get_openai_client()``.
    """
    prepared = await prepare_prompt(path, inputs=inputs, extra_messages=extra_messages)
    llm_client = client or get_openai_client()
    create_kwargs = _completion_kwargs(prepared.parameters, extra_create_kwargs)

    started = time.perf_counter()
    response = await asyncio.to_thread(
        llm_client.chat.completions.create,
        model=prepared.model,
        messages=prepared.messages,
        **create_kwargs,
    )
    latency_seconds = time.perf_counter() - started

    content = ""
    if response.choices and response.choices[0].message:
        content = (response.choices[0].message.content or "").strip()

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

    logger.info(
        "Prompty completion finished",
        extra={
            "prompty_name": prepared.name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": round(latency_seconds, 4),
        },
    )
    return PromptCompletion(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_seconds=latency_seconds,
        messages=prepared.messages,
        model=prepared.model,
    )


class PromptStream:
    """Async iterator over the text deltas of one streamed Prompty completion.

    ``get_openai_client()`` returns the *sync* Azure SDK client, so iterating its
    stream would block the event loop and stall every other request. A worker
    thread drains the SDK iterator into an ``asyncio.Queue`` instead, and this
    object replays it as an async iterator.

    Iterate it to completion, then read :attr:`completion` for the assembled text
    and its telemetry::

        stream = await stream_prompt(path)
        async for delta in stream:
            ...
        record(stream.completion)
    """

    def __init__(self, prepared: PreparedPrompt, client: Any, create_kwargs: dict[str, Any]) -> None:
        self._prepared = prepared
        self._client = client
        self._create_kwargs = create_kwargs
        self._completion: Optional[PromptCompletion] = None

    @property
    def prepared(self) -> PreparedPrompt:
        """The rendered prompt this stream was opened for."""
        return self._prepared

    @property
    def completion(self) -> PromptCompletion:
        """The finished completion. Only valid once the stream is exhausted."""
        if self._completion is None:
            raise RuntimeError("PromptStream must be fully iterated before reading `completion`")
        return self._completion

    def _open(self) -> Iterator[Any]:
        """Start the SDK stream, asking for usage totals when the API allows it."""
        base = {
            "model": self._prepared.model,
            "messages": self._prepared.messages,
            "stream": True,
            **self._create_kwargs,
        }
        try:
            return self._client.chat.completions.create(**base, stream_options={"include_usage": True})
        except BadRequestError:
            # Older API versions reject stream_options; give up the token counts
            # rather than the stream itself.
            logger.warning(
                "stream_options unsupported; streaming without usage totals",
                extra={"prompty_name": self._prepared.name},
            )
            return self._client.chat.completions.create(**base)

    async def __aiter__(self) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        done = object()

        def pump() -> None:
            """Drain the blocking SDK iterator onto the loop's queue."""
            try:
                for chunk in self._open():
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except BaseException as error:  # noqa: BLE001 - re-raised on the consumer side
                loop.call_soon_threadsafe(queue.put_nowait, error)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        chunks: list[str] = []
        usage: Any = None
        started = time.perf_counter()
        threading.Thread(target=pump, name=f"prompty-stream-{self._prepared.name}", daemon=True).start()

        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, BaseException):
                raise item
            usage = getattr(item, "usage", None) or usage
            for choice in getattr(item, "choices", None) or []:
                text = getattr(getattr(choice, "delta", None), "content", None)
                if text:
                    chunks.append(text)
                    yield text

        latency_seconds = time.perf_counter() - started
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

        logger.info(
            "Prompty stream finished",
            extra={
                "prompty_name": self._prepared.name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": round(latency_seconds, 4),
            },
        )
        self._completion = PromptCompletion(
            content="".join(chunks).strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency_seconds,
            messages=self._prepared.messages,
            model=self._prepared.model,
        )


async def stream_prompt(
    path: Path | str,
    inputs: Optional[dict[str, Any]] = None,
    extra_messages: Optional[list[dict[str, Any]]] = None,
    client: Any = None,
    extra_create_kwargs: Optional[dict[str, Any]] = None,
) -> PromptStream:
    """Prepare a Prompty file and open it as a token stream.

    The returned :class:`PromptStream` has not called the API yet — the request
    is issued on first iteration.
    """
    prepared = await prepare_prompt(path, inputs=inputs, extra_messages=extra_messages)
    return PromptStream(
        prepared,
        client or get_openai_client(),
        _completion_kwargs(prepared.parameters, extra_create_kwargs),
    )

async def complete_messages(
    messages: list[dict[str, Any]],
    *,
    model: Optional[str] = None,
    client: Any = None,
    label: str = "ad-hoc",
    **create_kwargs: Any,
) -> PromptCompletion:
    """Complete a message list that was built at run time, not loaded from a Prompty file.

    The optimization loop runs two prompts that only exist in the database: the
    prompt under test (a user-supplied template rendered per test case) and the
    generated LLM-as-a-judge prompt. Neither can be a ``.prompty`` file, but both
    still need the NIQ CIS client and the same token/latency record every other
    call produces — hence this sibling of :func:`complete_prompt`.
    """
    llm_client = client or get_openai_client()
    deployment = model or os.getenv("CIS_LLM_4_DOT_1_DEPLOYMENT") or ""

    started = time.perf_counter()
    response = await asyncio.to_thread(
        llm_client.chat.completions.create,
        model=deployment,
        messages=messages,
        **create_kwargs,
    )
    latency_seconds = time.perf_counter() - started

    content = ""
    if response.choices and response.choices[0].message:
        content = (response.choices[0].message.content or "").strip()

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

    logger.info(
        "Ad-hoc completion finished",
        extra={
            "prompty_name": label,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": round(latency_seconds, 4),
        },
    )
    return PromptCompletion(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_seconds=latency_seconds,
        messages=messages,
        model=deployment,
    )
