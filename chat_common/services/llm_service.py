"""LLM Adapter for Arthur Discover Chat.

This module provides functions to get LLM and embedding instances.
"""

import os
from typing import Any, Optional

from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

from chat_common.common.logging import logger, transaction_id_var

# Cost estimation constants
INPUT_COST_PER_1K = 0.01
OUTPUT_COST_PER_1K = 0.03

# Singleton instances
_openai_client: Optional[AzureOpenAI] = None
_embeddings_instance: Optional[AzureOpenAI] = None
_langchain_llm_instance = None


def get_openai_client():
    """Initialize the AzureOpenAI client (raw OpenAI SDK) if not already initialized.

    Returns:
        The AzureOpenAI instance.
    """
    global _openai_client
    logger.info("Attempting to initialize the AzureOpenAI class", extra={"environment": os.getenv("environment")})
    if _openai_client is None:
        headers = {"X-NIQ-CIS-Consumer": os.getenv("cis_chat_rms_consumer_id")}
        _openai_client = AzureOpenAI(
            azure_endpoint=os.getenv("cis_llm_endpoint"),
            api_key=os.getenv("cis_llm_apikey"),
            api_version=os.getenv("cis_llm_apiversion"),
            default_headers=headers,
        )
        logger.info("AzureOpenAI instance created successfully")
        return _openai_client
    logger.info("AzureOpenAI instance already exists")
    _openai_client.default_headers["X-NIQ-TNXN-ID"] = transaction_id_var.get()
    return _openai_client


def get_embeddings_instance():
    """Initialize the AzureOpenAI embeddings client if not already initialized.

    Returns:
        The AzureOpenAI instance configured for embeddings.
    """
    global _embeddings_instance
    logger.info("Attempting to initialize the AzureOpenAI class from langchain", extra={"environment": os.getenv("environment")})

    if _embeddings_instance is None:
        headers = {"X-NIQ-CIS-Consumer": os.getenv("cis_chat_rms_consumer_id")}
        _embeddings_instance = AzureOpenAI(
            azure_endpoint=os.getenv("cis_llm_endpoint"),
            api_key=os.getenv("cis_llm_apikey"),
            api_version=os.getenv("cis_llm_apiversion"),
            azure_deployment=os.getenv("cis_embedding_deployment", "AskArthur-DocusSearch-text-embedding-ada-002"),
            default_headers=headers,
        )
        logger.info("AzureOpenAI instance created successfully")
        return _embeddings_instance
    logger.info("AzureOpenAI instance already exists")
    return _embeddings_instance


def get_langchain_llm_instance():
    """Get a LangChain-compatible LLM instance for async operations.

    Returns:
        LangChain AzureChatOpenAI instance configured for Azure.
    """
    global _langchain_llm_instance

    if _langchain_llm_instance is None:
        try:
            transaction_id = transaction_id_var.get()
        except LookupError:
            transaction_id = None
            logger.debug("Transaction ID not available in current context")

        headers = {
            "X-NIQ-CIS-Consumer": os.getenv("cis_chat_rms_consumer_id"),
            "X-NIQ-DETECT-HALLUCINATIONS": os.getenv("DETECT_HALLUCINATIONS"),
        }
        if transaction_id:
            headers["X-NIQ-TNXN-ID"] = transaction_id

        try:
            _langchain_llm_instance = AzureChatOpenAI(
                azure_endpoint=os.getenv("cis_llm_endpoint"),
                api_key=os.getenv("cis_llm_apikey"),
                api_version=os.getenv("cis_llm_apiversion"),
                model=os.getenv("CIS_LLM_4_DOT_1_DEPLOYMENT"),
                azure_deployment=os.getenv("CIS_LLM_4_DOT_1_DEPLOYMENT"),
                temperature=0,
                top_p=0.1,
                default_headers=headers,
            )
            logger.info("LangChain AzureChatOpenAI instance created successfully")
        except ImportError:
            logger.warning("langchain_openai not available, using fallback")
            raise ImportError("langchain_openai is required for get_langchain_llm_instance()")

    return _langchain_llm_instance


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1000) * INPUT_COST_PER_1K + (completion_tokens / 1000) * OUTPUT_COST_PER_1K


def _log_token_usage(usage: Any) -> None:
    cost = _estimate_cost(usage.prompt_tokens, usage.completion_tokens)

    logger.info(
        "LLM call executed",
        extra={
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "estimated_cost": round(cost, 6),
        },
    )


def call_llm(
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
):
    """Helper function to call the LLM. Logs token usage and estimated cost if usage data is available."""
    client = get_openai_client()

    payload: dict[str, object] = {
        "model": os.getenv("CIS_LLM_4_DOT_1_DEPLOYMENT"),
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response = client.chat.completions.create(**payload)

    if response.usage:
        _log_token_usage(response.usage)

    return response