"""Structured-output temperature clamp (issue #5815).

Strict OpenAI-compatible endpoints reject a non-default ``temperature`` combined
with ``response_format: json_schema`` with HTTP 400, which surfaces as failed
retain/consolidation with no obvious cause. Structured output is effectively
deterministic anyway, so a structured call sends temperature 0; free-form calls
keep the configured sampling temperature.
"""

import types
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


class _Facts(BaseModel):
    facts: list[str]


def _make_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="sk-test",
        base_url="",
        model="gpt-4o-mini",
    )


def _response(content: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                finish_reason="stop",
                message=types.SimpleNamespace(content=content, tool_calls=None, refusal=None),
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="gpt-4o-mini",
    )


@pytest.mark.asyncio
async def test_structured_call_sends_zero_temperature():
    llm = _make_llm()

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _response('{"facts": ["a"]}')
        await llm.call(
            messages=[{"role": "user", "content": "extract facts"}],
            response_format=_Facts,
            temperature=0.7,
        )

    assert create.call_args.kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_freeform_call_keeps_configured_temperature():
    llm = _make_llm()

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _response("ok")
        await llm.call(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.7,
        )

    assert create.call_args.kwargs["temperature"] == 0.7
