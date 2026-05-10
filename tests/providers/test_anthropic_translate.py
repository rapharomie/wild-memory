"""
Tests for AnthropicLLM._translate.

We don't make network calls. Instead we hand the translator fake response
objects shaped like `anthropic.types.Message` and verify the translation into
`LLMResponse` is what core code expects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wild_memory.providers.anthropic_llm import AnthropicLLM
from wild_memory.providers.base import ToolResult


# ── Fake response shapes (mimic anthropic.types.Message duck-type) ─────


@dataclass
class _FakeBlock:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict | None = None


class _FakeMsg:
    def __init__(self, content: list[Any], stop_reason: str):
        self.content = content
        self.stop_reason = stop_reason


# ── Tests ─────────────────────────────────────────────────────────────


def test_translate_plain_text():
    msg = _FakeMsg(
        content=[_FakeBlock(type="text", text="Hello, world!")],
        stop_reason="end_turn",
    )
    r = AnthropicLLM._translate(msg)
    assert r.text == "Hello, world!"
    assert r.tool_calls == []
    assert r.stop_reason == "end"
    assert r.raw is msg


def test_translate_tool_use_only():
    msg = _FakeMsg(
        content=[
            _FakeBlock(
                type="tool_use",
                id="t_abc",
                name="recall_memory",
                input={"query": "favorite color", "search_type": "semantic"},
            )
        ],
        stop_reason="tool_use",
    )
    r = AnthropicLLM._translate(msg)
    assert r.text == ""
    assert r.stop_reason == "tool_use"
    assert len(r.tool_calls) == 1
    call = r.tool_calls[0]
    assert call.id == "t_abc"
    assert call.name == "recall_memory"
    assert call.arguments == {"query": "favorite color", "search_type": "semantic"}


def test_translate_mixed_text_and_tool_use():
    msg = _FakeMsg(
        content=[
            _FakeBlock(type="text", text="Let me check that. "),
            _FakeBlock(
                type="tool_use",
                id="t_xyz",
                name="save_observation",
                input={"content": "User likes blue", "obs_type": "preference", "importance": 6},
            ),
            _FakeBlock(type="text", text="One moment."),
        ],
        stop_reason="tool_use",
    )
    r = AnthropicLLM._translate(msg)
    assert r.text == "Let me check that. One moment."
    assert r.stop_reason == "tool_use"
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "save_observation"


def test_translate_max_tokens_stop_reason():
    msg = _FakeMsg(
        content=[_FakeBlock(type="text", text="...")],
        stop_reason="max_tokens",
    )
    r = AnthropicLLM._translate(msg)
    assert r.stop_reason == "max_tokens"


def test_translate_unknown_stop_reason_defaults_to_end():
    msg = _FakeMsg(
        content=[_FakeBlock(type="text", text="done")],
        stop_reason="something_weird",
    )
    r = AnthropicLLM._translate(msg)
    assert r.stop_reason == "end"


def test_format_tool_results_shape():
    """Tool results must be wrapped in a single user-role message that the API accepts."""
    llm = AnthropicLLM.__new__(AnthropicLLM)  # bypass __init__ so we don't need the SDK
    out = llm.format_tool_results(
        [
            ToolResult(tool_call_id="t1", content="result A"),
            ToolResult(tool_call_id="t2", content="result B", is_error=True),
        ]
    )
    assert out["role"] == "user"
    assert isinstance(out["content"], list)
    assert len(out["content"]) == 2
    assert out["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "result A",
        "is_error": False,
    }
    assert out["content"][1]["is_error"] is True
