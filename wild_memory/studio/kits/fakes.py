"""
Deterministic LLM and embedding providers for offline kit runs.

The real providers cost money and require API keys; we don't want a
green Test Kit dashboard to be gated on either. These fakes hash inputs
into stable outputs, so the same conversation always produces the same
observations and retrievals.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from wild_memory.providers.base import LLMResponse, ToolCall, ToolResult


class FakeLLM:
    """A predictable LLM. Can emit tool calls when the user message has
    save/recall hints; otherwise returns a stub answer derived from the
    most recent memory snippet seen in the system prompt."""

    name = "fake"

    def __init__(self):
        self.calls = 0

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls += 1
        last_user = _last_user_text(messages)
        lower = last_user.lower()

        # If the last message is a tool_result block, do a final answer.
        if _last_message_has_tool_results(messages):
            answer = _stub_answer(last_user, system)
            return LLMResponse(text=answer, stop_reason="end")

        # Distillation prompts ask for JSON observations — return a JSON list.
        # The conversation we should mine lives in `system`, not `messages`.
        if "extract" in system.lower() and "observation" in system.lower():
            import json
            obs = _extract_observations(system)
            return LLMResponse(text=json.dumps(obs), stop_reason="end")

        # Goal detection prompts return a single short label.
        if "goal" in system.lower() and len(system) < 1000:
            return LLMResponse(text=_pick_goal(lower), stop_reason="end")

        # Conflict resolution: trivially say ADD (kits handle conflict
        # explicitly via the resolver, not via the LLM path).
        if "classify" in system.lower() or "conflict" in system.lower():
            return LLMResponse(
                text='{"action":"ADD","reason":"fake-llm"}', stop_reason="end"
            )

        # Otherwise, decide whether to call a tool or answer directly.
        if tools and _looks_like_question(lower):
            tool_call = ToolCall(
                id=f"call_{self.calls}",
                name="recall_memory",
                arguments={"query": last_user, "search_type": "semantic"},
            )
            return LLMResponse(text="", tool_calls=[tool_call], stop_reason="tool_use")

        # Direct text answer.
        return LLMResponse(text=_stub_answer(last_user, system), stop_reason="end")

    def format_tool_results(self, results: list[ToolResult]) -> dict:
        # Match the Anthropic-shaped wrapper so working memory + downstream
        # Anthropic adapters could consume the same conversation history.
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.tool_call_id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in results
            ],
        }


class FakeEmbedding:
    """Deterministic text → vector via hash expansion + normalization.

    Produces vectors of the configured dimension. Crucially, similar
    strings (sharing tokens) get correlated vectors — enough for the kit
    retrieval tests to be meaningful without a real embedding model.
    """

    name = "fake"

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        return _hash_vector(text, self.dimensions)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(t, self.dimensions) for t in texts]


# ── helpers ────────────────────────────────────────────────────────


def _hash_vector(text: str, dim: int) -> list[float]:
    """Token-bag hashed-feature vector, L2-normalized."""
    vec = [0.0] * dim
    tokens = [t for t in re.findall(r"\w+", text.lower()) if t]
    if not tokens:
        # Stable noise for empty text so vec0 still has something.
        h = hashlib.md5(b"empty").digest()
        for i in range(dim):
            vec[i] = (h[i % len(h)] / 255.0) - 0.5
    else:
        for tok in tokens:
            h = hashlib.md5(tok.encode()).digest()
            for i in range(dim):
                # Bipolar feature in [-0.5, 0.5].
                vec[i] += (h[i % len(h)] / 255.0) - 0.5
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # tool_result wrapper — skip those, find prior text user msg.
                continue
    return ""


def _first_user_text(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
    return ""


def _last_message_has_tool_results(messages: list[dict]) -> bool:
    if not messages:
        return False
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, list):
        return any(b.get("type") == "tool_result" for b in content)
    return False


def _looks_like_question(text: str) -> bool:
    return "?" in text or text.strip().startswith(
        ("what", "who", "when", "where", "why", "how", "do you", "did i", "remember")
    )


def _pick_goal(text: str) -> str:
    if not text:
        return "general"
    return " ".join(text.split()[:5])


_FACT_PATTERNS = [
    # "my favorite X is Y" → preference
    (r"my favou?rite (\w+) is ([\w\s]+)", "preference", "User's favourite {0} is {1}"),
    # "I have a X named Y" → fact
    (r"i have a (\w+) (?:named|called) ([\w\s]+)", "fact", "User has a {0} named {1}"),
    # "my name is X" / "I am X"
    (r"my name is (\w+)", "fact", "User's name is {0}"),
    (r"i am (\w+)\b", "fact", "User identifies as {0}"),
    # "I prefer X" / "I like X" → preference
    (r"i (?:prefer|like) ([\w\s]+)", "preference", "User prefers {0}"),
    # "I decided to X" → decision
    (r"i decided to ([\w\s]+)", "decision", "User decided to {0}"),
    # "actually X" / "now I want X" → correction
    (r"(?:actually|now) i want ([\w\s]+)", "correction", "User now wants {0}"),
]


def _extract_observations(text: str) -> list[dict]:
    """Pattern-match facts out of free text. Used by the FakeLLM during
    distillation prompts."""
    out: list[dict] = []
    lower = text.lower()
    for pattern, obs_type, template in _FACT_PATTERNS:
        for match in re.finditer(pattern, lower):
            groups = [g.strip().rstrip(".,!?") for g in match.groups()]
            content = template.format(*groups)
            importance = 8 if obs_type in ("decision", "correction") else 6
            out.append(
                {
                    "content": content,
                    "obs_type": obs_type,
                    "importance": importance,
                    "entities": [],
                    "emotional_valence": "neutral",
                    "emotional_intensity": 0,
                }
            )
    return out


def _stub_answer(question: str, system: str) -> str:
    """Synthesize a one-line answer using the briefing context if present."""
    # Look for any embedded "User..." facts in the system prompt that the
    # briefing would have injected.
    facts = re.findall(r"User[^.\n]{3,80}\.", system)
    if facts:
        return facts[-1].strip()
    if not question:
        return "Acknowledged."
    return f"Got it: {question.strip()}"
