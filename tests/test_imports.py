"""Smoke tests: the package imports cleanly without optional vendor SDKs."""


def test_top_level_imports():
    from wild_memory import WildMemory, WildMemoryConfig  # noqa: F401


def test_provider_protocols_importable():
    from wild_memory.providers.base import (
        EmbeddingProvider,
        LLMProvider,
        LLMResponse,
        ToolCall,
        ToolResult,
    )
    # Sanity: the dataclasses are constructible.
    r = LLMResponse(text="hello")
    assert r.tool_calls == []
    assert r.stop_reason == "end"

    c = ToolCall(id="t1", name="recall_memory", arguments={"query": "x"})
    assert c.arguments == {"query": "x"}

    res = ToolResult(tool_call_id="t1", content="result")
    assert res.is_error is False


def test_task_tier_map_has_no_domain_specific_keys():
    """The router's task names must be domain-neutral after the v4 cleanup."""
    from wild_memory.infra.model_router import TASK_TIER_MAP

    for forbidden in ("lead_conversation", "complex_objection_handling", "bee_distill", "flush_distill"):
        assert forbidden not in TASK_TIER_MAP, (
            f"Task name '{forbidden}' is domain-coupled and must not be in the router map"
        )
    assert "agent_response" in TASK_TIER_MAP
    assert "distillation" in TASK_TIER_MAP
    assert "distillation_flush" in TASK_TIER_MAP


def test_no_supabase_import_in_providers():
    """The providers package must not pull in supabase or other backend SDKs."""
    import sys

    if "wild_memory.providers" in sys.modules:
        del sys.modules["wild_memory.providers"]
    import wild_memory.providers  # noqa: F401

    # We can't assert that `supabase` is absent globally (other modules may have
    # imported it), but the providers module itself must not depend on it.
    import wild_memory.providers.base as base
    import wild_memory.providers.anthropic_llm as anth
    import wild_memory.providers.openai_embedding as oai
    for mod in (base, anth, oai):
        src = open(mod.__file__).read()
        assert "supabase" not in src.lower(), f"{mod.__name__} should not reference supabase"
