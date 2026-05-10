"""
Interface-level tests for MemoryStore + LegacySupabaseStore.

Real-database parity tests live in tests/store/test_parity.py and land in
Phase 3 alongside SQLiteStore. For now we just verify that the abstract
interface is stable and the legacy adapter implements every method.
"""
import inspect

from wild_memory.store.base import MemoryStore
from wild_memory.store._supabase_legacy import LegacySupabaseStore


def test_legacy_implements_full_interface():
    """LegacySupabaseStore must implement every abstract method on MemoryStore."""
    abstracts = {
        name
        for name, value in inspect.getmembers(MemoryStore)
        if getattr(value, "__isabstractmethod__", False)
    }
    assert abstracts, "MemoryStore should have abstract methods"

    legacy_methods = {
        name for name, _ in inspect.getmembers(LegacySupabaseStore, inspect.isfunction)
    }

    missing = abstracts - legacy_methods
    assert not missing, f"LegacySupabaseStore is missing: {sorted(missing)}"


def test_legacy_constructible_with_fake_client():
    """We can construct the adapter without a real supabase connection."""

    class FakeClient:
        pass

    store = LegacySupabaseStore(FakeClient())
    assert store.client is not None


def test_no_self_db_in_layers():
    """After Phase 2 no consumer should hold a `self.db` attribute or call
    `.table()/.rpc()` on its own. The store is the single point of access."""
    import importlib
    import re

    modules = [
        "wild_memory.layers.observation",
        "wild_memory.layers.reflection",
        "wild_memory.layers.entity_graph",
        "wild_memory.layers.feedback",
        "wild_memory.layers.procedural",
        "wild_memory.layers.imprint",
        "wild_memory.processes.session_logger",
        "wild_memory.processes.ant_decay",
        "wild_memory.processes.bee_distiller",
        "wild_memory.retrieval.conflict_resolver",
        "wild_memory.audit.citation_logger",
        "wild_memory.audit.memory_audit",
        "wild_memory.infra.checkpoint",
        "wild_memory.infra.semantic_cache",
    ]
    bad_pattern = re.compile(r"self\.db\.(table|rpc)\(")
    offenders = []
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        src = inspect.getsource(mod)
        if bad_pattern.search(src):
            offenders.append(mod_name)
    assert not offenders, f"These modules still call self.db directly: {offenders}"
