"""
Interface-level tests for MemoryStore.

Verify the abstract interface is stable and that every shipped backend
(SQLite, Postgres) implements every abstract method. Per-backend
behavior tests live in tests/store/test_<backend>_*.py.
"""
import inspect

from wild_memory.store.base import MemoryStore


def _abstract_methods() -> set[str]:
    return {
        name
        for name, value in inspect.getmembers(MemoryStore)
        if getattr(value, "__isabstractmethod__", False)
    }


def test_sqlite_implements_full_interface():
    from wild_memory.store.sqlite import SQLiteStore

    abstracts = _abstract_methods()
    assert abstracts
    methods = {name for name, _ in inspect.getmembers(SQLiteStore, inspect.isfunction)}
    missing = abstracts - methods
    assert not missing, f"SQLiteStore is missing: {sorted(missing)}"


def test_postgres_implements_full_interface():
    from wild_memory.store.postgres import PostgresStore

    abstracts = _abstract_methods()
    assert abstracts
    methods = {name for name, _ in inspect.getmembers(PostgresStore, inspect.isfunction)}
    missing = abstracts - methods
    assert not missing, f"PostgresStore is missing: {sorted(missing)}"


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
