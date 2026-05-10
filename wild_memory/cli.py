"""
Wild Memory — CLI

Commands:
    wild-memory init      Scaffold wild_memory.yaml in the current directory
    wild-memory migrate   Apply schema to the configured store
    wild-memory info      Print resolved configuration

The full CLI rewrite (template-driven init from package-data, studio
launch, test kits) lands in Phase 6.
"""
from __future__ import annotations

import asyncio
import sys


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    cmd = sys.argv[1]
    if cmd == "init":
        print("Phase 6 will provide a full template-driven `init`.")
        print("For now, write a wild_memory.yaml manually with at least:")
        print("  store:\n    kind: sqlite\n    path: ./wild_memory.db")
    elif cmd == "migrate":
        asyncio.run(_run_migrate())
    elif cmd == "info":
        _print_info()
    else:
        print(f"Unknown command: {cmd}")
        print_help()


def print_help():
    print(
        """
Wild Memory CLI

Commands:
  wild-memory init      Scaffold wild_memory.yaml (Phase 6)
  wild-memory migrate   Connect to the configured store and apply the schema
  wild-memory info      Show the resolved configuration

Environment variables:
  ANTHROPIC_API_KEY        Required for the LLM
  OPENAI_API_KEY           Required for embeddings (default provider)
  DATABASE_URL             Postgres DSN (sets store.kind=postgres)
  WILD_MEMORY_DB_PATH      SQLite file path
  WILD_MEMORY_STORE_KIND   'sqlite' (default) or 'postgres'
"""
    )


async def _run_migrate():
    from wild_memory.config import WildMemoryConfig
    from wild_memory.orchestrator import _build_store_from_config

    config = WildMemoryConfig.from_yaml("wild_memory.yaml")
    print(f"Migrating store: kind={config.store.kind}")
    store = await _build_store_from_config(config)
    health = await store.health_check()
    print(f"Migration complete. Health: {health}")
    await store.close()


def _print_info():
    from wild_memory.config import WildMemoryConfig

    try:
        config = WildMemoryConfig.from_yaml("wild_memory.yaml")
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Hint: run `wild-memory init` first.")
        return
    print(
        f"""
Wild Memory v4

Store
  kind:           {config.store.kind}
  path:           {config.store.path}
  dsn:            {'<set>' if config.store.dsn else '<unset>'}

Models
  premium:        {config.models.premium.model}
  economy:        {config.models.economy.model}
  embedding:      {config.embedding.model} ({config.embedding.dimensions}d)

Memory tuning
  decay/day:      {config.decay.daily_rate}
  stale at:       {config.decay.stale_threshold}
  cache:          {'enabled' if config.cache.enabled else 'disabled'} (threshold: {config.cache.similarity_threshold})
  weights:        semantic={config.retrieval_weights.semantic} entity={config.retrieval_weights.entity_match} fts={config.retrieval_weights.fts_keyword}
"""
    )


if __name__ == "__main__":
    main()
