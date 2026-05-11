"""
Wild Memory — Postgres + pgvector (asyncpg).

Usage:
    pip install "wild-memory[postgres]"
    export DATABASE_URL=postgres://user:pw@host/db
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    python examples/02_postgres.py
"""
from __future__ import annotations

import asyncio
import os

from wild_memory import WildMemory, WildMemoryConfig
from wild_memory.store import PostgresStore


async def main():
    dsn = os.environ["DATABASE_URL"]
    store = PostgresStore(dsn, embedding_dim=1536)
    await store.connect()
    await store.migrate(embedding_dim=1536)

    memory = WildMemory(WildMemoryConfig(), store=store)
    print(await memory.process_message(
        agent_id="demo",
        user_id="bob",
        session_id="s1",
        message="Remember: I work in marine biology.",
    ))
    print(await memory.process_message(
        agent_id="demo",
        user_id="bob",
        session_id="s1",
        message="What field am I in?",
    ))

    await memory.end_session("demo", "bob", "s1")
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
