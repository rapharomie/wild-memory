"""
Wild Memory — quickstart with SQLite + sqlite-vec.

Runs entirely on your laptop. No cloud. The only external services are the
LLM and embedding APIs (Anthropic + OpenAI by default).

Usage:
    pip install "wild-memory[sqlite]"
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    python examples/01_quickstart_sqlite.py
"""
from __future__ import annotations

import asyncio

from wild_memory import WildMemory, WildMemoryConfig
from wild_memory.store import SQLiteStore


async def main():
    # 1. Open + migrate a SQLite-backed store. Put it in memory for the demo.
    store = SQLiteStore(":memory:", embedding_dim=1536)
    await store.connect()
    await store.migrate(embedding_dim=1536)

    # 2. Build the orchestrator. Default LLM = AnthropicLLM, default
    #    embedding = OpenAIEmbedding (both read API keys from env).
    config = WildMemoryConfig()
    memory = WildMemory(config, store=store)

    agent_id, user_id, session_id = "demo", "alice", "demo-session"

    # 3. First turn — teach a fact.
    reply1 = await memory.process_message(
        agent_id=agent_id,
        user_id=user_id,
        session_id=session_id,
        message="My favorite color is blue. Please remember that.",
    )
    print("turn 1:", reply1)

    # 4. Second turn — ask about it. The Elephant should pull the saved
    #    observation into context, and the model should answer "blue".
    reply2 = await memory.process_message(
        agent_id=agent_id,
        user_id=user_id,
        session_id=session_id,
        message="What is my favorite color?",
    )
    print("turn 2:", reply2)

    # 5. End the session — flushes any remaining distillation + reflection.
    await memory.end_session(agent_id, user_id, session_id)
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
