# Quickstart

```bash
pip install "wild-memory[sqlite,studio]"
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

wild-memory init        # ./wild_memory.yaml + ./memory/imprint.yaml
wild-memory migrate     # creates wild_memory.db
wild-memory studio      # http://127.0.0.1:5050 — Test Kits
```

That's the whole loop. The Studio's three kits should all turn green.

## Verify in 5 lines of Python

```python
import asyncio
from wild_memory import WildMemory, WildMemoryConfig
from wild_memory.store import SQLiteStore

async def main():
    store = SQLiteStore(":memory:", embedding_dim=1536)
    await store.connect()
    await store.migrate(embedding_dim=1536)
    memory = WildMemory(WildMemoryConfig(), store=store)

    await memory.process_message("agent", "alice", "s1",
        "Remember: my favorite color is blue.")
    print(await memory.process_message("agent", "alice", "s1",
        "What's my favorite color?"))

    await memory.end_session("agent", "alice", "s1")
    await store.close()

asyncio.run(main())
```

## Common knobs

- **Different LLM / embedding model** — edit `models:` and `embedding:` in
  `wild_memory.yaml`. Dimensions matter: changing `embedding.dimensions`
  means re-running `wild-memory migrate` against a fresh database (the
  store will fail-fast on connect if the meta table disagrees).
- **Postgres instead of SQLite** — `pip install "wild-memory[postgres]"`,
  set `DATABASE_URL`, run `wild-memory migrate`.
- **Custom LLM provider** — see `examples/03_custom_provider.py`.
- **Custom backend** — subclass `wild_memory.store.MemoryStore`, pass the
  parity tests at `tests/store/test_parity.py`.

## What's next

- `docs/architecture.md` — the six layers in more detail.
- `docs/backends.md` — SQLite vs Postgres trade-offs, how to add a third.
- `docs/providers.md` — writing an LLM/embedding adapter.
