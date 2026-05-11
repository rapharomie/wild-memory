# Wild Memory

**Persistent, opinionated memory for AI agents.** Think six small layers
working together: identity, distillation, retrieval, an entity graph,
forgetting, and feedback-driven adaptation.

```bash
pip install "wild-memory[sqlite,studio]"
wild-memory init
wild-memory studio                # http://127.0.0.1:5050 — runs the test kits
```

No cloud needed. Anthropic + OpenAI for the LLM and embeddings; SQLite
or Postgres for persistence; everything else is in-process.

---

## What you get

- **Pluggable storage.** SQLite + `sqlite-vec` for local. Postgres +
  `pgvector` for shared. Both behind one `MemoryStore` interface; bring
  your own backend by subclassing it.
- **Pluggable LLMs and embeddings.** `LLMProvider` / `EmbeddingProvider`
  Protocols + Anthropic and OpenAI adapters. Swap to any vendor by
  implementing the Protocol (no fork required).
- **Studio web UI** with three Test Kits — pre-built scenarios that
  prove the memory actually works (saves the right thing, retrieves the
  right thing, forgets the right thing, scales to many users without
  leakage). Click "Run", see PASS/FAIL.
- **Domain-neutral defaults.** No hardcoded language or business
  vocabulary in the framework. You configure what matters in your
  `wild_memory.yaml`.

## Quickstart (60 seconds)

```bash
pip install "wild-memory[sqlite,studio]"
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

wild-memory init        # writes ./wild_memory.yaml + ./memory/imprint.yaml
wild-memory migrate     # creates wild_memory.db, loads sqlite-vec
wild-memory studio      # opens http://127.0.0.1:5050 with the Test Kits
```

Or skip the YAML and use the package directly:

```python
import asyncio
from wild_memory import WildMemory, WildMemoryConfig
from wild_memory.store import SQLiteStore

async def main():
    store = SQLiteStore(":memory:", embedding_dim=1536)
    await store.connect()
    await store.migrate(embedding_dim=1536)

    memory = WildMemory(WildMemoryConfig(), store=store)
    print(await memory.process_message(
        agent_id="demo", user_id="alice", session_id="s1",
        message="My favorite color is blue. Remember that.",
    ))
    print(await memory.process_message(
        agent_id="demo", user_id="alice", session_id="s1",
        message="What's my favorite color?",
    ))
    await store.close()

asyncio.run(main())
```

More worked examples: `examples/01_quickstart_sqlite.py`,
`examples/02_postgres.py`, `examples/03_custom_provider.py`.

## The Test Kits

Three pre-built kits answer three questions. Each spins up a throwaway
SQLite sandbox so they're safe to run anywhere — and they default to
mock providers so you don't even need API keys for the green badge.

| Kit | Question | What it does | Time |
|-----|----------|--------------|------|
| 1 | **Funciona?** (Does it work?) | Teach 4 facts, ask 4 questions, verify each one comes back. | ~1s |
| 2 | **Funciona ao longo do tempo?** (Does it work over time?) | Simulate a week — preferences, contradictions, decay, archive. Decisions survive; chitchat fades. | ~1s |
| 3 | **Aguenta volume?** (Does it scale?) | 10 parallel users × 6 observations × 3 recalls. Zero cross-user leakage required. | ~1s |

```bash
wild-memory test 1            # one kit
wild-memory test all          # all three
wild-memory test 1 --live     # use Anthropic+OpenAI for real
wild-memory studio            # web UI for the same kits
```

## Architecture, in 30 seconds

Six layers, named after animals with extraordinary memory:

- 🐟 **Salmon** — identity (who I am)
- 🐝 **Bee** — distillation (what matters; gate filters trivial messages, distiller turns the rest into observations)
- 🐘 **Elephant** — retrieval (the right thing, at the right time; 5-signal scoring across semantic similarity, entity match, full-text, recency, decay, emotional valence)
- 🐬 **Dolphin** — entity graph (who relates to whom)
- 🐜 **Ant** — forgetting (decay rates, archive thresholds, type-based protection)
- 🦎 **Chameleon** — feedback / adaptation (procedures, citations, semantic cache, checkpoints)

All six talk to a single `MemoryStore` and use a single `LLMProvider` /
`EmbeddingProvider` pair. See `docs/architecture.md` for the full map.

## Configuration

`wild-memory init` writes a fully-commented `wild_memory.yaml` and
`memory/imprint.yaml` into your current directory. Edit those.

The `store:` section picks the backend:

```yaml
store:
  kind: sqlite              # or "postgres"
  path: ./wild_memory.db    # for sqlite
  dsn: ""                   # for postgres (or set DATABASE_URL)
```

Env vars override:

| Variable | Effect |
|----------|--------|
| `DATABASE_URL` | Auto-selects Postgres backend, fills `store.dsn` |
| `WILD_MEMORY_DB_PATH` | Override SQLite path |
| `WILD_MEMORY_STORE_KIND` | Force `sqlite` or `postgres` |
| `ANTHROPIC_API_KEY` | LLM provider (default Anthropic) |
| `OPENAI_API_KEY` | Embedding provider (default OpenAI) |

## Adding your own backend or provider

`MemoryStore` is a Python ABC with ~35 methods grouped by entity. Subclass
it, pass parity tests at `tests/store/test_parity.py`, you're in.

`LLMProvider` and `EmbeddingProvider` are Protocols (one async method each
in the embedding case; two in the LLM case). The orchestrator only ever
sees `LLMResponse` dataclasses, so any vendor can plug in.

See `CONTRIBUTING.md` for the full contract.

## Project layout

```
wild_memory/
  orchestrator.py         # WildMemory — the entry point
  config.py               # Pydantic config (StoreConfig, ModelsConfig, ...)
  models.py
  tools.py                # Tool schemas (recall_memory, save_observation, update_entity)
  layers/                 # imprint, working, observation, procedural, entity_graph, reflection, feedback
  processes/              # bee_distiller, distillation_gate, ant_decay, session_logger, ner_pipeline
  retrieval/              # elephant_recall, briefing_builder, briefing_cache, goal_cache, conflict_resolver
  audit/                  # citation_logger, memory_audit
  infra/                  # model_router, embedding_cache, semantic_cache, checkpoint
  providers/              # base + AnthropicLLM + OpenAIEmbedding
  store/                  # base (ABC), scoring, sqlite, postgres + per-backend migrations
  studio/                 # Flask blueprint + the three Test Kits
  templates/              # Templates copied by `wild-memory init`
examples/                 # Runnable demos
tests/                    # 32 passing
docs/                     # Quickstart, architecture, backends, providers
```

## Status

**v4.0 (alpha).** Domain-neutral, two backends, two providers, one Studio,
three test kits, 32 tests green. The `dashboard` extra (now `[studio]`)
gives you the visual UI; the package itself is small.

## License

MIT — Raphael Romie, São Paulo. Originally extracted from a sales-agent
product (Closi-AI) and stripped to a domain-agnostic framework. See
`CHANGELOG.md` for the v4 reset.
