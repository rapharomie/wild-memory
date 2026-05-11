# Architecture

Wild Memory is six small layers wired together by a single orchestrator
(`WildMemory` in `wild_memory/orchestrator.py`). Each layer has a tight
single responsibility and talks to the rest only via well-typed
interfaces.

```
                    ┌─────────────┐
                    │  WildMemory │  process_message / end_session / cron
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Working  │ │ Elephant │ │   Bee    │
       │ memory   │ │ recall   │ │ distill  │
       └──────────┘ └──────────┘ └──────────┘
                          │            │
              ┌───────────┴────────────┴────────────┐
              │             MemoryStore             │
              │  observations · entities · edges    │
              │  reflections · feedback · citations │
              │  procedures · sessions · cache      │
              │  checkpoints · imprints             │
              └─────────────────────────────────────┘
                  ▲                          ▲
                  │                          │
             SQLiteStore               PostgresStore
           (sqlite-vec)                 (asyncpg+pgvector)
```

## The six layers

| Animal | Module | Job |
|--------|--------|-----|
| 🐟 **Salmon** | `layers/imprint.py` | Permanent agent identity (loaded from `memory/imprint.yaml`). Read-only at runtime. |
| 🐝 **Bee** | `layers/observation.py`, `processes/bee_distiller.py`, `processes/distillation_gate.py` | Filter trivial messages → distill the rest into atomic Observations. |
| 🐘 **Elephant** | `retrieval/elephant_recall.py`, `retrieval/briefing_builder.py`, `infra/semantic_cache.py` | Score memory by 5 signals (semantic, entity-match, FTS, recency, decay, emotion) → assemble briefing for the system prompt. |
| 🐬 **Dolphin** | `layers/entity_graph.py` | Track who/what is connected to who/what. |
| 🐜 **Ant** | `processes/ant_decay.py`, `layers/reflection.py` | Apply daily decay → mark stale → archive low-decay (with type-based protection). Reflection synthesizes patterns. |
| 🦎 **Chameleon** | `layers/feedback.py`, `layers/procedural.py`, `infra/checkpoint.py` | Capture outcome signals, run procedures, save state for crash recovery. |

## Key cross-cutting patterns

- **One embedding per turn.** `embedding_cache.embed()` is awaited once,
  the vector is reused across cache check, recall, conflict detection,
  and distillation. `clear_turn()` resets at the end.
- **Two-tier model routing.** `ModelRouter.call(task=…)` picks the
  premium model for `agent_response` and the economy model for
  everything else (distillation, reflection, conflict resolution, goal
  detection). ~70% cheaper.
- **Single-source scoring.** The 5-signal scorer lives in
  `wild_memory/store/scoring.py` and is used by both `SQLiteStore` and
  `PostgresStore`. Parity tests enforce identical ordering across
  backends.
- **Briefing cache invalidation.** Any tool call that writes (`save_observation`,
  `update_entity`) invalidates the briefing cache so the next turn
  rebuilds context.
- **Type-based decay protection.** `decision` and `correction`
  observations are exempt from automatic archive regardless of their
  decay score (configurable via `decay.protected_types`).

## Provider boundary

The orchestrator never imports `anthropic` or `openai`. It talks to
`LLMProvider` and `EmbeddingProvider` Protocols (`wild_memory/providers/base.py`)
that the adapters in `wild_memory/providers/anthropic_llm.py` and
`wild_memory/providers/openai_embedding.py` implement. Swap providers
by implementing those Protocols and passing your instance to
`WildMemory(config, llm=…, embedding=…)`.

## Storage boundary

The orchestrator never writes SQL. It calls `self.store.<method>()` on
a `MemoryStore` (`wild_memory/store/base.py`), and the concrete backend
takes it from there. Adding a new backend means subclassing the ABC and
implementing all 35 methods (none are optional). The parity test suite
at `tests/store/test_parity.py` ensures equivalent behavior.

## Lifecycle and async

Everything that touches storage is `async` end-to-end. The two backends
ship with proper async drivers (aiosqlite, asyncpg). The orchestrator
spawns distillation as a background task so it never blocks the
user-facing response.
