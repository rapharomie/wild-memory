# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Wild Memory v4.0 — a biomimetic persistent memory framework for AI agents, organized as 6 "animal" layers (Salmon = identity, Bee = distillation, Elephant = retrieval, Dolphin = entity graph, Ant = decay/reflection, Chameleon = feedback/adaptation). Two storage backends ship: SQLite + sqlite-vec (`pip install "wild-memory[sqlite]"`) and Postgres + pgvector via asyncpg (`[postgres]` extra). Two LLM/embedding adapters: Anthropic + OpenAI. Python ≥3.10, Pydantic v2.

This is a clean-break OSS rewrite. The v3 codebase was tightly coupled to Supabase and to a Brazilian medical-residency sales product (Closi-AI / MedReview). All of that is gone — see `CHANGELOG.md`'s `4.0.0a1` entry for the full migration.

## Commands

```bash
# Install
pip install -e ".[sqlite,studio,dev]"      # local dev
pip install "wild-memory[sqlite,studio]"   # consumer install
pip install "wild-memory[postgres,studio]" # postgres backend

# CLI (after install)
wild-memory init                  # scaffold ./wild_memory.yaml + ./memory/
wild-memory migrate               # apply schema to the configured store
wild-memory info                  # print resolved config
wild-memory studio [--port N]     # web UI on http://127.0.0.1:5050
wild-memory test 1|2|3|all        # run a Test Kit in the terminal
                                  # add --live for real LLM/embedding
                                  # add --json out.json for a JSON report

# Tests
pytest tests/ -v
pytest tests/store/test_sqlite_smoke.py -v          # SQLite-only
DATABASE_URL=postgres://... pytest tests/store/     # Postgres parity too

# Single test
pytest tests/studio/test_kits.py::test_kit_passes_with_mock_providers -v
```

**Required env vars (live mode):** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. Optional: `DATABASE_URL` (auto-selects Postgres backend), `WILD_MEMORY_DB_PATH`, `WILD_MEMORY_STORE_KIND`. Mock mode (Studio + `wild-memory test` defaults) needs no API keys.

## Architecture

**Single entry point: `wild_memory.WildMemory`** (`wild_memory/orchestrator.py`).

```python
from wild_memory import WildMemory, WildMemoryConfig
from wild_memory.store import SQLiteStore  # or PostgresStore

store = SQLiteStore(":memory:", embedding_dim=1536)
await store.connect()
await store.migrate(embedding_dim=1536)

memory = WildMemory(WildMemoryConfig(), store=store)
reply = await memory.process_message(agent_id, user_id, session_id, message)
```

The constructor **requires an explicit `store=`** in v4 (no default — clean break). `WildMemory.from_config(path)` is async and builds the store from the YAML's `store:` section.

Methods:
- `process_message(agent_id, user_id, message, session_id) -> str` — full per-turn pipeline (semantic cache → checkpoint restore → Elephant context → LLM call with memory tools → tool-loop → citation → distillation → cache store → checkpoint → session log).
- `end_session(agent_id, user_id, session_id)` — flush distillation + run reflection.
- Cron methods: `run_daily_decay`, `run_daily_reflection`, `run_daily_feedback_analysis`, `run_cache_cleanup`, `run_session_cleanup`, `run_checkpoint_cleanup`.

**Boundaries that matter:**
- `wild_memory/store/` — `MemoryStore` ABC + `SQLiteStore`, `PostgresStore`. ~35 abstract methods grouped by entity (observations, entities, edges, reflections, feedback, procedures, citations, sessions, semantic_cache, checkpoints, imprints, lifecycle). The orchestrator never writes SQL — every consumer file calls `await self.store.<method>(...)`. Migrations live under `wild_memory/store/migrations/{sqlite,postgres}/` as Jinja-templated `.sql.j2` parameterized by `embedding_dim`.
- `wild_memory/providers/` — `LLMProvider` and `EmbeddingProvider` Protocols + `AnthropicLLM`/`OpenAIEmbedding` adapters + `LLMResponse`/`ToolCall`/`ToolResult` dataclasses. The orchestrator never imports `anthropic` or `openai` directly.
- `wild_memory/store/scoring.py` — single source of truth for the 5-signal Elephant score. Both backends call it (Postgres after pgvector ANN; SQLite after sqlite-vec ANN). Parity tests at `tests/store/test_parity.py` enforce identical ordering.

**Other layers (`wild_memory/`):**
- `layers/` — long-lived state per concept: `imprint` (Salmon), `working`, `observation`, `procedural`, `entity_graph` (Dolphin), `reflection` (Ant), `feedback` (Chameleon).
- `processes/` — `bee_distiller`, `distillation_gate`, `ant_decay`, `session_logger`, `ner_pipeline`.
- `retrieval/` — `elephant_recall`, `briefing_builder`, `briefing_cache`, `goal_cache`, `conflict_resolver`, `dynamic_recall`.
- `audit/` — `citation_logger`, `memory_audit` (privacy/right-to-be-forgotten lives here).
- `infra/` — `model_router` (premium/economy task routing), `embedding_cache` (per-turn dedup), `semantic_cache`, `checkpoint`. All four delegate to `MemoryStore` for persistence.
- `studio/` — Flask Blueprint mounted at `/`. `studio/kits/{kit1_smoke,kit2_lifecycle,kit3_scale,runner,fakes,reports}.py` runs the three Test Kits, defaulting to deterministic mock providers (`fakes.py`).
- `templates/` — package-data shipped to `wild-memory init`: `wild_memory.yaml.j2`, `imprint.yaml`, `procedures/example_workflow.md`.

**Cross-cutting patterns:**
- **Two-tier LLM routing.** `ModelRouter.call(task=...)` picks premium for `agent_response`, economy for `distillation`/`distillation_flush`/`reflection`/`goal_detection`/`conflict_resolution`/`feedback_analysis`/`entity_extraction`/`summary_compression`. Tune in `wild_memory.yaml → models`.
- **One embedding per turn.** `embedding_cache.embed(text)` is awaited once and reused across cache check, recall, conflict checks, distillation. `clear_turn()` resets at the end of each turn.
- **Provider-neutral tool loop.** The orchestrator's `_handle_response` operates on `LLMResponse.tool_calls` (list of `ToolCall` dataclasses); vendor adapters translate. Three built-in tools (`recall_memory`, `save_observation`, `update_entity`) defined in `wild_memory/tools.py` as JSON Schema.
- **Briefing cache invalidation.** Any write tool invalidates the briefing cache so the next turn rebuilds context.
- **Decay & TTL.** Per-type TTLs in `wild_memory.yaml`; `protected_types` (decision, correction) are exempt from automatic archive.
- **Distillation gate** filters trivial messages BEFORE distillation. Defaults are empty in v4 — configure `gate.trivial_patterns` and `gate.signal_keywords` for your language/domain.

## Test Kits — what they prove

The Studio at `/` exposes three pre-built kits. Each spins up a throwaway SQLite sandbox (deleted on completion) and runs against deterministic mock providers by default (so they're free + fast + always green when wiring is correct).

- **Kit 1 (smoke, ~50ms)** — teach 4 facts via the BeeDistiller, then ask 4 questions via ObservationLayer.retrieve. PASS if ≥3/4 facts are saved AND ≥3/4 retrievals return the right one.
- **Kit 2 (lifecycle, ~50ms)** — simulate 7 days of a single user. Inserts a preference, then a contradicting one (must be detected as similar), then a `decision` (protected type), runs decay 4×. PASS if ≥4/5 verifications succeed (conflict detected, old preference invalidated, new preference active, decision survived archive, chitchat archived).
- **Kit 3 (scale, ~50ms)** — 10 synthetic users in batches of 4 × 6 obs each × 3 recall queries. PASS if zero cross-user leakage, ≥80% queries return the exact target, p50 retrieval latency < 200ms.

`wild_memory/studio/kits/fakes.py` has a pattern-based FakeLLM that extracts observations from text via regex (used by the BeeDistiller flow) and a hash-based FakeEmbedding (deterministic, dimension-agnostic).

## Configuration

`wild_memory.yaml` (root, written by `wild-memory init`) controls runtime. Defaults live in `wild_memory/config.py` (Pydantic v2). YAML ↔ env var precedence: env vars override YAML for the storage backend (`DATABASE_URL`, `WILD_MEMORY_DB_PATH`, `WILD_MEMORY_STORE_KIND`); other settings are YAML-driven.

`memory/imprint.yaml` (also created by `init`) is the agent's permanent identity (Salmon layer). Edited only by humans; never overwritten at runtime. `memory/procedures/*.md` are procedural-memory step definitions.

## Repo quirks worth knowing

- **`memory/` and `wild_memory.yaml` are user artifacts**, not committed to the repo. The package ships templates under `wild_memory/templates/`; `wild-memory init` copies them out. The repo's `.gitignore` excludes both.
- **Per-backend migrations live in the package**, not at the repo root. `wild_memory/store/migrations/{sqlite,postgres}/` ships as package-data so `pip install + wild-memory migrate` works post-install.
- **Postgres parity tests skip without `DATABASE_URL`.** The `test-postgres` CI job sets it via a `pgvector/pgvector:pg16` service container.
- **The framework defaults to mock providers in Test Kits** so the green badge doesn't require API keys. Real providers are opt-in via `--live` (CLI) or by skipping `use_mock=True` in the Python API.

## Status

v4.0.0a1, 32 tests green, ready for OSS launch. Open items: studio "/inspect" tab (table viewer) and "/trace" tab (ad-hoc message debug) were planned but not built; CI workflow is scaffolded but `continue-on-error` on lint until codebase passes ruff cleanly. Both are tracked in CHANGELOG / GitHub issues going forward.
