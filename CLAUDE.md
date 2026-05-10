# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Wild Memory v3.0 — a biomimetic memory framework for AI agents, organized as 6 "animal" layers (Salmon = identity, Bee = distillation, Elephant = retrieval, Dolphin = entity graph, Ant = decay/reflection, Chameleon = feedback/adaptation). Backed by Supabase (PostgreSQL + pgvector). Python ≥3.10, Pydantic v2.

This checkout is also pre-configured for the **Closi-AI / MedReview** domain (Brazilian medical-residency sales agent). The domain-specific pieces are isolated in `wild_memory/medreview_domain.py` and `wild_memory/init_medreview.py`; everything else is domain-agnostic.

## Commands

```bash
# Install (editable + optional extras)
pip install -e .                      # core
pip install -e ".[dashboard]"         # + Flask dashboard
pip install -e ".[all]"               # + APScheduler

# Database — apply schema to Supabase
psql $DATABASE_URL < migrations/002_wild_memory_schema.sql
# or via CLI (uses wild_memory.yaml + env vars; falls back to manual SQL on failure)
wild-memory migrate

# CLI
wild-memory init        # scaffold wild_memory.yaml + memory/ files
wild-memory info        # print resolved config

# Tests (pytest)
python -m pytest tests/ -v
python -m pytest tests/test_wild_memory_setup.py::test_config_yaml_loads -v   # single test
```

**Required env vars** (loaded by `WildMemoryConfig.from_yaml`, override YAML):
`WILD_MEMORY_SUPABASE_URL`, `WILD_MEMORY_SUPABASE_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. The Closi-AI initializer (`init_medreview.py`) also accepts the unprefixed `SUPABASE_URL` / `SUPABASE_KEY`.

## Architecture

**Single entry point: `wild_memory.WildMemory`** (`wild_memory/orchestrator.py`). It wires every layer, retrieval component, background process, and infra singleton in `__init__`. The two methods callers normally use:

- `process_message(agent_id, user_id, message, session_id) -> str` — the full per-turn pipeline (semantic cache check → checkpoint restore → Elephant context build → LLM call with memory tools → tool-loop handling → citation log → distillation gate → cache store → checkpoint → session log).
- `end_session(agent_id, user_id, session_id)` — flush distillation + run reflection.

Cron methods on the same object: `run_daily_decay`, `run_daily_reflection`, `run_daily_feedback_analysis`, `run_cache_cleanup`, `run_session_cleanup`, `run_checkpoint_cleanup`.

**Code layout (under `wild_memory/`):**
- `layers/` — long-lived state per concept (`imprint`, `working`, `observation`, `procedural`, `entity_graph`, `reflection`, `feedback`).
- `processes/` — background work (`bee_distiller`, `distillation_gate`, `ant_decay`, `session_logger`, `ner_pipeline`).
- `retrieval/` — Elephant stack (`elephant_recall` is the 5-signal scorer; `briefing_builder` + `briefing_cache` assemble the system-prompt context; `goal_cache`, `conflict_resolver`, `dynamic_recall`).
- `infra/` — `db` (Supabase factory), `model_router` (premium vs economy routing for cost), `embedding_cache`, `semantic_cache`, `checkpoint`.
- `audit/` — citation + audit logs.
- `dashboard/` — self-contained Flask Blueprint mounted at `/wild-memory`; host wires it via `register_dashboard(app, adapter=...)` where the adapter supplies the Supabase client and agent_id.

**Key cross-cutting patterns:**
- **Two-tier LLM routing.** `ModelRouter.call(task=...)` selects premium (lead conversation) vs economy (distillation, goal detection, reflection, conflict checks). Tune in `wild_memory.yaml → models`.
- **Embeddings are computed once per turn.** `process_message` calls `embedding_cache.embed(message)` and reuses the vector through cache check, recall, and conflict checks; ends the turn with `embedding_cache.clear_turn()`.
- **Tool loop.** The LLM may emit `recall_memory`, `save_observation`, or `update_entity` tool calls (see `wild_memory/tools.py`). `_handle_response` runs at most 5 iterations; saves go through `ConflictResolver`, which can return `NOOP` to suppress duplicates.
- **Briefing cache invalidation.** Any write tool (`save_observation`, `update_entity`) invalidates the briefing cache so the next turn rebuilds context.
- **Decay & TTL.** Configured per observation type in `wild_memory.yaml`; `protected_types` (decision, correction) are exempt from decay above `protected_min_importance`. `AntDecay.run_daily()` applies it.
- **Distillation Gate** (`processes/distillation_gate.py`) filters trivial messages (regex + min length + signal keywords from config) before any LLM distillation call — protects token budget.

## Integration patterns

The `integration_examples/` directory shows four usage tiers (direct, shadow observer, context injector, full lifecycle). The Closi-AI–specific path is `wild_memory/init_medreview.py`, which lazy-imports, swallows missing-dependency errors, and swaps in the MedReview NER on the singleton. New integrations should follow the same shape (singleton + adapter), not import from `init_medreview.py`.

## Configuration

`wild_memory.yaml` (root) is the runtime config. Defaults live in `wild_memory/config.py` (Pydantic). YAML ↔ env var precedence: env vars override YAML for Supabase credentials only; everything else is YAML-driven.

`memory/imprint.yaml` is the agent's permanent identity (Salmon layer). Edited only by humans — never overwritten at runtime. `memory/procedures/*.md` are procedural-memory step definitions consumed by `ProceduralMemory`.

## Repo quirks worth knowing

- **`wild-memory/` (hyphen) at repo root is an untracked nested copy** of the entire project (with its own `.git/`) introduced by the `wild-memory-v3-uploaded-files` commit message. The source of truth is the top-level `wild_memory/` (underscore) Python package; do not edit files inside the nested `wild-memory/` directory.
- **`tests/test_wild_memory_setup.py` assumes a host project layout** — the integrity tests (`test_original_*_unchanged`) read `core/memory.py`, `core/database.py`, `agents/sales/agent.py`, `agents/sales/prompts/system_prompt.md`. Those files do not exist in this standalone checkout, so those tests will fail here; they only pass when this package is dropped inside the Closi-AI host repo.
- **`migrations/002_wild_memory_schema.sql` is the full schema.** It must be applied before any code runs; it creates all tables (`observations`, `entity_nodes`, `entity_edges`, `reflections`, `feedback_signals`, `procedures`, `citation_trails`, `session_logs`, `semantic_cache`, `agent_checkpoints`, `agent_imprints`, `broadcast_events`) plus pgvector and the RPCs the code calls (`reinforce_observation`, `apply_daily_decay`, `mark_stale_observations`, `retrieve_observations`, `search_semantic_cache`, `find_similar_observations`).
- Comments and config strings are mixed Portuguese/English (the framework is authored in Brazil); preserve language when editing existing strings.
