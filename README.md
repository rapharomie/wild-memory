# Wild Memory v3.0

**Biomimetic Memory Framework for AI Agents**

Created by Raphael Romie | MIT License

Inspired by 6 animals with extraordinary memory:

- **Salmon** — Identity (who I am)
- **Bee** — Distillation (what matters)
- **Elephant** — Retrieval (the right thing, at the right time)
- **Dolphin** — Connection (who relates to whom)
- **Ant** — Forgetting (what to release)
- **Chameleon** — Adaptation (how to improve)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

With dashboard support:

```bash
pip install -e ".[dashboard]"
```

### 2. Configure Supabase

Wild Memory uses **Supabase (PostgreSQL + pgvector)** as its database. You need:

1. A Supabase project (https://supabase.com)
2. Enable the **vector** and **pg_trgm** extensions in SQL Editor
3. Run the schema migration:

```bash
psql $DATABASE_URL < migrations/002_wild_memory_schema.sql
```

Or paste the contents of `migrations/002_wild_memory_schema.sql` into the Supabase SQL Editor and execute.

### 3. Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

Required variables:

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# LLM APIs
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Wild Memory Feature Flags
WILD_MEMORY_SHADOW=true     # Enable shadow observation mode
WILD_MEMORY_CONTEXT=true    # Enable context injection into prompts
WILD_MEMORY_AGENT_ID=my-agent  # Your agent's identifier
```

### 4. Configure Agent Identity

Edit `memory/imprint.yaml` to define your agent's identity, tone, values, and constraints.

Edit `wild_memory.yaml` to tune memory behavior (decay rates, retrieval weights, cache settings, etc).

### 5. Basic Usage

```python
from wild_memory import WildMemory

# Initialize from config file
memory = WildMemory.from_config("wild_memory.yaml")

# Process a message through the full pipeline
reply = await memory.process_message(
    agent_id="my-agent",
    user_id="user_123",
    message="Hello!",
    session_id="session_abc"
)

# End session (triggers full distillation + reflection)
await memory.end_session("my-agent", "user_123", "session_abc")
```

---

## Integration with Your Project

Wild Memory is designed to be **pluggable**. There are 4 integration patterns, from simplest to most complete:

### Pattern 1: Direct Usage (simplest)

```python
from wild_memory import WildMemory

memory = WildMemory.from_config("wild_memory.yaml")
reply = await memory.process_message(agent_id, user_id, message, session_id)
```

### Pattern 2: Shadow Observer (observation-only, zero risk)

The Shadow Observer watches your agent's conversations and distills knowledge in the background, without affecting responses.

```python
from integration_examples.shadow_observer import WildMemoryShadow

shadow = WildMemoryShadow()
# After your agent responds:
shadow.observe(session_id, user_message, assistant_response, user_id)
```

### Pattern 3: Context Injection (enrich prompts with memory)

Injects relevant past observations into the LLM system prompt.

```python
from integration_examples.context_injector import WildMemoryContextInjector

injector = WildMemoryContextInjector()
briefing = injector.get_context(session_id, user_message, user_id)
# briefing is a formatted string to inject into your system prompt, or None
```

### Pattern 4: Full Lifecycle (complete integration)

Handles escalations, session endings, and daily maintenance.

```python
from integration_examples.lifecycle_hooks import WildMemoryLifecycle

lifecycle = WildMemoryLifecycle()
lifecycle.on_escalation(session_id, user_id, metadata)
lifecycle.on_session_end(session_id, user_id, reason="reset")
results = lifecycle.run_daily_maintenance()
```

### Dashboard Integration (Flask)

```python
from flask import Flask
from wild_memory.dashboard import register_dashboard
from wild_memory.dashboard.adapter import WildMemoryAdapter

class MyAdapter(WildMemoryAdapter):
    def get_supabase_client(self):
        from myapp.db import supabase_client
        return supabase_client

    def get_agent_id(self):
        return "my-agent"

app = Flask(__name__)
register_dashboard(app, adapter=MyAdapter())
# Dashboard available at /wild-memory
```

---

## Project Structure

```
wild_memory_standalone/
|
|-- wild_memory/              # Core package
|   |-- __init__.py           # Entry point (WildMemory, WildMemoryConfig)
|   |-- orchestrator.py       # Main orchestrator connecting all 6 layers
|   |-- config.py             # Configuration (Pydantic models)
|   |-- models.py             # Data models
|   |-- tools.py              # LLM tool definitions
|   |-- cli.py                # CLI interface
|   |
|   |-- layers/               # Memory layers
|   |   |-- imprint.py        # Salmon: Agent identity
|   |   |-- working.py        # Working memory (per-session)
|   |   |-- observation.py    # Bee: Observation storage
|   |   |-- procedural.py     # Procedural memory
|   |   |-- entity_graph.py   # Dolphin: Entity connections
|   |   |-- reflection.py     # Ant: Pattern detection
|   |   |-- feedback.py       # Chameleon: Adaptation
|   |
|   |-- processes/            # Background processes
|   |   |-- bee_distiller.py  # Bee: LLM-powered distillation
|   |   |-- distillation_gate.py  # Filter trivial messages
|   |   |-- ant_decay.py      # Ant: Daily forgetting
|   |   |-- session_logger.py # Raw session capture
|   |   |-- ner_pipeline.py   # Named entity recognition
|   |
|   |-- retrieval/            # Elephant: Memory retrieval
|   |   |-- elephant_recall.py    # 5-signal combined retrieval
|   |   |-- briefing_builder.py   # Context briefing construction
|   |   |-- briefing_cache.py     # Briefing caching
|   |   |-- goal_cache.py         # Goal detection cache
|   |   |-- conflict_resolver.py  # Conflict detection
|   |   |-- dynamic_recall.py     # Tool-based recall
|   |
|   |-- infra/               # Infrastructure
|   |   |-- db.py             # Supabase client factory
|   |   |-- model_router.py   # LLM routing (premium/economy)
|   |   |-- embedding_cache.py  # Embedding cache
|   |   |-- semantic_cache.py   # Response cache
|   |   |-- checkpoint.py       # State checkpointing
|   |
|   |-- prompts/              # LLM prompt templates
|   |-- audit/                # Citation & audit logging
|   |-- dashboard/            # Web dashboard (Flask Blueprint)
|   |-- privacy/              # Privacy controls
|   |-- sync/                 # Multi-agent sync
|
|-- dashboard/                # Standalone dashboard copy
|-- memory/                   # Agent identity & procedures
|   |-- imprint.yaml          # Agent identity config
|   |-- procedures/           # Procedural memory files
|
|-- migrations/               # Database schema
|   |-- 002_wild_memory_schema.sql  # Complete PostgreSQL schema
|
|-- integration_examples/     # Integration patterns
|   |-- closi_adapter.py      # Example: Closi-AI adapter
|   |-- shadow_observer.py    # Example: Shadow observation
|   |-- context_injector.py   # Example: Context injection
|   |-- lifecycle_hooks.py    # Example: Full lifecycle
|
|-- tests/                    # Test suite
|-- docs/                     # Documentation
|
|-- wild_memory.yaml          # Framework configuration
|-- requirements.txt          # Python dependencies
|-- pyproject.toml            # Package configuration
|-- .env.example              # Environment variables template
```

---

## Cron Jobs

Wild Memory needs periodic maintenance. Schedule these daily:

```python
# Daily decay (Ant: pheromone evaporation)
await memory.run_daily_decay()

# Daily reflection (Ant: pattern detection)
await memory.run_daily_reflection(agent_id)

# Daily feedback analysis (Chameleon: adaptation)
await memory.run_daily_feedback_analysis(agent_id)

# Cache cleanup
await memory.run_cache_cleanup()
await memory.run_session_cleanup()
await memory.run_checkpoint_cleanup()
```

---

## Adapting for Your Project

1. **Copy** this entire directory into your project
2. **Run** the migration SQL on your Supabase
3. **Set** environment variables
4. **Edit** `memory/imprint.yaml` with your agent's identity
5. **Edit** `wild_memory.yaml` to tune behavior
6. **Create** your own adapter (see `integration_examples/closi_adapter.py`)
7. **Import** and use `WildMemory` in your agent code

The framework is designed to be **domain-agnostic**. The `medreview_domain.py` file is specific to the MedReview/Closi-AI domain and can be replaced or removed for your project.

---

## License

MIT License - Created by Raphael Romie
