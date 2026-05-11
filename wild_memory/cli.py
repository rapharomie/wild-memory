"""
Wild Memory — CLI

Commands:
    wild-memory init      Scaffold wild_memory.yaml + memory/ in the current dir
    wild-memory migrate   Apply schema to the configured store
    wild-memory info      Print resolved configuration
    wild-memory studio    Launch the web Studio (kits dashboard)
    wild-memory test 1|2|3|all  Run a Test Kit in the terminal
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from importlib import resources
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd == "init":
        sys.exit(_run_init(rest))
    elif cmd == "migrate":
        asyncio.run(_run_migrate())
    elif cmd == "info":
        _print_info()
    elif cmd == "studio":
        _run_studio(rest)
    elif cmd == "test":
        sys.exit(_run_test(rest))
    elif cmd in ("--help", "-h", "help"):
        print_help()
    else:
        print(f"Unknown command: {cmd}")
        print_help()
        sys.exit(2)


def print_help():
    print(
        """
Wild Memory CLI

Commands:
  wild-memory init [--postgres DSN]   Scaffold wild_memory.yaml + memory/
  wild-memory migrate                 Connect to the configured store, apply schema
  wild-memory info                    Show the resolved configuration
  wild-memory studio [--port N]       Launch the web Studio (kits dashboard)
  wild-memory test 1|2|3|all          Run a Test Kit in the terminal
                                      Add --json out.json to write a JSON report
                                      Add --live to use real LLM/embedding providers

Environment variables:
  ANTHROPIC_API_KEY        Required for the LLM in live mode
  OPENAI_API_KEY           Required for embeddings in live mode
  DATABASE_URL             Postgres DSN (sets store.kind=postgres)
  WILD_MEMORY_DB_PATH      SQLite file path
  WILD_MEMORY_STORE_KIND   'sqlite' (default) or 'postgres'
"""
    )


# ── init ───────────────────────────────────────────────────────────────────


def _run_init(args: list[str]) -> int:
    """Scaffold wild_memory.yaml + memory/imprint.yaml + memory/procedures/."""
    postgres_dsn = ""
    force = False
    for i, a in enumerate(args):
        if a == "--postgres" and i + 1 < len(args):
            postgres_dsn = args[i + 1]
        elif a == "--force":
            force = True

    store_kind = "postgres" if postgres_dsn else "sqlite"
    store_path = "./wild_memory.db"

    targets = [
        (
            "wild_memory.yaml",
            _render_yaml(
                store_kind=store_kind,
                store_path=store_path,
                store_dsn=postgres_dsn,
                embedding_dim=1536,
            ),
        ),
        ("memory/imprint.yaml", _read_template_text("imprint.yaml")),
        (
            "memory/procedures/example_workflow.md",
            _read_template_text("procedures/example_workflow.md"),
        ),
    ]

    created, skipped = [], []
    for rel_path, content in targets:
        dst = Path(rel_path)
        if dst.exists() and not force:
            skipped.append(rel_path)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content)
        created.append(rel_path)

    for p in created:
        print(f"  created  {p}")
    for p in skipped:
        print(f"  skipped  {p}  (exists; pass --force to overwrite)")

    if created:
        print(
            "\nNext:\n"
            "  1. Edit memory/imprint.yaml with your agent's identity.\n"
            "  2. Set ANTHROPIC_API_KEY and OPENAI_API_KEY in your shell.\n"
            "  3. Run `wild-memory migrate` to create the schema.\n"
            "  4. Run `wild-memory studio` to verify with the Test Kits.\n"
        )
    return 0


def _render_yaml(*, store_kind: str, store_path: str, store_dsn: str, embedding_dim: int) -> str:
    raw = _read_template_text("wild_memory.yaml.j2")
    try:
        from jinja2 import Template
    except ImportError:
        # Fall back to a hand-rolled placeholder swap.
        return (
            raw.replace("{{ store_kind|default('sqlite') }}", store_kind)
            .replace("{{ store_path|default('./wild_memory.db') }}", store_path)
            .replace("{{ store_dsn|default('') }}", store_dsn)
            .replace("{{ embedding_dim|default(1536) }}", str(embedding_dim))
        )
    return Template(raw).render(
        store_kind=store_kind,
        store_path=store_path,
        store_dsn=store_dsn,
        embedding_dim=embedding_dim,
    )


def _read_template_text(rel_path: str) -> str:
    """Read a template file shipped as package-data inside wild_memory.templates."""
    parts = rel_path.split("/")
    pkg = "wild_memory.templates" + "".join(f".{p}" for p in parts[:-1])
    return resources.files(pkg).joinpath(parts[-1]).read_text()


# ── migrate ────────────────────────────────────────────────────────────────


async def _run_migrate():
    from wild_memory.config import WildMemoryConfig
    from wild_memory.orchestrator import _build_store_from_config

    config = WildMemoryConfig.from_yaml("wild_memory.yaml")
    print(f"Migrating store: kind={config.store.kind}")
    store = await _build_store_from_config(config)
    health = await store.health_check()
    print(f"Migration complete. Health: {health}")
    await store.close()


# ── info ───────────────────────────────────────────────────────────────────


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


# ── studio ─────────────────────────────────────────────────────────────────


def _run_studio(args: list[str]):
    port = 5050
    if "--port" in args:
        idx = args.index("--port")
        if idx + 1 < len(args):
            port = int(args[idx + 1])
    try:
        from wild_memory.studio.blueprint import create_app
    except ImportError as e:
        print("Studio requires Flask. Install with: pip install 'wild-memory[studio]'")
        print(f"({e})")
        sys.exit(1)
    app = create_app()
    print(f"Wild Memory Studio → http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


# ── test ───────────────────────────────────────────────────────────────────


def _run_test(args: list[str]) -> int:
    if not args:
        print("Usage: wild-memory test 1|2|3|all [--json out.json] [--live]")
        return 2
    target = args[0]
    json_path = None
    use_mock = True
    if "--json" in args:
        idx = args.index("--json")
        if idx + 1 < len(args):
            json_path = args[idx + 1]
    if "--live" in args:
        use_mock = False

    from wild_memory.studio.kits import KITS
    from wild_memory.studio.kits.runner import run_kit

    kit_ids = list(KITS.keys()) if target == "all" else [target]
    results = []
    overall_ok = True
    for kid in kit_ids:
        if kid not in KITS:
            print(f"Unknown kit: {kid}")
            return 2
        kit_fn, title, _ = KITS[kid]
        report = asyncio.run(run_kit(kit_fn, kit_id=kid, title=title, use_mock=use_mock))
        results.append(report.to_dict())
        _print_report(report)
        if not report.passed:
            overall_ok = False

    if json_path:
        with open(json_path, "w") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2)
        print(f"\nWrote JSON report to {json_path}")

    return 0 if overall_ok else 1


def _print_report(report) -> None:
    verdict = "PASS" if report.passed else "FAIL"
    print(
        f"\n=== Kit {report.kit_id} · {report.title} · {verdict} "
        f"({report.pass_count}/{report.total_count}, "
        f"{report.duration_seconds:.2f}s) ==="
    )
    for c in report.checks:
        mark = "✓" if c.passed else "✗"
        print(f"  {mark} {c.name}")
        if c.detail:
            print(f"      {c.detail}")
    if report.metrics:
        print("  Metrics:")
        for k, v in report.metrics.items():
            print(f"    {k}: {v}")
    if report.error:
        print(f"\n  ERROR: {report.error}")


if __name__ == "__main__":
    main()
