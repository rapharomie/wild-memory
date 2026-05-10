"""
Wild Memory — CLI

Commands:
    wild-memory init      Scaffold wild_memory.yaml in the current directory
    wild-memory migrate   Apply schema to the configured store
    wild-memory info      Print resolved configuration
    wild-memory studio    Launch the web Studio (kits dashboard)
    wild-memory test 1|2|3|all  Run a Test Kit in the terminal
"""
from __future__ import annotations

import asyncio
import json
import sys


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd == "init":
        print("Phase 6 will provide a full template-driven `init`.")
        print("For now, write a wild_memory.yaml manually with at least:")
        print("  store:\n    kind: sqlite\n    path: ./wild_memory.db")
    elif cmd == "migrate":
        asyncio.run(_run_migrate())
    elif cmd == "info":
        _print_info()
    elif cmd == "studio":
        _run_studio(rest)
    elif cmd == "test":
        sys.exit(_run_test(rest))
    else:
        print(f"Unknown command: {cmd}")
        print_help()


def print_help():
    print(
        """
Wild Memory CLI

Commands:
  wild-memory init                 Scaffold wild_memory.yaml (Phase 6)
  wild-memory migrate              Connect to the configured store, apply schema
  wild-memory info                 Show the resolved configuration
  wild-memory studio [--port N]    Launch the web Studio (kits dashboard)
  wild-memory test 1|2|3|all       Run a Test Kit in the terminal
                                   Add --json out.json to write a JSON report
                                   Add --live to use real LLM/embedding providers

Environment variables:
  ANTHROPIC_API_KEY        Required for live LLM mode
  OPENAI_API_KEY           Required for live embedding mode
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


def _run_studio(args: list[str]):
    port = 5050
    if "--port" in args:
        idx = args.index("--port")
        if idx + 1 < len(args):
            port = int(args[idx + 1])
    try:
        from wild_memory.studio.blueprint import create_app
    except ImportError as e:
        print(f"Studio requires Flask. Install with: pip install 'wild-memory[studio]'")
        print(f"({e})")
        sys.exit(1)
    app = create_app()
    print(f"Wild Memory Studio → http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


def _run_test(args: list[str]) -> int:
    if not args:
        print("Usage: wild-memory test 1|2|3|all [--json out.json] [--live]")
        return 2
    target = args[0]
    json_path = None
    use_mock = True  # default: mock providers (no API needed)
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
    print(f"\n=== Kit {report.kit_id} · {report.title} · {verdict} "
          f"({report.pass_count}/{report.total_count}, "
          f"{report.duration_seconds:.2f}s) ===")
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
