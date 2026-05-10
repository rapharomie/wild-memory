"""
Flask blueprint for Wild Memory Studio.

Mounts the test-kit dashboard at `/`. Delegates all business logic to
`wild_memory.studio.kits` so the same kits can be invoked from the CLI
or from CI without booting Flask.
"""
from __future__ import annotations

import asyncio
import os
from functools import wraps

try:
    from flask import Blueprint, jsonify, render_template
except ImportError:  # pragma: no cover - importable without flask installed
    Blueprint = None  # type: ignore[assignment]

from wild_memory.studio.kits import KITS
from wild_memory.studio.kits.runner import run_kit


def create_blueprint():
    """Build and return the Flask blueprint. Raises if Flask isn't installed."""
    if Blueprint is None:
        raise ImportError(
            "Flask is required to mount the studio. Install with "
            "`pip install wild-memory[dashboard]`."
        )

    bp = Blueprint(
        "wild_memory_studio",
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )

    use_mock_default = (
        not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENAI_API_KEY"))
    )

    @bp.route("/")
    def home():
        kit_cards = [
            {"id": kid, "title": title, "description": desc}
            for kid, (_, title, desc) in KITS.items()
        ]
        return render_template(
            "kits.html",
            kit_cards=kit_cards,
            provider_mode="mock" if use_mock_default else "live",
        )

    @bp.route("/api/kit/<kit_id>/run", methods=["POST"])
    def run(kit_id: str):
        if kit_id not in KITS:
            return jsonify({"error": f"unknown kit: {kit_id}"}), 404
        kit_fn, title, _desc = KITS[kit_id]
        report = asyncio.run(
            run_kit(kit_fn, kit_id=kit_id, title=title)
        )
        return jsonify(report.to_dict())

    @bp.route("/api/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "kits": list(KITS.keys()),
                "provider_mode_default": "mock" if use_mock_default else "live",
            }
        )

    return bp


def create_app():
    """Standalone Flask app (used by `wild-memory studio`)."""
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(create_blueprint())
    return app
