"""Report dataclasses for kit results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Check:
    """A single pass/fail assertion within a kit."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class KitReport:
    """Final result of a kit run."""

    kit_id: str
    title: str
    passed: bool
    duration_seconds: float
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_count(self) -> int:
        return len(self.checks)

    def to_dict(self) -> dict:
        return {
            "kit_id": self.kit_id,
            "title": self.title,
            "passed": self.passed,
            "duration_seconds": round(self.duration_seconds, 2),
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
            "metrics": self.metrics,
            "log": self.log,
            "error": self.error,
            "pass_count": self.pass_count,
            "total_count": self.total_count,
        }
