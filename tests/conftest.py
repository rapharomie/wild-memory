"""Test configuration. asyncio_mode=auto so async tests don't need a decorator."""
import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-mark async tests with @pytest.mark.asyncio (asyncio_mode=auto equivalent)."""
    import asyncio
    import inspect
    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.asyncio)
