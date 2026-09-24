"""
conftest.py — Shared pytest fixtures for all test modules.

Fixtures here are available to every test file without explicit imports.
Grows as phases are completed — Phase 0 provides the config fixture baseline.
"""

import pathlib
from typing import Any

import pytest
import yaml


PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers used across the suite so `-m` selection works cleanly."""
    config.addinivalue_line("markers", "unit: fast, isolated test with no external dependencies")
    config.addinivalue_line("markers", "integration: test spanning multiple modules or the filesystem")
    config.addinivalue_line("markers", "performance: latency/throughput test against real model artifacts")


@pytest.fixture(scope="session")
def project_root() -> pathlib.Path:
    """Absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """
    Load and return config/config.yaml as a dict.

    Scope=session so config is parsed once per pytest run, not per test.
    """
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    assert config_path.exists(), f"config.yaml not found at {config_path}"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
