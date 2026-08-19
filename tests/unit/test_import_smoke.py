"""
tests/unit/test_import_smoke.py

Phase D5 clean-env import smoke test (docs/IMPLEMENTATION_PLAN.md).

Background: `scripts/run_phase2_eval.py` imported `catboost`, which was
never declared in requirements.txt — invisible in day-to-day dev (catboost
happened to already be installed locally) but a guaranteed ImportError on a
fresh `pip install -r requirements.txt` followed by running the script
(the Docker image would fail the same way). No test caught it because
nothing imported every module the way a clean install + a cold run would.

This test walks every top-level Python module under src/ and every
standalone script under scripts/ and imports it, so a missing or
undeclared dependency fails `pytest` in CI instead of only surfacing the
first time someone happens to run that specific script. Every scripts/*.py
file guards its entry point behind `if __name__ == "__main__":`, so
importing them here has no side effects (no data/network/model access).

Runnable in CI: no network, no data files, no GPU required.
"""

import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import src  # noqa: E402


def _discover_src_modules() -> List[str]:
    """Every importable dotted module name under the `src` package.

    pkgutil.walk_packages defaults to onerror=None, which SWALLOWS
    ImportError raised while it imports a subpackage's __init__.py to read
    its __path__ (there are 13 subpackages under src/, e.g. src/api,
    src/training) — the exact "missing/undeclared dependency" failure mode
    this whole test file exists to catch would silently vanish from the
    discovered list instead of failing CI. Pass an onerror that re-raises
    so a broken subpackage fails loudly here instead.
    """
    def _reraise(_name: str) -> None:
        raise

    names = [src.__name__]
    for _, name, _is_pkg in pkgutil.walk_packages(
        src.__path__, prefix=f"{src.__name__}.", onerror=_reraise
    ):
        names.append(name)
    return sorted(names)


def _discover_scripts() -> List[Path]:
    """Every standalone script under scripts/ (not a package — no __init__.py)."""
    scripts_dir = PROJECT_ROOT / "scripts"
    return sorted(scripts_dir.glob("*.py"))


SRC_MODULE_NAMES = _discover_src_modules()
SCRIPT_PATHS = _discover_scripts()


@pytest.mark.parametrize("module_name", SRC_MODULE_NAMES)
def test_src_module_imports_cleanly(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize("script_path", SCRIPT_PATHS, ids=lambda p: p.name)
def test_script_imports_cleanly(script_path: Path) -> None:
    """Load each scripts/*.py file as a module (they aren't a package) and
    import it — this is exactly the step that would have caught the
    undeclared `catboost` import in scripts/run_phase2_eval.py."""
    spec = importlib.util.spec_from_file_location(
        f"_import_smoke_{script_path.stem}", script_path
    )
    assert spec is not None and spec.loader is not None, (
        f"Could not build an import spec for {script_path}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_at_least_one_src_module_and_one_script_were_discovered() -> None:
    """Guards against this test silently doing nothing if the discovery
    globs above ever stop matching anything (e.g. a directory rename)."""
    assert len(SRC_MODULE_NAMES) > 10
    assert len(SCRIPT_PATHS) > 0
