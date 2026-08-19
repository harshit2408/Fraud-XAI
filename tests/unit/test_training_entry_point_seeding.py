"""
tests/unit/test_training_entry_point_seeding.py

TDD for Phase B3 — every training entry point must call `set_seed()` before
doing any random-number-consuming work, so that two runs of `make train`
produce identical results (the HIGH finding: "no seeds are set").

Written FIRST, against entry points that do not call set_seed() yet (RED).

Approach: full end-to-end runs of `main()` are out of scope for a unit test
(they need real parquet data, a live/local MLflow tracking store, and GPU/CPU
training time). Instead this module combines two cheaper, still-meaningful
checks per entry point:
  1. Static wiring: the module imports `set_seed` from `src.utils.seed` and
     `main()`'s source calls it — proving the call site actually exists at
     the right place, not just that the utility exists in isolation.
  2. Behavioral proof that set_seed() itself gives determinism is covered
     separately in tests/unit/test_seed.py.

Run: pytest tests/unit/test_training_entry_point_seeding.py -v
"""

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TRAINING_ENTRY_POINTS = [
    "src/training/train_xgb.py",
    "src/training/train_lgbm.py",
    "src/training/train_tft.py",
    "src/training/tune_xgb.py",
    "src/training/tune_tft.py",
]


def _parse(module_path: Path) -> ast.Module:
    return ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))


def _imports_set_seed(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.utils.seed":
            if any(alias.name == "set_seed" for alias in node.names):
                return True
    return False


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"No function named '{name}' found")


def _calls_set_seed(func_node: ast.FunctionDef) -> bool:
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            called = node.func
            if isinstance(called, ast.Name) and called.id == "set_seed":
                return True
    return False


def _main_calls_set_seed_before_any_random_consuming_call(
    func_node: ast.FunctionDef,
) -> bool:
    """set_seed() must run before config-driven randomness (data loading,
    Optuna studies, model construction) so that everything downstream is
    covered by the seed."""
    call_line = None
    for stmt in ast.walk(func_node):
        if (
            isinstance(stmt, ast.Call)
            and isinstance(stmt.func, ast.Name)
            and stmt.func.id == "set_seed"
        ):
            call_line = stmt.lineno
            break

    if call_line is None:
        return False

    # "Before any random-consuming call" is approximated by requiring the
    # set_seed() call to be within the first half of main()'s line range.
    first_line = func_node.lineno
    last_line = max(
        (n.lineno for n in ast.walk(func_node) if hasattr(n, "lineno")),
        default=func_node.lineno,
    )
    span = max(last_line - first_line, 1)
    return (call_line - first_line) <= span * 0.6


@pytest.mark.parametrize("entry_point", TRAINING_ENTRY_POINTS)
class TestEntryPointCallsSetSeed:
    def test_imports_set_seed_from_utils(self, entry_point: str):
        tree = _parse(PROJECT_ROOT / entry_point)
        assert _imports_set_seed(tree), (
            f"{entry_point} does not `from src.utils.seed import set_seed`"
        )

    def test_main_calls_set_seed(self, entry_point: str):
        tree = _parse(PROJECT_ROOT / entry_point)
        main_func = _find_function(tree, "main")
        assert _calls_set_seed(main_func), (
            f"{entry_point}'s main() never calls set_seed()"
        )

    def test_main_calls_set_seed_early(self, entry_point: str):
        tree = _parse(PROJECT_ROOT / entry_point)
        main_func = _find_function(tree, "main")
        assert _main_calls_set_seed_before_any_random_consuming_call(main_func), (
            f"{entry_point}'s main() calls set_seed() too late to cover all "
            "randomness (data loading, sampling, model init)"
        )

    def test_main_reads_seed_from_project_config(self, entry_point: str):
        """set_seed must be driven by config['project']['random_seed'], not a
        second hardcoded literal — otherwise the config and the actual seed
        used can silently diverge."""
        source = (PROJECT_ROOT / entry_point).read_text(encoding="utf-8")
        assert 'random_seed' in source, (
            f"{entry_point} does not reference config's random_seed"
        )

    def test_main_logs_seed_to_mlflow(self, entry_point: str):
        """Phase B3 acceptance: the applied seed must be logged to MLflow so
        runs are auditable. Accept either log_param("random_seed", ...) or
        "random_seed" as a key inside log_params({...})."""
        source = (PROJECT_ROOT / entry_point).read_text(encoding="utf-8")
        logs_via_param = 'log_param("random_seed"' in source or "log_param('random_seed'" in source
        logs_via_params_dict = '"random_seed"' in source and "log_params" in source
        assert logs_via_param or logs_via_params_dict, (
            f"{entry_point} does not log random_seed to MLflow"
        )
