"""
tests/unit/test_run_logging.py

Unit coverage for src/training/run_logging.py's RunLogger — the shared
TensorBoard + checkpoint infrastructure closing findings F1/F2 (2026-08-19
metrics audit: the ensemble weight sweep wrote zero artifacts, and
LightGBM training had no manifest, no checksums, and no mid-training
visibility).

All tests use tmp_path for both base_dir and checkpoint_base_dir so they
never touch the real repo's runs/ or checkpoints/ directories.
"""

from pathlib import Path

import pytest

from src.training.run_logging import RunLogger


@pytest.fixture
def run_logger(tmp_path: Path) -> RunLogger:
    logger = RunLogger(
        run_type="unittest",
        run_name="probe",
        base_dir=tmp_path / "runs",
        checkpoint_base_dir=tmp_path / "checkpoints",
    )
    yield logger
    logger.close()


@pytest.mark.unit
def test_run_logger_creates_distinct_log_and_checkpoint_dirs(run_logger: RunLogger, tmp_path: Path):
    assert run_logger.log_dir.exists()
    assert run_logger.checkpoint_dir.exists()
    assert run_logger.log_dir != run_logger.checkpoint_dir
    assert run_logger.log_dir.is_relative_to(tmp_path / "runs" / "unittest")
    assert run_logger.checkpoint_dir.is_relative_to(tmp_path / "checkpoints" / "unittest")


@pytest.mark.unit
def test_run_logger_run_name_includes_timestamp_for_uniqueness(tmp_path: Path):
    """Two RunLoggers with the same run_name must not collide — each gets
    its own directory (a UTC timestamp suffix), or one run's TensorBoard
    curves would silently overwrite another's."""
    a = RunLogger(run_type="t", run_name="same_name", base_dir=tmp_path / "runs",
                  checkpoint_base_dir=tmp_path / "ckpt")
    b = RunLogger(run_type="t", run_name="same_name", base_dir=tmp_path / "runs",
                  checkpoint_base_dir=tmp_path / "ckpt")
    try:
        assert a.log_dir != b.log_dir
        assert a.run_id != b.run_id
    finally:
        a.close()
        b.close()


@pytest.mark.unit
def test_log_scalar_writes_a_tensorboard_event_file(run_logger: RunLogger):
    """A TensorBoard event file must actually appear on disk after logging
    — this is the artifact `tensorboard --logdir runs/` reads."""
    run_logger.log_scalar("loss", 0.5, step=0)
    run_logger.log_scalar("loss", 0.3, step=1)
    run_logger.close()

    event_files = list(run_logger.log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) >= 1, (
        f"No TensorBoard event file found in {run_logger.log_dir} after "
        "log_scalar — TensorBoard would show nothing for this run."
    )


@pytest.mark.unit
def test_log_scalars_skips_none_values_without_raising(run_logger: RunLogger):
    """A metric that happens to be None (e.g. an undefined ratio) must be
    silently skipped, not crash the training loop it's called from."""
    run_logger.log_scalars({"a": 1.0, "b": None, "c": 2.0}, step=0)  # must not raise


@pytest.mark.unit
def test_checkpoint_path_is_deterministic_and_sortable(run_logger: RunLogger):
    p10 = run_logger.checkpoint_path(10, suffix="ubj")
    p2 = run_logger.checkpoint_path(2, suffix="ubj")
    p999 = run_logger.checkpoint_path(999, suffix="ubj")

    assert p10.parent == run_logger.checkpoint_dir
    assert p10.suffix == ".ubj"
    # Zero-padding must make lexicographic order match numeric step order —
    # prune_checkpoints relies on this.
    assert sorted([p999.name, p10.name, p2.name]) == [p2.name, p10.name, p999.name]


@pytest.mark.unit
def test_prune_checkpoints_keeps_only_the_most_recent_n(run_logger: RunLogger):
    for step in range(6):
        run_logger.checkpoint_path(step, suffix="txt").write_text("x")

    run_logger.prune_checkpoints(keep_last=3)

    remaining = sorted(p.name for p in run_logger.checkpoint_dir.glob("checkpoint_step_*"))
    expected = sorted(run_logger.checkpoint_path(s, suffix="txt").name for s in (3, 4, 5))
    assert remaining == expected, (
        f"Expected only the 3 most recent checkpoints to survive, got {remaining}"
    )


@pytest.mark.unit
def test_prune_checkpoints_keep_last_zero_or_negative_is_a_noop(run_logger: RunLogger):
    """keep_last<=0 must never be interpreted as 'delete everything' — a
    caller passing 0 by mistake should not wipe a run's checkpoints."""
    for step in range(3):
        run_logger.checkpoint_path(step, suffix="txt").write_text("x")

    run_logger.prune_checkpoints(keep_last=0)

    remaining = list(run_logger.checkpoint_dir.glob("checkpoint_step_*"))
    assert len(remaining) == 3


@pytest.mark.unit
def test_run_logger_is_a_context_manager(tmp_path: Path):
    with RunLogger(
        run_type="t", run_name="ctx", base_dir=tmp_path / "runs",
        checkpoint_base_dir=tmp_path / "ckpt",
    ) as rl:
        rl.log_scalar("x", 1.0, step=0)
        log_dir = rl.log_dir
    # __exit__ must have closed the writer without raising, and the
    # directory (with its event file) must persist after the block exits.
    assert log_dir.exists()
