"""
tests/unit/test_feature_engineering.py

Phase 1 TDD — feature engineering tests written BEFORE implementation.
All tests use synthetic DataFrames — no real data loaded, runs in milliseconds.

Run: pytest tests/unit/test_feature_engineering.py -v
"""

import copy
import inspect
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.data import preprocess as preprocess_module
from src.data.data_splitter import time_based_split
from src.data.feature_engineering import (
    TARGET_ENCODING_PRIOR_WEIGHT,
    FeatureEngineer,
    _partial_fit_batches,
)
from src.data.preprocess import run_pipeline


# ─── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame that mimics IEEE-CIS transaction structure.
    500 rows, sorted by TransactionDT (temporal), with known fraud rate.
    """
    n = 500
    rng = np.random.default_rng(42)

    # Temporal column: seconds offset (like IEEE-CIS TransactionDT)
    transaction_dt = np.arange(0, n * 3600, 3600, dtype=np.float64)  # 1 hour gaps

    df = pd.DataFrame(
        {
            "TransactionDT": transaction_dt,
            "TransactionAmt": rng.uniform(1.0, 5000.0, n),
            "isFraud": (rng.random(n) < 0.035).astype(int),
            "card1": rng.integers(1000, 9999, n),
            "ProductCD": rng.choice(["W", "H", "C", "S", "R"], n),
            "card4": rng.choice(["visa", "mastercard", "discover", np.nan], n),
            "card6": rng.choice(["debit", "credit", np.nan], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan], n),
            "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", np.nan], n),
            # V-features (abbreviated — real data has V1-V339)
            **{f"V{i}": rng.uniform(-1, 1, n) for i in range(1, 11)},
            # Some with NaN to test imputation
            "M1": rng.choice(["T", "F", np.nan], n),
        }
    )
    return df


# ─── Split tests ──────────────────────────────────────────────────────────────


def test_no_data_leakage_in_split(raw_df: pd.DataFrame) -> None:
    """
    Critical: train max TransactionDT must be strictly less than test min TransactionDT.
    This is the core anti-leakage guarantee for time-based splits.
    """
    X = raw_df.drop(columns=["isFraud"])
    y = raw_df["isFraud"]

    X_train, X_test, y_train, y_test = time_based_split(
        df=raw_df, temporal_col="TransactionDT", ratio=0.80
    )

    train_max_dt = X_train["TransactionDT"].max()
    test_min_dt = X_test["TransactionDT"].min()

    assert train_max_dt < test_min_dt, (
        f"DATA LEAKAGE DETECTED: train max DT ({train_max_dt}) >= "
        f"test min DT ({test_min_dt}). Temporal ordering is broken."
    )


def test_split_sizes_respect_ratio(raw_df: pd.DataFrame) -> None:
    """Train/test row counts must respect the 80/20 ratio (±2 rows for rounding)."""
    X_train, X_test, y_train, y_test = time_based_split(
        df=raw_df, temporal_col="TransactionDT", ratio=0.80
    )

    total = len(raw_df)
    expected_train = int(total * 0.80)

    assert (
        abs(len(X_train) - expected_train) <= 2
    ), f"Train set size {len(X_train)} deviates from expected ~{expected_train}"
    assert (
        len(X_train) + len(X_test) == total
    ), "Train + test rows must sum to total rows (no rows dropped or duplicated)"


# ─── Temporal feature tests ───────────────────────────────────────────────────


def test_temporal_features_created(raw_df: pd.DataFrame) -> None:
    """
    FeatureEngineer.create_temporal_features() must produce all 6 cyclical columns.
    No raw TransactionDT in output (it's dropped later but at minimum transformed).
    """
    fe = FeatureEngineer()
    result = fe.create_temporal_features(raw_df.copy())

    expected_cols = [
        "hour_of_day",
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing expected temporal column: '{col}'"

    # Cyclical features must be in [-1, 1] range (sin/cos bounded)
    for cyc_col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
        assert (
            result[cyc_col].between(-1.0, 1.0).all()
        ), f"Cyclical feature '{cyc_col}' has values outside [-1, 1]"


# ─── Amount feature tests ─────────────────────────────────────────────────────


def test_amount_features_created(raw_df: pd.DataFrame) -> None:
    """
    FeatureEngineer.create_amount_features() must produce amount_log and amount_cents.
    amount_log must be non-negative (log1p of positive amounts).
    """
    fe = FeatureEngineer()
    result = fe.create_amount_features(raw_df.copy())

    assert "amount_log" in result.columns, "Missing 'amount_log' column"
    assert "amount_cents" in result.columns, "Missing 'amount_cents' column"

    # log1p of any positive number is non-negative
    assert (result["amount_log"] >= 0).all(), "amount_log must be >= 0"

    # amount_cents is the fractional part: must be in [0, 1)
    assert (
        result["amount_cents"].between(0.0, 1.0, inclusive="left").all()
    ), "amount_cents must be in [0.0, 1.0)"


# ─── Missing value tests ──────────────────────────────────────────────────────


def test_missing_value_handling(raw_df: pd.DataFrame) -> None:
    """
    After handle_missing_values(), no NaN values must remain in the DataFrame.
    Numerical NaN → -999, Categorical NaN → 'MISSING'.
    """
    fe = FeatureEngineer()

    # Inject explicit NaN into numerical column
    raw_df_with_nan = raw_df.copy()
    raw_df_with_nan.loc[0:10, "TransactionAmt"] = np.nan

    result = fe.handle_missing_values(raw_df_with_nan, fit=True)

    nan_counts = result.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]

    assert (
        len(cols_with_nan) == 0
    ), f"NaN values remain after handle_missing_values():\n{cols_with_nan}"


# ─── Phase A1: Orchestrator-level data-leakage tests ──────────────────────────
#
# THESE TESTS ARE EXPECTED TO FAIL until Phase A2/A3/A4 land.
# They expose the defect documented in docs/IMPLEMENTATION_PLAN.md Phase A:
#   run_pipeline() fits every stateful transformer on the *full* DataFrame
#   and only calls time_based_split_3way() afterwards, so val/test rows
#   contaminate PCA components, frequency maps, label-encoder vocabularies,
#   the card-hash frequency map, and — most critically — the global
#   target-encoding prior (_global_target_mean).
#
# The FeatureEngineer methods themselves are already correct: each honours
# fit=False. The defect is purely in the orchestrator, so every test below
# drives run_pipeline() rather than the individual transform methods.
#
# handle_missing_values is deliberately not covered here: it stores the
# constant -999.0 per numeric column, so it holds no data-dependent state
# that val/test rows could contaminate.
#
# Exit criterion: all tests below FAIL now (correct) and PASS after A2-A4.
# Do NOT mark them xfail/skip — visible failure is the required signal.
#
# Reference: docs/IMPLEMENTATION_PLAN.md  Phase A (A1 RED phase)


# ─── Synthetic raw-DataFrame factory ─────────────────────────────────────────


def _make_pipeline_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """
    Produce a synthetic DataFrame that satisfies DataLoader.validate_schema.

    Columns present:
      TransactionID, TransactionDT, TransactionAmt, isFraud,
      card1..card6, addr1, ProductCD, P_emaildomain, R_emaildomain,
      M1, V1..V10 (with some NaNs injected).

    Rows are already sorted by TransactionDT (ascending) — the pipeline's
    temporal sort is a no-op, so the split boundary is deterministic.
    """
    rng = np.random.default_rng(seed)

    transaction_dt = np.arange(0, n * 3600, 3600, dtype=np.float64)

    # V-features with intentional NaNs (~10 %)
    v_data = {f"V{i}": rng.uniform(-2.0, 2.0, n) for i in range(1, 11)}
    for v_col in list(v_data.keys())[:3]:
        nan_mask = rng.random(n) < 0.10
        v_data[v_col] = np.where(nan_mask, np.nan, v_data[v_col])

    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": transaction_dt,
            "TransactionAmt": rng.uniform(1.0, 5000.0, n),
            "isFraud": (rng.random(n) < 0.035).astype(int),
            "card1": rng.integers(1000, 9999, n),
            "card2": rng.integers(100, 999, n),
            "card3": rng.integers(100, 199, n),
            "card5": rng.integers(100, 230, n),
            "addr1": rng.integers(100, 500, n).astype(float),
            "ProductCD": rng.choice(["W", "H", "C", "S", "R"], n),
            "card4": rng.choice(["visa", "mastercard", "discover", np.nan], n),
            "card6": rng.choice(["debit", "credit", np.nan], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan], n),
            "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", np.nan], n),
            "M1": rng.choice(["T", "F", np.nan], n),
            **v_data,
        }
    )
    return df


def _make_perturbed_twin(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Return a copy of *df* that is bit-identical over the train slice but has
    every held-out (val + test) row replaced with values that would shift the
    fitted transformer state if they were observed at fit time.

    Perturbing val as well as test is deliberate: Phase A's acceptance
    criterion is that *no* transformer observes val or test rows during fit,
    so a fix that split the frame but still fitted on train+val must also
    fail these tests.

    Perturbations applied to the held-out slice only:
      - isFraud labels flipped  → shifts _global_target_mean
      - V* columns scaled ×10   → shifts IncrementalPCA components / mean
      - card1 / addr1 combos replaced with out-of-range integers
                                → shifts _card_hash_freq and target-enc maps
      - ProductCD / card4 / card6 / M1 set to novel unseen category "ZZZ"
                                → extends LabelEncoder.classes_ vocabularies
      - P_emaildomain / R_emaildomain set to "novel-domain.zz"
                                → shifts _freq_encoders maps

    TransactionDT is left untouched so both frames split at the same boundary.
    """
    train_ratio = config["data"].get("train_split_ratio", 0.70)

    n = len(df)
    # Same boundary time_based_split_3way uses for the end of the train slice
    train_end = int(n * train_ratio)
    holdout_n = n - train_end

    perturbed = df.copy()

    # .loc with a RangeIndex is label-based and inclusive of train_end, which
    # is exactly the first held-out row.
    perturbed.loc[train_end:, "isFraud"] = 1 - perturbed.loc[train_end:, "isFraud"]

    v_cols = [c for c in df.columns if c.startswith("V")]
    perturbed.loc[train_end:, v_cols] = perturbed.loc[train_end:, v_cols] * 10.0

    perturbed.loc[train_end:, "card1"] = np.arange(99000, 99000 + holdout_n)
    perturbed.loc[train_end:, "addr1"] = np.arange(9000, 9000 + holdout_n, dtype=float)

    for col in ["ProductCD", "card4", "card6", "M1"]:
        if col in perturbed.columns:
            perturbed.loc[train_end:, col] = "ZZZ"

    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in perturbed.columns:
            perturbed.loc[train_end:, col] = "novel-domain.zz"

    return perturbed


def _is_fit_call(bound_method, args: tuple, kwargs: dict) -> bool:
    """Whether a FeatureEngineer method call passes fit=True, by keyword or position."""
    if "fit" in kwargs:
        return kwargs["fit"] is True

    try:
        params = list(inspect.signature(bound_method).parameters)
        # The method is already bound, so `self` is absent from the signature.
        fit_idx = params.index("fit")
    except ValueError:
        return False  # method has no `fit` parameter

    return fit_idx < len(args) and args[fit_idx] is True


class _RecordingFE:
    """
    Stand-in for FeatureEngineer that delegates to a real instance while
    recording both the instance and the order of fit=True calls.

    run_pipeline neither returns nor exposes its FeatureEngineer, and
    target-encoding state is not written by save_transformers, so capturing
    the live instance is the only way to inspect the fitted state.
    """

    instances: list = []
    call_order: list = []  # ("fit", method_name) and ("split",) markers

    @classmethod
    def reset(cls) -> None:
        cls.instances.clear()
        cls.call_order.clear()

    def __init__(self) -> None:
        self._real = FeatureEngineer()
        _RecordingFE.instances.append(self._real)

    def __getattr__(self, name: str):
        real_attr = getattr(self._real, name)
        if not callable(real_attr):
            return real_attr

        def _wrapper(*args, **kwargs):
            if _is_fit_call(real_attr, args, kwargs):
                _RecordingFE.call_order.append(("fit", name))
            return real_attr(*args, **kwargs)

        return _wrapper


def _run_pipeline_capturing_state(
    df: pd.DataFrame,
    cfg: dict,
    tmp_path,
) -> FeatureEngineer:
    """
    Run run_pipeline() against the synthetic *df* and return the fitted
    FeatureEngineer so tests can inspect its state.

    The raw loader is replaced with a stub returning *df*, FeatureEngineer is
    replaced with _RecordingFE, and time_based_split_3way is wrapped so the
    ordering of split vs fit calls is recorded in _RecordingFE.call_order.
    processed_dir AND serving.transformer_path are redirected to tmp_path so
    no repo files are written — run_pipeline's Step 8 dual-writes transformers
    to serving.transformer_path (default models/transformers/), and without
    this redirect a synthetic-data test run would overwrite the real serving
    artifacts.
    """
    _RecordingFE.reset()

    # Deep-copy so the session-scoped config fixture is never mutated
    run_cfg = copy.deepcopy(cfg)
    run_cfg["data"]["processed_dir"] = str(tmp_path)
    run_cfg.setdefault("serving", {})["transformer_path"] = str(
        tmp_path / "serving_transformers"
    )

    loader = mock.MagicMock()
    loader.load_raw.return_value = df.copy()
    loader.validate_schema.return_value = None
    loader.sort_temporal.side_effect = lambda d: d.sort_values(
        "TransactionDT", ascending=True
    ).reset_index(drop=True)

    real_split = preprocess_module.time_based_split_3way

    def _spy_split(*args, **kwargs):
        _RecordingFE.call_order.append(("split",))
        return real_split(*args, **kwargs)

    with (
        mock.patch(
            "src.data.preprocess.DataLoader", mock.MagicMock(return_value=loader)
        ),
        mock.patch("src.data.preprocess.FeatureEngineer", _RecordingFE),
        mock.patch("src.data.preprocess.time_based_split_3way", _spy_split),
    ):
        run_pipeline(run_cfg)

    assert _RecordingFE.instances, "run_pipeline created no FeatureEngineer instance"
    return _RecordingFE.instances[0]


# ─── Test 1: _global_target_mean (label leakage — most severe) ───────────────


@pytest.mark.unit
def test_pipeline_target_mean_not_contaminated_by_holdout_labels(
    config: dict, tmp_path
) -> None:
    """
    EXPECTED TO FAIL until Phase A2 lands.

    _global_target_mean is set by create_target_encoding() from
    df["isFraud"].mean() over the *full* DataFrame (L483 feature_engineering.py,
    called at L117 preprocess.py — before the split).

    Perturbation-invariance property: if run_pipeline splits before fitting,
    flipping the held-out labels must NOT change _global_target_mean,
    because only train rows are observed.  Under the current (buggy) code
    the full-frame mean is used, so the two values will differ.
    """
    df_orig = _make_pipeline_df(n=500, seed=1)
    df_pert = _make_perturbed_twin(df_orig, config)

    fe_orig = _run_pipeline_capturing_state(df_orig, config, tmp_path / "orig")
    fe_pert = _run_pipeline_capturing_state(df_pert, config, tmp_path / "pert")

    assert fe_orig._global_target_mean == fe_pert._global_target_mean, (
        f"TARGET-ENCODING LABEL LEAKAGE DETECTED: "
        f"_global_target_mean differs between original ({fe_orig._global_target_mean:.6f}) "
        f"and perturbed ({fe_pert._global_target_mean:.6f}) frames. "
        f"The full-dataset isFraud mean is being used instead of train-only mean. "
        f"Fix: call time_based_split_3way BEFORE create_target_encoding."
    )


# ─── Test 2: _pca components (IncrementalPCA leakage) ────────────────────────


@pytest.mark.unit
def test_pipeline_pca_not_contaminated_by_holdout_rows(config: dict, tmp_path) -> None:
    """
    EXPECTED TO FAIL until Phase A2 lands.

    reduce_v_features(fit=True) is called at L75 preprocess.py — before the
    split — so IncrementalPCA observes V* values from val/test rows.

    Perturbation-invariance: scaling held-out V* by 10× must not change the
    PCA components / mean, because a correctly placed fit sees only train rows.
    Under the current bug the PCA is refitted on all rows.
    """
    df_orig = _make_pipeline_df(n=500, seed=2)
    df_pert = _make_perturbed_twin(df_orig, config)

    fe_orig = _run_pipeline_capturing_state(df_orig, config, tmp_path / "orig")
    fe_pert = _run_pipeline_capturing_state(df_pert, config, tmp_path / "pert")

    assert (
        fe_orig._pca is not None
    ), "PCA was not fitted — check V-feature columns in synthetic df"
    assert fe_pert._pca is not None, "PCA was not fitted on perturbed frame"

    assert (
        fe_orig._pca.components_.shape == fe_pert._pca.components_.shape
    ), "PCA component shapes differ — unexpected structural change."

    np.testing.assert_allclose(
        fe_orig._pca.components_,
        fe_pert._pca.components_,
        rtol=1e-4,
        atol=1e-5,
        err_msg=(
            "PCA LEAKAGE DETECTED: IncrementalPCA components differ between the "
            "original and perturbed (held-out V* scaled x10) frames. "
            "Fix: split BEFORE calling reduce_v_features(fit=True)."
        ),
    )
    np.testing.assert_allclose(
        fe_orig._pca.mean_,
        fe_pert._pca.mean_,
        rtol=1e-4,
        atol=1e-5,
        err_msg="PCA mean_ differs — held-out rows are influencing the fit.",
    )


# ─── Test 3: _freq_encoders (frequency maps leakage) ─────────────────────────


@pytest.mark.unit
def test_pipeline_freq_encoders_not_contaminated_by_holdout_rows(
    config: dict, tmp_path
) -> None:
    """
    EXPECTED TO FAIL until Phase A3 lands.

    encode_categoricals(fit=True) at L126 preprocess.py computes
    value_counts(normalize=True) over the full frame, so frequency maps for
    P_emaildomain / R_emaildomain include held-out frequencies.

    Perturbation-invariance: replacing held-out email domains with
    "novel-domain.zz" must not change _freq_encoders, because only train rows
    should be counted.  Under the current bug they will differ.
    """
    df_orig = _make_pipeline_df(n=500, seed=3)
    df_pert = _make_perturbed_twin(df_orig, config)

    fe_orig = _run_pipeline_capturing_state(df_orig, config, tmp_path / "orig")
    fe_pert = _run_pipeline_capturing_state(df_pert, config, tmp_path / "pert")

    assert fe_orig._freq_encoders.keys() == fe_pert._freq_encoders.keys(), (
        f"_freq_encoders key sets differ: "
        f"orig={set(fe_orig._freq_encoders)} pert={set(fe_pert._freq_encoders)}"
    )

    for col, orig_map in fe_orig._freq_encoders.items():
        pert_map = fe_pert._freq_encoders[col]
        assert orig_map == pert_map, (
            f"FREQ-ENCODER LEAKAGE DETECTED for column '{col}': "
            f"frequency map differs between original and perturbed (novel test-domain) frames. "
            f"Fix: call time_based_split_3way BEFORE encode_categoricals(fit=True)."
        )


# ─── Test 4: _label_encoders vocabularies leakage ────────────────────────────


@pytest.mark.unit
def test_pipeline_label_encoders_not_contaminated_by_holdout_rows(
    config: dict, tmp_path
) -> None:
    """
    EXPECTED TO FAIL until Phase A3 lands.

    encode_categoricals(fit=True) at L126 calls LabelEncoder.fit() on the full
    frame, so novel held-out categories (e.g. 'ZZZ') enter classes_.

    Perturbation-invariance: injecting 'ZZZ' into held-out categorical
    columns must not add 'ZZZ' to any LabelEncoder.classes_, because only
    train rows should be fitted over.  Under the bug, 'ZZZ' will appear.
    """
    df_orig = _make_pipeline_df(n=500, seed=4)
    df_pert = _make_perturbed_twin(df_orig, config)

    fe_orig = _run_pipeline_capturing_state(df_orig, config, tmp_path / "orig")
    fe_pert = _run_pipeline_capturing_state(df_pert, config, tmp_path / "pert")

    assert fe_orig._label_encoders.keys() == fe_pert._label_encoders.keys(), (
        f"_label_encoders key sets differ: "
        f"orig={set(fe_orig._label_encoders)} pert={set(fe_pert._label_encoders)}"
    )

    for col, orig_le in fe_orig._label_encoders.items():
        pert_le = fe_pert._label_encoders[col]
        orig_vocab = set(orig_le.classes_)
        pert_vocab = set(pert_le.classes_)
        assert orig_vocab == pert_vocab, (
            f"LABEL-ENCODER LEAKAGE DETECTED for column '{col}': "
            f"vocabularies differ. "
            f"Extra categories in perturbed encoder: {pert_vocab - orig_vocab}. "
            f"Test categories ('ZZZ') leaked into LabelEncoder.classes_. "
            f"Fix: call time_based_split_3way BEFORE encode_categoricals(fit=True)."
        )


# ─── Test 5: _card_hash_freq (card-hash frequency leakage) ───────────────────


@pytest.mark.unit
def test_pipeline_card_hash_freq_not_contaminated_by_holdout_rows(
    config: dict, tmp_path
) -> None:
    """
    EXPECTED TO FAIL until Phase A3 lands.

    create_card_hash_features(fit=True) at L108 computes value_counts over
    the full DataFrame, so held-out card/addr combos enter _card_hash_freq.

    Perturbation-invariance: replacing held-out card1 / addr1 with
    out-of-range integers (unseen combos) must not change _card_hash_freq,
    because only train rows should be counted.  Under the bug it will differ.
    """
    df_orig = _make_pipeline_df(n=500, seed=5)
    df_pert = _make_perturbed_twin(df_orig, config)

    fe_orig = _run_pipeline_capturing_state(df_orig, config, tmp_path / "orig")
    fe_pert = _run_pipeline_capturing_state(df_pert, config, tmp_path / "pert")

    assert fe_orig._card_hash_freq == fe_pert._card_hash_freq, (
        "CARD-HASH LEAKAGE DETECTED: _card_hash_freq differs between original "
        "and perturbed (novel test card1/addr1 combos) frames. "
        f"Keys only in perturbed: "
        f"{set(fe_pert._card_hash_freq) - set(fe_orig._card_hash_freq)}. "
        "Fix: call time_based_split_3way BEFORE create_card_hash_features(fit=True)."
    )


# ─── Test 6: structural — split must precede every fit=True call ─────────────


@pytest.mark.unit
def test_pipeline_split_precedes_all_fit_calls(config: dict, tmp_path) -> None:
    """
    EXPECTED TO FAIL until Phase A2 lands.

    The structural property underlying every test above: time_based_split_3way
    must be called BEFORE any FeatureEngineer method is invoked with fit=True.
    run_pipeline currently does the opposite — every fit=True call happens
    first, and the split follows at L145.

    This is the assertion that most directly pins the Phase A2 refactor.
    """
    df = _make_pipeline_df(n=500, seed=6)

    _run_pipeline_capturing_state(df, config, tmp_path)
    call_order = list(_RecordingFE.call_order)

    split_positions = [i for i, entry in enumerate(call_order) if entry[0] == "split"]
    assert split_positions, (
        "time_based_split_3way was never called by run_pipeline — "
        "the ordering property cannot be evaluated."
    )

    fit_before_split = [
        entry[1] for entry in call_order[: split_positions[0]] if entry[0] == "fit"
    ]

    assert not fit_before_split, (
        "ORCHESTRATOR LEAKAGE DETECTED: these transformers were fitted BEFORE "
        f"time_based_split_3way was called: {fit_before_split}. "
        "run_pipeline must split first and fit on the train slice only, "
        "transforming val/test with fit=False. This is the Phase A2 refactor."
    )


# ─── Phase A2-A4: fit/transform separation invariants ────────────────────────
#
# Splitting before fitting only stays correct if the per-split transforms
# reproduce what a single pass over the full ordered frame would have produced.
# These two tests pin the equivalences that the A2 orchestrator relies on.


@pytest.fixture()
def target_encoding_df() -> pd.DataFrame:
    """Temporally ordered frame with repeated entities across the split boundary."""
    n = 600
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "TransactionDT": np.arange(n, dtype=float),
            "isFraud": (rng.random(n) < 0.2).astype(int),
            "card1": rng.integers(0, 25, n),
            "addr1": rng.integers(0, 10, n).astype(float),
        }
    )


@pytest.mark.unit
def test_target_encoding_split_wise_matches_full_frame(
    target_encoding_df: pd.DataFrame,
) -> None:
    """
    Running create_target_encoding per split (train fit → val update → test)
    must yield exactly the values a single full-frame pass gives, so that
    fixing the leakage does not silently reset each entity's history at the
    split boundary.
    """
    df = target_encoding_df
    train_end, val_end = 420, 480

    # Reference: one pass over everything, with the prior pinned to train-only
    reference_fe = FeatureEngineer()
    reference_fe._global_target_mean = float(df.iloc[:train_end]["isFraud"].mean())
    reference = reference_fe.create_target_encoding(df, fit=False)

    fe = FeatureEngineer()
    train = fe.create_target_encoding(
        df.iloc[:train_end].reset_index(drop=True), fit=True
    )
    val = fe.create_target_encoding(
        df.iloc[train_end:val_end].reset_index(drop=True), fit=False, update_state=True
    )
    test = fe.create_target_encoding(
        df.iloc[val_end:].reset_index(drop=True), fit=False, update_state=False
    )
    stitched = pd.concat([train, val, test], ignore_index=True)

    assert fe._global_target_mean == pytest.approx(
        df.iloc[:train_end]["isFraud"].mean()
    ), "Target-encoding prior must come from train labels only."

    for col in ["card1_target_enc", "addr1_target_enc"]:
        np.testing.assert_allclose(
            stitched[col].to_numpy(),
            reference[col].to_numpy(),
            rtol=1e-9,
            atol=1e-12,
            err_msg=(
                f"'{col}' differs between the split-wise and full-frame passes: "
                "entity history is not carrying across the split boundary."
            ),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "n, batch_size, min_batch",
    [
        (0, 10, 3),
        (5, 10, 3),
        (10, 10, 3),
        (11, 10, 3),
        (13, 10, 3),
        (25, 10, 3),
        (35, 10, 30),  # min_batch > batch_size: every batch, not just the
        (60, 10, 30),  # last, must grow past batch_size to stay >= min_batch.
    ],
)
def test_partial_fit_batches_never_yields_a_short_batch(
    n: int, batch_size: int, min_batch: int
) -> None:
    """
    Every batch _partial_fit_batches yields must have >= min_batch elements
    (IncrementalPCA.partial_fit raises otherwise), the batches must partition
    the input exactly once with no gaps, overlaps, or reordering, and n=0
    must raise rather than silently yield an empty batch.
    """
    positions = np.arange(n)

    if n == 0:
        with pytest.raises(ValueError, match="empty"):
            list(_partial_fit_batches(positions, batch_size, min_batch))
        return

    batches = list(_partial_fit_batches(positions, batch_size, min_batch))

    assert all(len(b) >= min_batch for b in batches), (
        f"A batch smaller than min_batch={min_batch} was yielded: "
        f"sizes={[len(b) for b in batches]}. IncrementalPCA.partial_fit would raise."
    )
    reassembled = np.concatenate(batches)
    np.testing.assert_array_equal(
        reassembled,
        positions,
        err_msg="Batches must partition positions exactly once, in order.",
    )


@pytest.mark.unit
def test_pca_fit_rows_matches_fitting_on_train_slice_alone() -> None:
    """
    reduce_v_features(fit_rows=mask) must produce the same components as
    fitting on the train slice by itself, while still transforming every row.
    This is what lets the reduction run before the sort (memory optimisation)
    without letting held-out rows into the fit.
    """
    n, train_end = 600, 420
    rng = np.random.default_rng(11)
    v_df = pd.DataFrame({f"V{i}": rng.uniform(-1, 1, n) for i in range(1, 12)})
    # Held-out rows are scaled hard: if they reached the fit, components move.
    v_df.loc[train_end:, :] = v_df.loc[train_end:, :] * 25.0

    train_mask = np.zeros(n, dtype=bool)
    train_mask[:train_end] = True

    masked_fe = FeatureEngineer()
    reduced = masked_fe.reduce_v_features(
        v_df, fit=True, n_components=5, fit_rows=train_mask
    )

    slice_fe = FeatureEngineer()
    slice_fe.reduce_v_features(
        v_df.iloc[:train_end].reset_index(drop=True), fit=True, n_components=5
    )

    assert masked_fe._pca.n_samples_seen_ == train_end, (
        f"PCA observed {masked_fe._pca.n_samples_seen_} rows, expected {train_end} "
        "train rows — fit_rows is not restricting the fit."
    )
    np.testing.assert_allclose(
        masked_fe._pca.components_,
        slice_fe._pca.components_,
        rtol=1e-6,
        atol=1e-8,
        err_msg="fit_rows components differ from fitting on the train slice alone.",
    )
    assert len(reduced) == n, "Every row must still be transformed, fit or not."


# ─── Phase A6: current-row exclusion regression tests ────────────────────────
#
# create_card_aggregates and create_target_encoding are the two computations
# that summarise an entity's own history into a feature on one of that
# entity's rows. Both are only leakage-free if row i's contribution is
# excluded from row i's own feature value — a `- df[col]` term and a shift(1)
# that a future refactor could drop without breaking any other test.
#
# Each computation is pinned two ways:
#   1. Against an independent naive oracle that literally loops over strictly
#      prior rows, so the vectorised expression cannot silently drift.
#   2. By perturbation invariance — mutating row i's own amount/label must
#      leave row i's history features untouched while still moving later rows
#      of the same entity, which rules out an oracle that is trivially
#      satisfied by an all-zero output.
#
# Reference: docs/IMPLEMENTATION_PLAN.md  Phase A (A6)

# Columns produced by create_card_aggregates that describe *prior* rows only.
# amount_vs_mean_ratio and amount_zscore_per_card are deliberately excluded:
# they compare the current amount against that history, so they must depend on
# the current row.
CARD_HISTORY_COLS = [
    "tx_count_per_card",
    "tx_sum_per_card",
    "mean_amount_per_card",
    "max_amount_per_card",
    "std_amount_per_card",
]


# Sentinel standing in for a missing entity key inside the oracles' dicts.
# Plain NaN cannot be used: dict lookup falls back to `==` after the hash hit,
# and nan != nan, so every NaN row would land in its own bucket and the oracle
# would disagree with a (correct) dropna=False groupby.
# object() rather than a string: identity-hashed, so no real data value can
# collide with it and silently merge a genuine group into the missing-key one.
_MISSING_KEY = object()


def _group_key(value: Any) -> Any:
    """Hashable, NaN-stable grouping key matching pandas' dropna=False groupby."""
    return _MISSING_KEY if pd.isna(value) else value


def _naive_card_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oracle for create_card_aggregates' history columns.

    Walks the frame in order and, for each row, aggregates the amounts of the
    rows *before* it that share its card1 — the definition the vectorised
    cumsum/shift expressions are supposed to implement. Empty history yields
    the zeros create_card_aggregates falls back to.
    """
    seen: dict[Any, list[float]] = {}
    rows = []

    for card, amount in zip(df["card1"], df["TransactionAmt"]):
        prior = seen.setdefault(_group_key(card), [])
        count = len(prior)
        total = float(np.sum(prior)) if prior else 0.0
        rows.append(
            {
                "tx_count_per_card": count,
                "tx_sum_per_card": total,
                "mean_amount_per_card": total / count if count else 0.0,
                "max_amount_per_card": max(prior) if prior else 0.0,
                # ddof=1 — the sample std the (sum_sq - sum^2/n)/(n-1) form computes
                "std_amount_per_card": (
                    float(np.std(prior, ddof=1)) if count > 1 else 0.0
                ),
            }
        )
        prior.append(float(amount))

    return pd.DataFrame(rows)


def _naive_target_encoding(
    df: pd.DataFrame, col: str, prior: float, weight: float
) -> np.ndarray:
    """
    Oracle for create_target_encoding: smoothed mean of the labels on strictly
    prior rows sharing the same entity value, falling back to the global prior
    before any history exists.
    """
    seen: dict[Any, list[int]] = {}
    encoded = []

    for key, label in zip(df[col], df["isFraud"]):
        history = seen.setdefault(_group_key(key), [])
        count = len(history)
        if count == 0:
            encoded.append(prior)
        else:
            encoded.append((sum(history) + weight * prior) / (count + weight))
        history.append(int(label))

    return np.asarray(encoded, dtype=float)


@pytest.fixture()
def card_history_df() -> pd.DataFrame:
    """
    Temporally ordered frame where every card recurs many times, so the
    history columns are exercised well past their first-occurrence branch.
    """
    n = 300
    rng = np.random.default_rng(23)
    return pd.DataFrame(
        {
            "TransactionDT": np.arange(n, dtype=float),
            "TransactionAmt": rng.uniform(1.0, 5000.0, n).round(2),
            "isFraud": (rng.random(n) < 0.2).astype(int),
            # 12 cards over 300 rows → ~25 transactions each
            "card1": rng.integers(0, 12, n),
            "addr1": rng.integers(0, 6, n).astype(float),
        }
    )


@pytest.mark.unit
def test_card_aggregates_exclude_the_current_row(
    card_history_df: pd.DataFrame,
) -> None:
    """
    Every card-history column must equal the aggregate over strictly prior
    transactions of the same card. If the `- df["TransactionAmt"]` term or the
    shift(1) were dropped, each row would include its own amount and every
    column below would disagree with the oracle.
    """
    result = FeatureEngineer().create_card_aggregates(card_history_df.copy())
    expected = _naive_card_history(card_history_df)

    for col in CARD_HISTORY_COLS:
        np.testing.assert_allclose(
            result[col].to_numpy(dtype=float),
            expected[col].to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-6,
            err_msg=(
                f"CURRENT-ROW LEAKAGE in '{col}': the vectorised aggregate does "
                "not match a per-row aggregate over strictly prior transactions "
                "of the same card. Row i is seeing its own TransactionAmt."
            ),
        )


@pytest.mark.unit
def test_card_aggregates_are_zero_on_a_cards_first_transaction(
    card_history_df: pd.DataFrame,
) -> None:
    """
    A card's first transaction has no history, so every history column must be
    exactly zero there — the sharpest single symptom of a self-inclusive sum.
    """
    result = FeatureEngineer().create_card_aggregates(card_history_df.copy())
    first_rows = result.groupby("card1", sort=False).head(1)

    assert (
        len(first_rows) == card_history_df["card1"].nunique()
    ), "Expected one first-transaction row per card."

    for col in CARD_HISTORY_COLS:
        offenders = first_rows.loc[first_rows[col] != 0.0, ["card1", col]]
        assert offenders.empty, (
            f"CURRENT-ROW LEAKAGE in '{col}': a card's first transaction has "
            f"non-zero history:\n{offenders}"
        )


@pytest.mark.unit
def test_card_aggregates_ignore_a_change_to_the_rows_own_amount(
    card_history_df: pd.DataFrame,
) -> None:
    """
    Perturbation invariance: multiplying one row's TransactionAmt by 1000 must
    leave that row's own history columns identical, while moving the later
    rows of the same card. The second half matters as much as the first — it
    proves the invariance is not the trivial one an all-zero column would give.
    """
    target_idx = 200  # deep into the frame, so the card has real history
    perturbed = card_history_df.copy()
    perturbed.loc[target_idx, "TransactionAmt"] *= 1000.0

    fe = FeatureEngineer()
    baseline = fe.create_card_aggregates(card_history_df.copy())
    after = fe.create_card_aggregates(perturbed)

    same_card = card_history_df.loc[target_idx, "card1"]
    later_same_card = card_history_df.index[
        (card_history_df["card1"] == same_card) & (card_history_df.index > target_idx)
    ]
    assert (
        len(later_same_card) > 0
    ), "Fixture must place at least one later transaction on the perturbed card."

    for col in CARD_HISTORY_COLS:
        assert baseline.loc[target_idx, col] == pytest.approx(
            after.loc[target_idx, col]
        ), (
            f"CURRENT-ROW LEAKAGE in '{col}': scaling row {target_idx}'s own "
            "TransactionAmt changed that same row's history feature."
        )

    downstream_changed = any(
        not np.allclose(
            baseline.loc[later_same_card, col].to_numpy(),
            after.loc[later_same_card, col].to_numpy(),
        )
        for col in CARD_HISTORY_COLS
    )
    assert downstream_changed, (
        "Later transactions on the same card are unchanged by a 1000x amount "
        "perturbation — the history columns are not accumulating at all, so "
        "the exclusion assertions above are vacuous."
    )


@pytest.mark.unit
def test_card_aggregates_order_history_by_time_not_input_order(
    card_history_df: pd.DataFrame,
) -> None:
    """
    "Prior" must mean earlier in time, not earlier in the caller's row order.
    create_card_aggregates re-sorts on TransactionDT internally, so a shuffled
    frame must yield the same per-transaction history as a pre-sorted one.

    This also pins the fact that the method returns rows in TransactionDT
    order with a reset index rather than the caller's order — a caller holding
    a separately-ordered label vector would otherwise misalign it silently.
    """
    shuffled = card_history_df.sample(frac=1.0, random_state=99).reset_index(drop=True)

    fe = FeatureEngineer()
    from_sorted = fe.create_card_aggregates(card_history_df.copy())
    from_shuffled = fe.create_card_aggregates(shuffled)

    np.testing.assert_array_equal(
        from_shuffled["TransactionDT"].to_numpy(),
        card_history_df["TransactionDT"].to_numpy(),
        err_msg="Output rows are not returned in TransactionDT order.",
    )

    for col in CARD_HISTORY_COLS:
        np.testing.assert_allclose(
            from_shuffled[col].to_numpy(dtype=float),
            from_sorted[col].to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-6,
            err_msg=(
                f"'{col}' depends on the caller's row order: history is being "
                "accumulated in input order rather than temporal order."
            ),
        )


@pytest.mark.unit
@pytest.mark.parametrize("col", ["card1", "addr1"])
def test_target_encoding_excludes_the_current_rows_label(
    card_history_df: pd.DataFrame, col: str
) -> None:
    """
    Each `<col>_target_enc` value must be the smoothed mean of the labels on
    strictly prior rows for that entity. Dropping the `- df[target_col]` term
    would encode the row's own isFraud into its own feature — direct label
    leakage into training.
    """
    fe = FeatureEngineer()
    result = fe.create_target_encoding(card_history_df.copy(), fit=True)

    expected = _naive_target_encoding(
        card_history_df,
        col=col,
        prior=fe._global_target_mean,
        weight=TARGET_ENCODING_PRIOR_WEIGHT,
    )

    np.testing.assert_allclose(
        result[f"{col}_target_enc"].to_numpy(dtype=float),
        expected,
        rtol=1e-9,
        atol=1e-12,
        err_msg=(
            f"LABEL LEAKAGE in '{col}_target_enc': the encoding does not match a "
            "per-row smoothed mean over strictly prior labels. Row i is seeing "
            "its own isFraud value."
        ),
    )


def _naive_target_encoding_lagged(
    df: pd.DataFrame,
    col: str,
    prior: float,
    weight: float,
    time_col: str,
    lag_seconds: float,
) -> np.ndarray:
    """
    Oracle for the lag-aware branch of create_target_encoding (finding F6):
    smoothed mean of labels on rows sharing the same entity whose OWN
    `time_col` is at least `lag_seconds` earlier than the row being encoded
    — not merely "earlier in the frame", which is all `_naive_target_encoding`
    above checks.
    """
    seen: dict[Any, list[tuple[float, int]]] = {}
    encoded = []

    for key, label, t in zip(df[col], df["isFraud"], df[time_col]):
        history = seen.setdefault(_group_key(key), [])
        eligible = [lbl for (ts, lbl) in history if ts <= t - lag_seconds]
        count = len(eligible)
        if count == 0:
            encoded.append(prior)
        else:
            encoded.append((sum(eligible) + weight * prior) / (count + weight))
        history.append((float(t), int(label)))

    return np.asarray(encoded, dtype=float)


@pytest.mark.unit
@pytest.mark.parametrize("col", ["card1", "addr1"])
def test_target_encoding_with_label_lag_excludes_recent_same_window_history(
    card_history_df: pd.DataFrame, col: str
) -> None:
    """
    Finding F6 (2026-08-19 metrics audit): with label_lag_seconds > 0, a
    row's encoding must only reflect entity history old enough to plausibly
    be labeled by the time the row is scored — not merely "earlier in this
    frame", which is all the unlagged path guarantees. Uses a lag of 50
    (card_history_df's TransactionDT spans 0..299), large enough that many
    rows lose access to their most recent same-entity transactions.
    """
    lag_seconds = 50.0
    fe = FeatureEngineer()
    result = fe.create_target_encoding(
        card_history_df.copy(),
        fit=True,
        time_col="TransactionDT",
        label_lag_seconds=lag_seconds,
    )
    expected = _naive_target_encoding_lagged(
        card_history_df,
        col=col,
        prior=fe._global_target_mean,
        weight=TARGET_ENCODING_PRIOR_WEIGHT,
        time_col="TransactionDT",
        lag_seconds=lag_seconds,
    )

    np.testing.assert_allclose(
        result[f"{col}_target_enc"].to_numpy(dtype=float),
        expected,
        rtol=1e-9,
        atol=1e-12,
        err_msg=(
            f"LATENCY LEAKAGE in '{col}_target_enc': with label_lag_seconds="
            f"{lag_seconds}, the encoding does not match a per-row smoothed "
            "mean over entity history at least that old — recent same-window "
            "labels are leaking in as if they were instantly known."
        ),
    )


@pytest.mark.unit
def test_target_encoding_lag_zero_matches_unlagged_path(
    card_history_df: pd.DataFrame,
) -> None:
    """
    label_lag_seconds=0.0 (the FeaturesConfig default) must reproduce the
    original same-window encoding exactly — every existing caller that
    omits time_col/label_lag_seconds must see no behavior change from this
    finding's fix.
    """
    without_lag_args = FeatureEngineer().create_target_encoding(
        card_history_df.copy(), fit=True
    )
    with_lag_zero = FeatureEngineer().create_target_encoding(
        card_history_df.copy(),
        fit=True,
        time_col="TransactionDT",
        label_lag_seconds=0.0,
    )
    pd.testing.assert_frame_equal(without_lag_args, with_lag_zero)


@pytest.mark.unit
def test_target_encoding_positive_lag_without_time_col_raises() -> None:
    """label_lag_seconds > 0 with no time_col is a programming error, not a
    silently-ignored option — must raise, not fall back to the unlagged path."""
    df = pd.DataFrame({"card1": [1, 1, 2], "isFraud": [0, 1, 0]})
    with pytest.raises(ValueError, match="time_col"):
        FeatureEngineer().create_target_encoding(df, fit=True, label_lag_seconds=30.0)


@pytest.mark.unit
def test_target_encoding_ignores_a_flip_of_the_rows_own_label(
    card_history_df: pd.DataFrame,
) -> None:
    """
    Perturbation invariance for the label path: flipping one row's isFraud
    must not change that row's own encoding, but must change the encoding of
    the later rows sharing its card — otherwise history is not accumulating
    and the invariance is vacuous.

    The prior is pinned with fit=False so the flip cannot reach the result
    through _global_target_mean instead of through the per-entity history.
    """
    target_idx = 200
    perturbed = card_history_df.copy()
    perturbed.loc[target_idx, "isFraud"] = 1 - perturbed.loc[target_idx, "isFraud"]

    fixed_prior = float(card_history_df["isFraud"].mean())

    def _encode(frame: pd.DataFrame) -> pd.DataFrame:
        fe = FeatureEngineer()
        fe._global_target_mean = fixed_prior
        return fe.create_target_encoding(frame, fit=False)

    baseline = _encode(card_history_df.copy())
    after = _encode(perturbed)

    same_card = card_history_df.loc[target_idx, "card1"]
    later_same_card = card_history_df.index[
        (card_history_df["card1"] == same_card) & (card_history_df.index > target_idx)
    ]
    assert (
        len(later_same_card) > 0
    ), "Fixture must place at least one later transaction on the perturbed card."

    assert baseline.loc[target_idx, "card1_target_enc"] == pytest.approx(
        after.loc[target_idx, "card1_target_enc"]
    ), (
        f"LABEL LEAKAGE: flipping row {target_idx}'s own isFraud changed that "
        "same row's card1_target_enc."
    )

    assert not np.allclose(
        baseline.loc[later_same_card, "card1_target_enc"].to_numpy(),
        after.loc[later_same_card, "card1_target_enc"].to_numpy(),
    ), (
        "Later transactions on the same card are unaffected by a label flip — "
        "the encoding is not accumulating history, so the exclusion assertion "
        "above is vacuous."
    )


@pytest.mark.unit
def test_target_encoding_falls_back_to_the_prior_on_first_sighting() -> None:
    """
    An entity's first row has no prior labels, so its encoding must be exactly
    the global prior — including when that first label is a fraud, which is the
    case a self-inclusive sum would visibly distort.
    """
    df = pd.DataFrame(
        {
            "TransactionDT": np.arange(4, dtype=float),
            # Each card appears for the first time on a fraudulent transaction
            "card1": [1, 2, 1, 2],
            "isFraud": [1, 1, 0, 0],
        }
    )

    fe = FeatureEngineer()
    result = fe.create_target_encoding(df, fit=True)
    prior = fe._global_target_mean

    first_sightings = result.iloc[:2]["card1_target_enc"].to_numpy()
    np.testing.assert_allclose(
        first_sightings,
        np.full(2, prior),
        rtol=1e-12,
        err_msg=(
            "LABEL LEAKAGE: an entity's first sighting is not encoded as the "
            f"global prior ({prior:.6f}); its own fraud label is bleeding in."
        ),
    )


@pytest.mark.unit
def test_target_encoding_treats_a_missing_entity_key_as_its_own_group() -> None:
    """
    A missing entity key must be encoded as its own expanding group, with the
    same current-row exclusion as any other key.

    This is not a hypothetical: `_apply_stateful_transforms` calls
    create_target_encoding BEFORE handle_missing_values (preprocess.py), so
    entity columns still carry raw NaNs here, and several of them
    (addr1, card2, R_emaildomain) are missing on a large fraction of IEEE-CIS
    rows. A groupby that drops NaN keys yields NaN cumcount, which makes the
    `cum_count == 0` prior fallback unreachable and emits a NaN feature —
    silently, because the downstream tree models accept NaN.
    """
    df = pd.DataFrame(
        {
            "TransactionDT": np.arange(6, dtype=float),
            "card1": [1.0, np.nan, 1.0, np.nan, np.nan, 1.0],
            "isFraud": [1, 1, 0, 1, 0, 0],
        }
    )

    fe = FeatureEngineer()
    result = fe.create_target_encoding(df.copy(), fit=True)
    encoded = result["card1_target_enc"]

    assert not encoded.isna().any(), (
        "Rows with a missing entity key produced NaN target encodings "
        f"({int(encoded.isna().sum())} of {len(encoded)}). The entity groupby is "
        "dropping NaN keys, so those rows get neither a group history nor the "
        "global-prior fallback."
    )

    # The NaN key is one group, so the oracle applies to it unchanged.
    expected = _naive_target_encoding(
        df,
        col="card1",
        prior=fe._global_target_mean,
        weight=TARGET_ENCODING_PRIOR_WEIGHT,
    )
    np.testing.assert_allclose(
        encoded.to_numpy(dtype=float),
        expected,
        rtol=1e-9,
        atol=1e-12,
        err_msg=(
            "Missing-key rows are not encoded as a single expanding group with "
            "the current row excluded."
        ),
    )


@pytest.mark.unit
def test_missing_key_history_carries_across_splits() -> None:
    """
    The carried per-entity state must round-trip the missing-key group too:
    `_accumulate_entity_totals` records it with dropna=False, so the in-frame
    groupby has to record it the same way or train's missing-key history is
    written but never read back on val/test.
    """
    train = pd.DataFrame({"card1": [np.nan, np.nan, np.nan], "isFraud": [1, 1, 1]})
    val = pd.DataFrame({"card1": [np.nan], "isFraud": [0]})

    fe = FeatureEngineer()
    fe.create_target_encoding(train, fit=True)
    val_out = fe.create_target_encoding(val, fit=False, update_state=True)

    prior = fe._global_target_mean
    # Three prior frauds on the missing-key group, none of them this row's own.
    expected = (3.0 + TARGET_ENCODING_PRIOR_WEIGHT * prior) / (
        3.0 + TARGET_ENCODING_PRIOR_WEIGHT
    )

    assert val_out.loc[0, "card1_target_enc"] == pytest.approx(expected), (
        "The missing-key group's train history did not carry into val: got "
        f"{val_out.loc[0, 'card1_target_enc']}, expected {expected}. Carried "
        "state is being written but not read for NaN keys."
    )


@pytest.mark.unit
def test_missing_key_history_accumulates_over_three_splits() -> None:
    """
    The missing-key group must occupy exactly one slot in the carried state as
    it is updated across successive splits.

    Two splits are not enough to catch this. NaN is not usable as a dict key —
    `nan != nan`, and each frame's groupby yields a fresh NaN object — so every
    `update_state=True` call appends *another* missing-key entry rather than
    accumulating onto the previous one. Train→val still reads back the first
    entry and looks correct; the damage only surfaces on the third call, which
    is exactly the train→val→test order `run_pipeline` uses:
      - val's history silently replaces train's instead of adding to it, and
      - the duplicated key makes the state a non-unique index, so the
        `Series.map` in `_carried_totals` raises InvalidIndexError and
        preprocessing dies on the test split.
    """
    train = pd.DataFrame({"card1": [np.nan, np.nan], "isFraud": [1, 1]})
    val = pd.DataFrame({"card1": [np.nan], "isFraud": [0]})
    test = pd.DataFrame({"card1": [np.nan], "isFraud": [1]})

    fe = FeatureEngineer()
    fe.create_target_encoding(train, fit=True)
    fe.create_target_encoding(val, fit=False, update_state=True)
    test_out = fe.create_target_encoding(test, fit=False, update_state=False)

    missing_keys = [
        k for k in fe._target_enc_state["card1"] if isinstance(k, float) and np.isnan(k)
    ]
    assert not missing_keys, (
        f"Carried state holds {len(missing_keys)} raw NaN key(s). NaN cannot "
        "accumulate as a dict key; the missing-key group needs a single "
        "canonical sentinel."
    )

    # Two frauds from train plus one legitimate from val, none of them this
    # row's own label: sum 2 over 3 observations.
    prior = fe._global_target_mean
    expected = (2.0 + TARGET_ENCODING_PRIOR_WEIGHT * prior) / (
        3.0 + TARGET_ENCODING_PRIOR_WEIGHT
    )
    assert test_out.loc[0, "card1_target_enc"] == pytest.approx(expected), (
        "The missing-key group's train+val history did not accumulate into "
        f"test: got {test_out.loc[0, 'card1_target_enc']}, expected {expected}."
    )


# ─── PRD Phase 9 P9-2: UID client-identifier aggregates ──────────────────────


def _uid_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the columns create_uid_features needs. Only 3
    distinct (card1, addr1, D1n) triples so most UIDs clear _UID_MIN_COUNT."""
    rng = np.random.default_rng(seed)
    card1 = rng.choice([1001, 1002, 1003], n)
    addr1 = rng.choice([200.0, 300.0], n)
    d1 = rng.choice([0.0, 5.0], n)
    return pd.DataFrame(
        {
            "TransactionDT": np.zeros(n, dtype=float),  # day ~ 0 -> D1n = -D1
            "TransactionAmt": rng.uniform(10.0, 500.0, n),
            "card1": card1,
            "addr1": addr1,
            "D1": d1,
        }
    )


def test_uid_features_fit_then_apply_adds_expected_columns_without_uid_itself():
    fe = FeatureEngineer()
    train = _uid_df(seed=1)
    out = fe.create_uid_features(train, fit=True)

    for col in (
        "uid_count", "uid_freq", "uid_amt_mean", "uid_amt_std",
        "uid_d1_mean", "uid_amt_ratio",
    ):
        assert col in out.columns
    # The identifier must never be handed to the model.
    assert "UID" not in out.columns and "uid" not in out.columns
    assert out["uid_freq"].between(0.0, 1.0).all()
    assert (out["uid_count"] >= 1).all()  # every train row's own UID counted


def test_uid_count_and_freq_match_hand_count():
    fe = FeatureEngineer()
    df = pd.DataFrame(
        {
            "TransactionDT": [0.0, 0.0, 0.0, 0.0],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "card1": [1, 1, 1, 2],
            "addr1": [5.0, 5.0, 5.0, 9.0],
            "D1": [0.0, 0.0, 0.0, 0.0],
        }
    )
    # UIDs: "1_5_0" x3, "2_9_0" x1.
    out = fe.create_uid_features(df, fit=True)
    assert out.loc[0, "uid_count"] == 3
    assert out.loc[3, "uid_count"] == 1
    assert out.loc[0, "uid_freq"] == pytest.approx(0.75)
    assert out.loc[3, "uid_freq"] == pytest.approx(0.25)
    # "1_5_0" has 3 rows (>= _UID_MIN_COUNT) -> real cohort mean 20.0, ratio kept.
    assert out.loc[0, "uid_amt_mean"] == pytest.approx(20.0)
    assert out.loc[1, "uid_amt_ratio"] == pytest.approx(1.0)  # 20 / 20
    # "2_9_0" is a singleton -> aggregates fall back to global, ratio = 1.0.
    assert out.loc[3, "uid_amt_mean"] == pytest.approx(fe._uid_global["uid_amt_mean"])
    assert out.loc[3, "uid_amt_ratio"] == pytest.approx(1.0)


def test_uid_small_and_unseen_cohorts_get_neutral_ratio_and_global_stats():
    fe = FeatureEngineer()
    train = _uid_df(seed=2)
    fe.create_uid_features(train, fit=True)

    unseen = pd.DataFrame(
        {
            "TransactionDT": [0.0, 0.0],
            "TransactionAmt": [123.0, 456.0],
            "card1": [999999, 888888],
            "addr1": [1.0, 2.0],
            "D1": [3.0, 4.0],
        }
    )
    out = fe.create_uid_features(unseen, fit=False)
    assert (out["uid_count"] == 0).all()
    assert (out["uid_freq"] == 0.0).all()
    assert out["uid_amt_mean"].eq(fe._uid_global["uid_amt_mean"]).all()
    assert out["uid_amt_std"].eq(fe._uid_global["uid_amt_std"]).all()
    # Unseen UID -> neutral ratio, NOT amt/global (mle-reviewer M1).
    assert (out["uid_amt_ratio"] == 1.0).all()


def test_uid_features_missing_required_column_is_a_noop_with_warning(caplog):
    fe = FeatureEngineer()
    df = _uid_df().drop(columns=["D1"])
    import logging

    with caplog.at_level(logging.WARNING):
        out = fe.create_uid_features(df, fit=True)
    assert "uid_freq" not in out.columns
    assert any("create_uid_features" in m for m in caplog.messages)


def test_uid_features_survive_transformer_save_load(tmp_path):
    fe = FeatureEngineer()
    train = _uid_df(seed=3)
    fe.create_uid_features(train, fit=True)
    fe.save_transformers(str(tmp_path))

    restored = FeatureEngineer()
    restored.load_transformers(str(tmp_path))
    assert restored._uid_maps == fe._uid_maps
    assert restored._uid_global == fe._uid_global

    a = fe.create_uid_features(train, fit=False)
    b = restored.create_uid_features(train, fit=False)
    pd.testing.assert_frame_equal(a, b)


def test_uid_missing_addr1_vs_missing_d1_do_not_collapse_together():
    """A card missing only addr1 must not share a UID with the same card
    missing only D1, nor with the same card missing both (mle-reviewer M3)."""
    fe = FeatureEngineer()
    df = pd.DataFrame(
        {
            "TransactionDT": [0.0, 0.0, 0.0],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "card1": [7, 7, 7],
            "addr1": [np.nan, 5.0, np.nan],
            "D1": [1.0, np.nan, np.nan],
        }
    )
    uid = fe._compute_uid(df)
    assert uid.nunique() == 3  # "7_naA_-1", "7_5_naD", "7_naA_naD" all distinct


# ─── PRD Phase 9 step 9.4 — multi-window RFM/velocity + dist1 ────────────────
#
# Written before the implementation (TDD RED). The properties below are the
# ones a fixed trailing window can silently get wrong: including the current
# row, counting another card's transactions, or letting the window boundary
# drift with input order rather than with TransactionDT.


ROLLING_WINDOW_COLS = [
    "tx_count_10min_per_card",
    "tx_count_1h_per_card",
    "tx_count_24h_per_card",
]


def _naive_rolling_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    O(n^2) per-row oracle: for each row, count STRICTLY PRIOR transactions of
    the same card whose TransactionDT falls inside the trailing window.

    Deliberately the dumbest correct implementation — the vectorised version
    is only trustworthy if it agrees with this on every row.
    """
    windows = {
        "tx_count_10min_per_card": 600.0,
        "tx_count_1h_per_card": 3600.0,
        "tx_count_24h_per_card": 86400.0,
    }
    out = pd.DataFrame(index=df.index)
    for col, span in windows.items():
        vals = []
        for i in df.index:
            card = df.at[i, "card1"]
            t = df.at[i, "TransactionDT"]
            prior = df[(df["card1"] == card) & (df["TransactionDT"] < t)]
            vals.append(int((prior["TransactionDT"] >= t - span).sum()))
        out[col] = vals
    return out


@pytest.mark.unit
def test_rolling_window_counts_match_naive_oracle(
    card_history_df: pd.DataFrame,
) -> None:
    """Vectorised trailing-window counts must equal the per-row oracle."""
    result = FeatureEngineer().create_velocity_features(card_history_df.copy())
    expected = _naive_rolling_counts(
        card_history_df.sort_values("TransactionDT").reset_index(drop=True)
    )

    for col in ROLLING_WINDOW_COLS:
        assert col in result.columns, f"missing rolling-window feature '{col}'"
        np.testing.assert_array_equal(
            result[col].to_numpy(dtype=int),
            expected[col].to_numpy(dtype=int),
            err_msg=(
                f"'{col}' disagrees with a per-row count over strictly prior "
                "same-card transactions inside the trailing window."
            ),
        )


@pytest.mark.unit
def test_rolling_window_counts_exclude_the_current_row(
    card_history_df: pd.DataFrame,
) -> None:
    """A card's first transaction has no prior history in ANY window, so every
    trailing count must be exactly 0 there — the sharpest symptom of a
    self-inclusive window."""
    result = FeatureEngineer().create_velocity_features(card_history_df.copy())
    first_rows = result.groupby("card1", sort=False).head(1)

    for col in ROLLING_WINDOW_COLS:
        offenders = first_rows.loc[first_rows[col] != 0, ["card1", col]]
        assert offenders.empty, (
            f"CURRENT-ROW LEAKAGE in '{col}': a card's first transaction has a "
            f"non-zero trailing count:\n{offenders}"
        )


@pytest.mark.unit
def test_rolling_windows_are_nested_and_bounded_by_expanding_count(
    card_history_df: pd.DataFrame,
) -> None:
    """10min <= 1h <= 24h <= expanding count, row-wise. A window that counted
    other cards, or that double-counted, breaks this ordering."""
    fe = FeatureEngineer()
    result = fe.create_card_aggregates(card_history_df.copy())
    result = fe.create_velocity_features(result)

    assert (result["tx_count_10min_per_card"] <= result["tx_count_1h_per_card"]).all()
    assert (result["tx_count_1h_per_card"] <= result["tx_count_24h_per_card"]).all()
    assert (
        result["tx_count_24h_per_card"] <= result["tx_count_per_card"]
    ).all(), "a trailing-window count exceeded the card's total prior count"


@pytest.mark.unit
def test_rolling_windows_ignore_other_cards() -> None:
    """Interleaving a second card's transactions must not change the first
    card's window counts."""
    base = pd.DataFrame(
        {
            "TransactionDT": [0.0, 100.0, 200.0, 300.0],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "card1": [1, 1, 1, 1],
        }
    )
    noisy = pd.DataFrame(
        {
            "TransactionDT": [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
            "TransactionAmt": [10.0, 99.0, 20.0, 99.0, 30.0, 99.0, 40.0],
            "card1": [1, 2, 1, 2, 1, 2, 1],
        }
    )
    fe = FeatureEngineer()
    a = fe.create_velocity_features(base.copy())
    b = fe.create_velocity_features(noisy.copy())
    b_card1 = b[b["card1"] == 1].reset_index(drop=True)

    for col in ROLLING_WINDOW_COLS:
        np.testing.assert_array_equal(
            a[col].to_numpy(dtype=int),
            b_card1[col].to_numpy(dtype=int),
            err_msg=f"'{col}' changed when another card's rows were interleaved",
        )


@pytest.mark.unit
def test_rolling_window_boundary_is_half_open_on_time_not_position() -> None:
    """A transaction exactly `span` seconds old is INSIDE the window
    (>= t-span); one a hair older is outside. Pins the boundary to time,
    not to row position."""
    fe = FeatureEngineer()
    df = pd.DataFrame(
        {
            "TransactionDT": [0.0, 1.0, 600.0],
            "TransactionAmt": [10.0, 10.0, 10.0],
            "card1": [1, 1, 1],
        }
    )
    out = fe.create_velocity_features(df.copy())
    # at t=600 the window opens at 0.0, so both prior rows are inside
    assert out.loc[2, "tx_count_10min_per_card"] == 2

    df2 = df.copy()
    df2.loc[2, "TransactionDT"] = 601.0  # window opens at 1.0 -> drops t=0
    out2 = fe.create_velocity_features(df2)
    assert out2.loc[2, "tx_count_10min_per_card"] == 1


@pytest.mark.unit
def test_short_vs_long_window_amount_deviation_excludes_current_row() -> None:
    """`amt_24h_vs_card_mean_ratio` compares the trailing-24h mean against the
    card's expanding mean, both over strictly prior rows. A card's first
    transaction has neither, so the ratio must take its neutral 1.0."""
    fe = FeatureEngineer()
    df = pd.DataFrame(
        {
            "TransactionDT": [0.0, 10.0, 20.0, 30.0],
            "TransactionAmt": [100.0, 100.0, 100.0, 5000.0],
            "card1": [1, 1, 1, 1],
        }
    )
    out = fe.create_card_aggregates(df.copy())
    out = fe.create_velocity_features(out)

    assert "amt_24h_mean_per_card" in out.columns
    assert "amt_24h_vs_card_mean_ratio" in out.columns
    # first row: no history at all -> neutral
    assert out.loc[0, "amt_24h_vs_card_mean_ratio"] == pytest.approx(1.0)
    # all prior amounts identical and inside 24h -> short mean == long mean
    assert out.loc[2, "amt_24h_vs_card_mean_ratio"] == pytest.approx(1.0)
    # the 5000 row's own amount must NOT enter its own trailing mean
    assert out.loc[3, "amt_24h_mean_per_card"] == pytest.approx(100.0)


@pytest.mark.unit
def test_rolling_windows_are_invariant_to_input_row_order(
    card_history_df: pd.DataFrame,
) -> None:
    """Shuffling input rows must not change the result — the function sorts by
    TransactionDT internally, as the batch pipeline relies on."""
    fe = FeatureEngineer()
    ordered = fe.create_velocity_features(card_history_df.copy())
    shuffled = fe.create_velocity_features(
        card_history_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    )
    for col in ROLLING_WINDOW_COLS:
        np.testing.assert_array_equal(
            ordered[col].to_numpy(dtype=int),
            shuffled[col].to_numpy(dtype=int),
            err_msg=f"'{col}' depends on input row order",
        )


# ─── 9.4 dist1 engineering (scoped to ProductCD == 'W') ──────────────────────


@pytest.mark.unit
def test_dist1_features_only_fire_on_product_w() -> None:
    """`dist1` is non-null for ProductCD=='W' only (verified against the raw
    data 2026-09-08). Non-W rows must take the neutral fill, never a value
    derived from a null."""
    fe = FeatureEngineer()
    df = pd.DataFrame(
        {
            "ProductCD": ["W", "W", "C", "H"],
            "dist1": [10.0, 100.0, np.nan, np.nan],
        }
    )
    out = fe.create_dist_features(df.copy(), fit=True)

    assert out.loc[0, "dist1_log"] == pytest.approx(np.log1p(10.0))
    assert out.loc[1, "dist1_log"] == pytest.approx(np.log1p(100.0))
    # non-W rows: neutral sentinel, and never flagged high
    assert out.loc[2, "dist1_log"] == -1.0
    assert out.loc[3, "dist1_log"] == -1.0
    assert out.loc[2, "dist1_high"] == 0
    assert out.loc[3, "dist1_high"] == 0


@pytest.mark.unit
def test_dist1_high_threshold_is_fit_on_train_only() -> None:
    """The top-quintile cut is fitted on train and REUSED unchanged on later
    splits. Re-fitting per split would leak the holdout's own distribution."""
    fe = FeatureEngineer()
    train = pd.DataFrame({"ProductCD": ["W"] * 10, "dist1": np.arange(1.0, 11.0)})
    fitted = fe.create_dist_features(train.copy(), fit=True)
    cut = fe._dist1_high_threshold
    assert cut is not None

    # A holdout whose values are all far larger must NOT move the cut.
    holdout = pd.DataFrame({"ProductCD": ["W"] * 5, "dist1": np.arange(500.0, 505.0)})
    applied = fe.create_dist_features(holdout.copy(), fit=False)
    assert fe._dist1_high_threshold == cut, "threshold was re-fit on holdout"
    assert (applied["dist1_high"] == 1).all(), "holdout rows are all above the cut"
    assert fitted["dist1_high"].sum() > 0


@pytest.mark.unit
def test_dist1_features_survive_transformer_save_load(tmp_path) -> None:
    """The fitted cut must round-trip, or serving silently re-derives a
    different `dist1_high` than training used."""
    fe = FeatureEngineer()
    train = pd.DataFrame({"ProductCD": ["W"] * 20, "dist1": np.arange(1.0, 21.0)})
    fe.create_dist_features(train.copy(), fit=True)
    fe.save_transformers(str(tmp_path))

    restored = FeatureEngineer()
    restored.load_transformers(str(tmp_path))
    assert restored._dist1_high_threshold == fe._dist1_high_threshold

    probe = pd.DataFrame({"ProductCD": ["W"] * 20, "dist1": np.arange(1.0, 21.0)})
    pd.testing.assert_frame_equal(
        fe.create_dist_features(probe.copy(), fit=False),
        restored.create_dist_features(probe.copy(), fit=False),
    )


@pytest.mark.unit
def test_dist_features_missing_columns_is_a_noop() -> None:
    """A frame without ProductCD/dist1 must pass through unchanged rather than
    raising — matching how the other feature groups degrade."""
    fe = FeatureEngineer()
    df = pd.DataFrame({"TransactionAmt": [1.0, 2.0]})
    out = fe.create_dist_features(df.copy(), fit=True)
    assert list(out.columns) == ["TransactionAmt"]
