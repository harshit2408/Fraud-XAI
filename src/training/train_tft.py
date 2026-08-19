"""
src/training/train_tft.py

Training script for the Temporal Fusion Transformer (TFT) fraud detector.

Mirrors the structure of train_xgb.py:
  - Loads processed parquet data
  - Builds sequences from flat features
  - Trains TFT with early stopping on validation PR-AUC
  - Evaluates on all splits with overfitting analysis
  - Logs everything to MLflow
  - Saves model checkpoint

Usage:
    python src/training/train_tft.py
    python src/training/train_tft.py --config config/config.yaml --epochs 50
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import Settings, load_settings
from src.data.sequence_builder import SequenceBuilder
from src.device import resolve_device
from src.evaluation.evaluator import ModelEvaluator
from src.models.tft_model import TemporalFusionTransformer
from src.training.losses import FocalLoss, WeightedBCELoss
from src.training.manifest import build_manifest, write_manifest
from src.utils.checksums import verify_checksums, write_checksums
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Imbalance handling (Phase B1/B2) ────────────────────────────────────────────

VALID_SAMPLING_STRATEGIES = {"none", "oversample", "smote"}
VALID_LOSS_FUNCTIONS = {"focal_loss", "weighted_bce", "bce"}


def resolve_imbalance_config(
    imb_cfg: Dict[str, Any],
    pos_count: int,
    neg_count: int,
) -> Tuple[bool, nn.Module]:
    """
    Resolve `config.imbalance` into a concrete (use_sampler, criterion) pair,
    guaranteeing exactly one imbalance-correction mechanism is ever active.

    Background (HIGH findings, docs/IMPLEMENTATION_PLAN.md Phase B):
      - `imbalance.strategy` used to be a single ambiguous key compared against
        "focal_loss" in the loss branch; the shipped value "smote" always fell
        through to WeightedBCELoss, silently disabling focal loss.
      - Independently, `oversample=True` installed a WeightedRandomSampler that
        rebalances the training stream to ~50/50 (~27x effective resampling for
        this dataset), while WeightedBCELoss *also* computed a ~27x pos_weight
        from the pre-sampling distribution — a ~729x double correction.

    This function fixes both: `sampling_strategy` and `loss_function` are
    validated independently and fail fast on unknown values, and when the
    sampler is active, `weighted_bce`'s count-based pos_weight collapses to
    1.0 (the sampler is left as the sole correction mechanism). FocalLoss's
    alpha is a fixed hyperparameter, not derived from the imbalance ratio, so
    it does not compound with the sampler the way pos_weight did.

    Args:
        imb_cfg: `config["imbalance"]` dict.
        pos_count: Number of positive (fraud) examples in the training split,
            before any resampling.
        neg_count: Number of negative examples in the training split.

    Returns:
        (use_sampler, criterion): whether `_create_dataloader` should install a
        WeightedRandomSampler, and the loss module to train with.

    Raises:
        ValueError: `sampling_strategy` or `loss_function` is missing or not
            one of the supported values.
        NotImplementedError: `sampling_strategy == "smote"` — SMOTE has no
            defined per-sequence semantics for the TFT's temporal input and is
            not implemented here.
    """
    sampling_strategy = imb_cfg.get("sampling_strategy")
    if sampling_strategy not in VALID_SAMPLING_STRATEGIES:
        raise ValueError(
            f"Unknown or missing imbalance.sampling_strategy: {sampling_strategy!r}. "
            f"Must be one of {sorted(VALID_SAMPLING_STRATEGIES)}."
        )

    loss_function = imb_cfg.get("loss_function")
    if loss_function not in VALID_LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown or missing imbalance.loss_function: {loss_function!r}. "
            f"Must be one of {sorted(VALID_LOSS_FUNCTIONS)}."
        )

    if sampling_strategy == "smote":
        raise NotImplementedError(
            "imbalance.sampling_strategy='smote' is not supported by TFTTrainer: "
            "SMOTE interpolates in flat feature space and has no defined "
            "equivalent for temporal sequences. Use 'oversample' or 'none'."
        )

    use_sampler = sampling_strategy == "oversample"

    if loss_function == "focal_loss":
        gamma = imb_cfg.get("focal_loss_gamma", 2.0)
        alpha = imb_cfg.get("focal_loss_alpha", 0.25)
        criterion: nn.Module = FocalLoss(gamma=gamma, alpha=alpha)
        logger.info(f"Using FocalLoss (gamma={gamma}, alpha={alpha})")
    elif loss_function == "weighted_bce":
        if use_sampler:
            # The sampler already rebalances the training stream; applying the
            # full count-based pos_weight on top would double-correct.
            pos_weight = 1.0
            logger.info(
                "Sampler active (sampling_strategy=oversample): forcing "
                "WeightedBCELoss pos_weight=1.0 to avoid double correction."
            )
        else:
            pos_weight = neg_count / max(pos_count, 1)
        criterion = WeightedBCELoss(pos_weight=pos_weight)
        logger.info(f"Using WeightedBCELoss (pos_weight={pos_weight:.2f})")
    else:  # "bce"
        criterion = WeightedBCELoss(pos_weight=1.0)
        logger.info("Using plain BCE (pos_weight=1.0)")

    return use_sampler, criterion


def _artifact_paths(base_path: Path) -> Dict[str, Path]:
    """
    Derive the three files a TFTTrainer checkpoint is actually split across
    (Phase D6, docs/IMPLEMENTATION_PLAN.md).

    `base_path` (e.g. "models/tft_model.ckpt") is treated purely as a stem
    — the literal path is never written to:
      - "weights":   torch.save() of the model's state_dict ONLY (a plain
                      OrderedDict[str, Tensor]) — loadable with
                      torch.load(weights_only=True), the strict unpickler
                      that refuses arbitrary classes. Everything that isn't
                      a tensor lives in "metadata" instead, so this file
                      never needs weights_only=False to load.
      - "metadata":  config, model_config, training_history, sequence
                      builder state (including the fitted scaler — an
                      sklearn transformer object), threshold, calibrator —
                      via joblib.
      - "checksums": sha256 of the two files above; verified before either
                      is deserialized (see TFTTrainer.load).
    """
    return {
        "weights": base_path.with_name(f"{base_path.stem}.weights.pt"),
        "metadata": base_path.with_name(f"{base_path.stem}.meta.joblib"),
        "checksums": base_path.with_name(f"{base_path.stem}.checksums.json"),
    }


# ── PyTorch Dataset wrapper ────────────────────────────────────────────────────


class FraudSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-built sequence arrays.

    Args:
        sequences: np.ndarray of shape (N, seq_len, feature_dim)
        static_features: np.ndarray of shape (N, static_dim)
        targets: np.ndarray of shape (N,)
        mask: np.ndarray of shape (N, seq_len), 1=real, 0=padding
    """

    def __init__(
        self,
        sequences: np.ndarray,
        static_features: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        self.sequences = torch.from_numpy(sequences).float()
        self.static_features = torch.from_numpy(static_features).float()
        self.targets = torch.from_numpy(targets).float()
        self.mask = torch.from_numpy(mask).float()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.sequences[idx],
            self.static_features[idx],
            self.mask[idx],
            self.targets[idx],
        )


# ── TFT Trainer ────────────────────────────────────────────────────────────────


class TFTTrainer:
    """
    End-to-end trainer for the Temporal Fusion Transformer.

    Handles model construction, training loop, evaluation, and persistence.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model: Optional[TemporalFusionTransformer] = None
        self.sequence_builder: Optional[SequenceBuilder] = None
        self.device = self._get_device()
        self._best_model_state: Optional[Dict] = None
        self._training_history: list = []
        # Phase C4: threshold and calibrator are fit once (on validation)
        # during training and frozen into the checkpoint. Serving must read
        # them via predict()/predict_proba_calibrated(), never recompute.
        self.threshold: Optional[float] = None
        self.calibrator: Any = None

    def set_threshold(self, threshold: float) -> None:
        """Freeze the validation-selected decision threshold (Phase C1/C4)."""
        self.threshold = threshold

    def set_calibrator(self, calibrator: Any) -> None:
        """Attach the validation-fitted probability calibrator (Phase C2/C4)."""
        self.calibrator = calibrator

    def _get_device(self) -> torch.device:
        """Detect best available device (CUDA > CPU).

        Phase D7: delegates to the shared src.device.resolve_device() helper
        (this method's own "auto" logic was the model for that extraction),
        so XGBoost/LightGBM/TFT all resolve "auto" identically.
        """
        tft_cfg = self.config.get("model", {}).get("tft", {})
        device_str = resolve_device(tft_cfg.get("device", "auto"))

        device = torch.device(device_str)
        if device_str == "cuda":
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            logger.info(f"Using device: {device}")

        return device

    def build_model(
        self,
        num_numeric_features: int,
        num_static_features: int,
        static_categorical_indices: Optional[list] = None,
        static_cardinalities: Optional[list] = None,
    ) -> TemporalFusionTransformer:
        """Instantiate TFT from config parameters.

        Phase B5: `static_categorical_indices`/`static_cardinalities` route
        label-encoded static columns (ProductCD, card4, card6, ...) through
        embedding tables instead of a Linear layer treating them as
        continuous. Omit for the original all-continuous behavior.
        """
        tft_cfg = self.config.get("model", {}).get("tft", {})

        self.model = TemporalFusionTransformer(
            num_numeric_features=num_numeric_features,
            num_static_features=num_static_features,
            hidden_size=tft_cfg.get("hidden_size", 64),
            num_attention_heads=tft_cfg.get("attention_head_size", 4),
            dropout=tft_cfg.get("dropout", 0.1),
            num_lstm_layers=tft_cfg.get("num_lstm_layers", 1),
            sequence_length=self.config["data"].get("sequence_length", 10),
            static_categorical_indices=static_categorical_indices,
            static_cardinalities=static_cardinalities,
        ).to(self.device)

        n_params = self.model.count_parameters()
        logger.info(f"TFT model built: {n_params:,} trainable parameters")

        return self.model

    def _build_sequences(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        split_name: str,
        fit_scaler: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Build sequences from features, optionally with labels (Phase B7)."""
        seq_len = self.config["data"].get("sequence_length", 10)

        if self.sequence_builder is None:
            self.sequence_builder = SequenceBuilder(
                sequence_length=seq_len,
                group_col="card1",
            )

        # card1 may have been dropped during preprocessing; reconstruct from data
        # If card1 is not in X, we need it for grouping
        if "card1" not in X.columns:
            logger.warning(
                "card1 not found in features. Using index-based sequential grouping."
            )
            X = X.copy()
            X["card1"] = 0  # All in one group — sequential sliding window

        logger.info(f"Building {split_name} sequences...")
        seq_data = self.sequence_builder.build_sequences(X, y, fit_scaler=fit_scaler)
        n_seq = len(seq_data["mask"])
        if seq_data.get("targets") is not None:
            logger.info(
                f"  {split_name}: {n_seq:,} sequences, "
                f"fraud rate: {seq_data['targets'].mean()*100:.2f}%"
            )
        else:
            logger.info(f"  {split_name}: {n_seq:,} sequences (no labels)")

        return seq_data

    def _build_sequences_for_splits(
        self,
        splits: list,
        fit_scaler: bool = False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Build sequences once over the concatenation of `splits` (Phase B6).

        Building each split in isolation truncates a card's history at the
        split boundary — e.g. val's first transaction for a card looks like
        it has no prior context, even though real history exists in train.
        This concatenates the given (name, X, y) splits in order, builds
        sequences once over the combined frame, then partitions the result
        back out by each row's original split membership so boundary cards
        retain their real history.

        Args:
            splits: List of (name, X, y) tuples, in temporal order. `y` may
                be None for all splits (Phase B7) or provided for all of them;
                mixing None and non-None across splits is not supported.
            fit_scaler: Passed through to the single combined build call —
                pass True only when the first split is the training set.

        Returns:
            Dict mapping each split name to its own sequence dict (same
            schema as `build_sequences`), with `original_indices` rebased to
            be relative to that split's own `X`.
        """
        names = [name for name, _, _ in splits]
        frames = [X for _, X, _ in splits]
        lengths = [len(X) for X in frames]
        labels = [y for _, _, y in splits]

        combined_X = pd.concat(frames, axis=0, ignore_index=True)
        if all(y is not None for y in labels):
            combined_y = pd.concat(
                [pd.Series(y).reset_index(drop=True) for y in labels],
                axis=0,
                ignore_index=True,
            )
        elif all(y is None for y in labels):
            combined_y = None
        else:
            raise ValueError(
                "_build_sequences_for_splits: labels must be provided for "
                "all splits or none — got a mix of labeled and unlabeled splits."
            )

        combined_seq = self._build_sequences(
            combined_X, combined_y, "+".join(names), fit_scaler=fit_scaler
        )

        result: Dict[str, Dict[str, np.ndarray]] = {}
        offset = 0
        indices = combined_seq["original_indices"]
        for name, length in zip(names, lengths):
            lo, hi = offset, offset + length
            keep = (indices >= lo) & (indices < hi)
            split_seq: Dict[str, np.ndarray] = {}
            for key, value in combined_seq.items():
                if isinstance(value, np.ndarray) and len(value) == len(keep):
                    split_seq[key] = value[keep]
                else:
                    split_seq[key] = value
            split_seq["original_indices"] = split_seq["original_indices"] - lo
            result[name] = split_seq
            offset = hi

        return result

    def _create_dataloader(
        self,
        seq_data: Dict[str, np.ndarray],
        batch_size: int,
        shuffle: bool = False,
        oversample: bool = False,
    ) -> DataLoader:
        """Create a DataLoader from sequence data."""
        dataset = FraudSequenceDataset(
            sequences=seq_data["sequences"],
            static_features=seq_data["static"],
            targets=seq_data["targets"],
            mask=seq_data["mask"],
        )

        sampler = None
        if oversample:
            # Weighted random sampling to handle class imbalance
            targets = seq_data["targets"]
            class_counts = np.bincount(targets.astype(int))
            weights = np.where(targets == 1, 1.0 / class_counts[1], 1.0 / class_counts[0])
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False  # Sampler handles ordering

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=0,  # Windows compatibility
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        trial: Optional[optuna.Trial] = None,
    ) -> Dict[str, list]:
        """
        Full training loop with early stopping on validation PR-AUC.

        Returns:
            Training history dict with per-epoch metrics.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        tft_cfg = self.config.get("model", {}).get("tft", {})
        imb_cfg = self.config.get("imbalance", {})
        batch_size = tft_cfg.get("batch_size", 128)
        max_epochs = tft_cfg.get("max_epochs", 30)
        lr = tft_cfg.get("learning_rate", 0.001)
        patience = tft_cfg.get("patience", 5)
        grad_clip = tft_cfg.get("gradient_clip_val", 1.0)

        # Build sequences once over train+val so val's first transactions per
        # card retain real train history instead of being padded at the
        # split boundary (Phase B6). Scaler is fit once, on train only.
        split_seqs = self._build_sequences_for_splits(
            [("train", X_train, y_train), ("val", X_val, y_val)],
            fit_scaler=True,
        )
        train_seq = split_seqs["train"]
        val_seq = split_seqs["val"]

        # Resolve imbalance handling: exactly one mechanism (sampler XOR loss
        # weighting), chosen explicitly by config — see resolve_imbalance_config.
        pos_count = int(train_seq["targets"].sum())
        neg_count = len(train_seq["targets"]) - pos_count
        use_sampler, criterion = resolve_imbalance_config(imb_cfg, pos_count, neg_count)
        criterion = criterion.to(self.device)

        # Create dataloaders
        train_loader = self._create_dataloader(
            train_seq, batch_size, shuffle=True, oversample=use_sampler
        )
        val_loader = self._create_dataloader(val_seq, batch_size, shuffle=False)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # Maximize PR-AUC
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6,
        )

        # Mixed precision (Phase B4): was hard-disabled to work around a NaN
        # issue plausibly caused by unscaled -999 sentinels dominating the
        # gradient signal. Now that features are scaled (see
        # _build_sequences_for_splits -> fit_scaler), re-enable by default;
        # kept config-controllable (imbalance/config.yaml model.tft.use_amp)
        # as an escape hatch if NaNs reappear on a given GPU/driver combo.
        # AMP is CUDA-only — autocast on CPU provides no benefit here.
        use_amp = tft_cfg.get("use_amp", True) and self.device.type == "cuda"
        scaler = GradScaler(enabled=use_amp)

        # Training loop
        evaluator = ModelEvaluator()
        best_val_pr_auc = 0.0
        epochs_without_improvement = 0
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_pr_auc": [],
            "val_pr_auc": [],
            "lr": [],
        }

        logger.info("=" * 60)
        logger.info("STARTING TFT TRAINING")
        logger.info(f"  Epochs: {max_epochs}, Batch size: {batch_size}")
        logger.info(f"  Learning rate: {lr}, Patience: {patience}")
        logger.info(f"  Device: {self.device}, AMP: {use_amp}")
        logger.info(f"  Train sequences: {len(train_seq['targets']):,}")
        logger.info(f"  Val sequences: {len(val_seq['targets']):,}")
        logger.info("=" * 60)

        train_start = time.time()

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            # ── Train epoch ──────────────────────────────────────────────
            self.model.train()
            train_losses = []
            train_preds = []
            train_labels = []

            for batch_idx, (seqs, static, masks, targets) in enumerate(train_loader):
                seqs = seqs.to(self.device)
                static = static.to(self.device)
                masks = masks.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                with autocast(enabled=use_amp):
                    output = self.model(seqs, static, masks)
                    logits = output["logits"].squeeze(-1)
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                train_losses.append(loss.item())
                with torch.no_grad():
                    probs = output["probabilities"].squeeze(-1)
                    train_preds.append(probs.cpu().numpy())
                    train_labels.append(targets.cpu().numpy())

            train_loss = np.mean(train_losses)
            train_preds_arr = np.concatenate(train_preds)
            train_labels_arr = np.concatenate(train_labels)
            train_pr_auc = evaluator.compute_pr_auc(train_labels_arr, train_preds_arr)

            # ── Validation epoch ─────────────────────────────────────────
            val_loss, val_preds_arr, val_labels_arr = self._evaluate_epoch(
                val_loader, criterion, use_amp
            )
            val_pr_auc = evaluator.compute_pr_auc(val_labels_arr, val_preds_arr)

            # Scheduler step
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_pr_auc)

            # Record history
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_pr_auc"].append(train_pr_auc)
            history["val_pr_auc"].append(val_pr_auc)
            history["lr"].append(current_lr)

            # Optuna Pruning
            if trial is not None:
                trial.report(val_pr_auc, epoch)
                if trial.should_prune():
                    logger.info(f"  Trial pruned at epoch {epoch}")
                    raise optuna.TrialPruned()

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train PR-AUC: {train_pr_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )

            # Log to MLflow
            if mlflow.active_run():
                mlflow.log_metrics(
                    {
                        "tft_train_loss": train_loss,
                        "tft_val_loss": val_loss,
                        "tft_train_pr_auc": train_pr_auc,
                        "tft_val_pr_auc": val_pr_auc,
                        "tft_lr": current_lr,
                    },
                    step=epoch,
                )

            # Early stopping check
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                epochs_without_improvement = 0
                # Save best model state
                self._best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                logger.info(f"  ★ New best Val PR-AUC: {best_val_pr_auc:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        f"  Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

        total_time = time.time() - train_start
        logger.info(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")

        # Restore best model
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            self.model.to(self.device)
            logger.info(f"Restored best model (Val PR-AUC: {best_val_pr_auc:.4f})")

        self._training_history = history
        return history

    def _evaluate_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        use_amp: bool,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Run validation/test epoch. Returns (avg_loss, predictions, labels)."""
        self.model.eval()
        losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for seqs, static, masks, targets in dataloader:
                seqs = seqs.to(self.device)
                static = static.to(self.device)
                masks = masks.to(self.device)
                targets = targets.to(self.device)

                with autocast(enabled=use_amp):
                    output = self.model(seqs, static, masks)
                    logits = output["logits"].squeeze(-1)
                    loss = criterion(logits, targets)

                losses.append(loss.item())
                probs = output["probabilities"].squeeze(-1)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

        return (
            np.mean(losses),
            np.concatenate(all_preds),
            np.concatenate(all_labels),
        )

    def predict_proba(
        self,
        X: pd.DataFrame,
        history_X: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Predict fraud probabilities for a set of transactions.

        Phase B7: takes `X` alone — no labels. Labels are not available at
        prediction time in production; the previous `predict_proba(X, y)`
        signature was a latent leakage smell (a real leak the moment someone
        wired unavailable labels into this call in serving).

        Args:
            X: Feature DataFrame (same format as training).
            history_X: Optional preceding transactions (chronologically
                before `X`, same schema) used only to build real historical
                context for `X`'s sequences (Phase B6) — e.g. pass the
                validation set when predicting on test, so test's first
                transactions per card aren't padded as if they had no prior
                context. Not required; omit for a standalone prediction.

        Returns:
            np.ndarray of fraud probabilities, shape (N,) — aligned to `X`'s
            row order regardless of `history_X`.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        tft_cfg = self.config.get("model", {}).get("tft", {})
        batch_size = tft_cfg.get("batch_size", 128)

        if history_X is not None and len(history_X) > 0:
            combined = pd.concat([history_X, X], axis=0, ignore_index=True)
            history_len = len(history_X)
        else:
            combined = X
            history_len = 0

        seq_data = self._build_sequences(combined, None, "predict")

        if history_len > 0:
            keep = seq_data["original_indices"] >= history_len
            filtered: Dict[str, np.ndarray] = {}
            for key, value in seq_data.items():
                if isinstance(value, np.ndarray) and len(value) == len(keep):
                    filtered[key] = value[keep]
                else:
                    filtered[key] = value
            filtered["original_indices"] = filtered["original_indices"] - history_len
            seq_data = filtered

        n = len(seq_data["mask"])
        if seq_data.get("targets") is None:
            seq_data = {**seq_data, "targets": np.zeros(n, dtype=np.float32)}

        dataloader = self._create_dataloader(seq_data, batch_size, shuffle=False)

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for seqs, static, masks, targets in dataloader:
                seqs = seqs.to(self.device)
                static = static.to(self.device)
                masks = masks.to(self.device)

                output = self.model(seqs, static, masks)
                probs = output["probabilities"].squeeze(-1)
                all_preds.append(probs.cpu().numpy())

        all_preds_concat = np.concatenate(all_preds)
        
        # Restore original chronological row order from the flat dataframe
        original_indices = seq_data.get("original_indices")
        if original_indices is not None and len(original_indices) == len(all_preds_concat):
            sorted_preds = np.zeros_like(all_preds_concat)
            sorted_preds[original_indices] = all_preds_concat
            return sorted_preds

        return all_preds_concat

    def predict_proba_calibrated(
        self, X: pd.DataFrame, history_X: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Raw predict_proba passed through the frozen calibrator (Phase C2/C4).

        Raises if no calibrator was attached via `set_calibrator` (or
        restored via `load`) — must never silently fall back to
        uncalibrated output.

        CAUTION: `self.threshold` was selected against RAW (uncalibrated)
        probabilities. Do NOT binarize this method's output at
        `self.threshold` — use `predict()` for the frozen operating
        point; use this method only where a calibrated confidence VALUE
        is needed.
        """
        if self.calibrator is None:
            raise ValueError("No calibrator attached. Call set_calibrator() or load a calibrated artifact.")
        return self.calibrator.predict(self.predict_proba(X, history_X=history_X))

    def predict(
        self, X: pd.DataFrame, history_X: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Binarize RAW predict_proba at the frozen threshold (Phase C1/C4).

        Raises if no threshold was frozen — serving must never invent a
        default or silently recompute one from whatever data it has.
        `self.threshold` is calibrated for raw probabilities specifically;
        see the caution note on `predict_proba_calibrated`.
        """
        if self.threshold is None:
            raise ValueError("No threshold frozen. Call set_threshold() or load a thresholded artifact.")
        return (self.predict_proba(X, history_X=history_X) >= self.threshold).astype(int)

    def save(self, path: str) -> None:
        """Save model checkpoint and metadata (Phase D6: no
        weights_only=False anywhere on this path).

        Writes three files derived from `path` (see `_artifact_paths`): the
        model's state_dict ALONE via torch.save (nothing else — no config
        dict, no sklearn scaler object — so it never needs
        weights_only=False to load back), a joblib metadata sidecar holding
        everything else (config, model_config, training_history, sequence
        builder state including the fitted scaler, frozen threshold,
        calibrator), and a sha256 checksum manifest covering both —
        verified by `load()` before either file is deserialized.
        """
        if self.model is None:
            raise ValueError("Model not trained. Cannot save.")

        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        paths = _artifact_paths(base_path)

        torch.save(self.model.state_dict(), paths["weights"])

        metadata = {
            "config": self.config,
            "model_config": {
                "num_numeric_features": self.model.num_numeric_features,
                "num_static_features": self.model.num_static_features,
                "hidden_size": self.model.hidden_size,
                "sequence_length": self.model.sequence_length,
                # Phase B5 — needed to reconstruct the embedding tables on load.
                "static_categorical_indices": self.model.static_categorical_indices,
                "static_cardinalities": [
                    emb.num_embeddings for emb in self.model.categorical_embeddings
                ],
            },
            "training_history": self._training_history,
            "sequence_builder": {
                "numeric_features": self.sequence_builder._numeric_features if self.sequence_builder else [],
                "static_features": self.sequence_builder._static_features if self.sequence_builder else [],
                "feature_dim": self.sequence_builder._feature_dim if self.sequence_builder else 0,
                "static_dim": self.sequence_builder._static_dim if self.sequence_builder else 0,
                # Phase B4 — persist the train-fitted scaler so val/test/
                # serving transform with it instead of silently going unscaled.
                "scaler": self.sequence_builder.scaler if self.sequence_builder else None,
                "static_cardinalities_by_name": self.sequence_builder._static_cardinalities if self.sequence_builder else None,
            },
            # Phase C4 — frozen validation-selected threshold and
            # validation-fitted calibrator; serving reads these, never
            # recomputes/refits.
            "threshold": self.threshold,
            "calibrator": self.calibrator,
        }
        joblib.dump(metadata, paths["metadata"])

        write_checksums(
            paths["checksums"], {"weights": paths["weights"], "metadata": paths["metadata"]}
        )

        logger.info(f"TFT model saved to {paths['weights']} (+ metadata, checksums)")

    @classmethod
    def load(cls, path: str, config: Optional[Dict] = None) -> "TFTTrainer":
        """Load model from a checkpoint saved by `save()` (Phase D6).

        Verifies the sha256 checksum manifest before deserializing anything
        — a corrupted or tampered artifact must never reach `joblib.load()`
        or `torch.load()`. Raises (does not fall back) on a failed check.

        The weights file contains nothing but the model's state_dict, so it
        is loaded with `weights_only=True` — torch's strict unpickler that
        refuses to construct anything but tensors and a few builtin
        containers. (torch 2.2.0, the version this project pins, still
        defaults `weights_only` to False; passing True explicitly here is
        the actual security boundary, not reliance on a default.)
        """
        base_path = Path(path)
        paths = _artifact_paths(base_path)

        verify_checksums(
            paths["checksums"], {"weights": paths["weights"], "metadata": paths["metadata"]}
        )

        metadata = joblib.load(paths["metadata"])
        cfg = config or metadata["config"]

        trainer = cls(cfg)
        # Phase D7: force CPU regardless of the device the checkpoint was
        # trained on — the serving host is not guaranteed to have a GPU.
        trainer.device = torch.device(resolve_device("auto", force_cpu=True))
        model_cfg = metadata["model_config"]

        trainer.build_model(
            num_numeric_features=model_cfg["num_numeric_features"],
            num_static_features=model_cfg["num_static_features"],
            static_categorical_indices=model_cfg.get("static_categorical_indices") or None,
            static_cardinalities=model_cfg.get("static_cardinalities") or None,
        )
        state_dict = torch.load(paths["weights"], map_location="cpu", weights_only=True)
        trainer.model.load_state_dict(state_dict)
        trainer.model.to(trainer.device)
        trainer._training_history = metadata.get("training_history", [])

        # Restore sequence builder metadata
        sb_data = metadata.get("sequence_builder", {})
        if sb_data.get("feature_dim", 0) > 0:
            trainer.sequence_builder = SequenceBuilder(
                sequence_length=cfg["data"].get("sequence_length", 10),
            )
            trainer.sequence_builder._numeric_features = sb_data.get("numeric_features", [])
            trainer.sequence_builder._static_features = sb_data.get("static_features", [])
            trainer.sequence_builder._feature_dim = sb_data.get("feature_dim", 0)
            trainer.sequence_builder._static_dim = sb_data.get("static_dim", 0)
            # Phase B4/B5 — restore the fitted scaler and cardinalities so
            # inference transforms exactly as training did, never refitting.
            trainer.sequence_builder.scaler = sb_data.get("scaler")
            trainer.sequence_builder._static_cardinalities = sb_data.get("static_cardinalities_by_name")

        # .get() — older checkpoints saved before Phase C4 won't have these keys.
        trainer.threshold = metadata.get("threshold")
        trainer.calibrator = metadata.get("calibrator")
        if trainer.threshold is None or trainer.calibrator is None:
            logger.warning(
                "Loaded checkpoint has no frozen threshold/calibrator — "
                "pre-Phase-C4 artifact. predict()/predict_proba_calibrated() will raise."
            )

        logger.info(f"TFT model loaded from {paths['weights']}")
        return trainer


# ── Main entry point ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TFT model for fraud detection.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    args = parser.parse_args()

    settings = load_settings(args.config)
    config = settings.model_dump()
    seed = set_seed(config.get("project", {}).get("random_seed", 42))

    # Apply CLI overrides
    if args.epochs is not None:
        config["model"]["tft"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config["model"]["tft"]["batch_size"] = args.batch_size

    if args.epochs is not None or args.batch_size is not None:
        # Phase D4 (mle-reviewer HIGH finding): re-validate the merged
        # config so `settings` — and therefore the manifest's config_hash
        # — reflects the EFFECTIVE config actually used for this run, not
        # the pre-override config.yaml. Same fix as train_xgb.py's
        # --tuned-params path; see that call site's comment for the full
        # rationale.
        settings = Settings.model_validate(config)

    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])

    # Load data
    logger.info("Loading parquet data...")
    X_train = pd.read_parquet(processed_dir / "train_features.parquet")
    y_train = pd.read_parquet(processed_dir / "train_labels.parquet").squeeze()
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze()
    X_test = pd.read_parquet(processed_dir / "test_features.parquet")
    y_test = pd.read_parquet(processed_dir / "test_labels.parquet").squeeze()

    logger.info(f"Data loaded: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")

    # Determine feature dimensions by doing a trial sequence build
    seq_builder = SequenceBuilder(
        sequence_length=config["data"].get("sequence_length", 10),
    )
    trial_seq = seq_builder.build_sequences(X_train.head(100), y_train.head(100))
    num_numeric = trial_seq["sequences"].shape[2]
    num_static = trial_seq["static"].shape[1]
    logger.info(f"Feature dimensions: {num_numeric} numeric, {num_static} static")
    del trial_seq
    gc.collect()

    # Phase B5 — cardinalities must reflect the true category range from the
    # full training set, not the 100-row shape-probe sample above (which may
    # not contain every category code).
    static_categorical_indices = seq_builder.static_categorical_indices
    static_cardinalities = None
    if static_categorical_indices:
        static_cardinalities = [
            int(X_train[seq_builder.static_features[i]].max()) + 1
            for i in static_categorical_indices
        ]
        logger.info(
            f"Static categorical embeddings: {len(static_categorical_indices)} columns, "
            f"cardinalities={static_cardinalities}"
        )

    # MLflow setup
    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "fraud_detection"))

    with mlflow.start_run(run_name="tft_sequential_model") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # Build and train
        trainer = TFTTrainer(config)
        trainer.build_model(
            num_numeric,
            num_static,
            static_categorical_indices=static_categorical_indices,
            static_cardinalities=static_cardinalities,
        )

        # Log params
        tft_cfg = config.get("model", {}).get("tft", {})
        mlflow.log_params({
            "model_type": "TFT",
            "hidden_size": tft_cfg.get("hidden_size", 64),
            "attention_heads": tft_cfg.get("attention_head_size", 4),
            "dropout": tft_cfg.get("dropout", 0.1),
            "learning_rate": tft_cfg.get("learning_rate", 0.001),
            "max_epochs": tft_cfg.get("max_epochs", 30),
            "batch_size": tft_cfg.get("batch_size", 128),
            "sequence_length": config["data"].get("sequence_length", 10),
            "num_numeric_features": num_numeric,
            "num_static_features": num_static,
            "total_parameters": trainer.model.count_parameters(),
            "device": str(trainer.device),
            "random_seed": seed,
            "imbalance_sampling_strategy": config.get("imbalance", {}).get("sampling_strategy"),
            "imbalance_loss_function": config.get("imbalance", {}).get("loss_function"),
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
        })

        # Train
        history = trainer.train(X_train, y_train, X_val, y_val)

        # Evaluate on all splits
        logger.info("=" * 60)
        logger.info("EVALUATING TFT ON ALL SPLITS")
        logger.info("=" * 60)

        evaluator = ModelEvaluator()

        # Phase B7: predict_proba(X) takes no labels. Phase B6: history_X
        # supplies real preceding transactions so val/test's first sequences
        # per card aren't padded as if they had no prior context.
        y_prob_train = trainer.predict_proba(X_train)
        y_prob_val = trainer.predict_proba(X_val, history_X=X_train)
        y_prob_test = trainer.predict_proba(
            X_test, history_X=pd.concat([X_train, X_val], axis=0, ignore_index=True)
        )

        pr_auc_train = evaluator.compute_pr_auc(y_train.values, y_prob_train)
        pr_auc_val = evaluator.compute_pr_auc(y_val.values, y_prob_val)
        pr_auc_test = evaluator.compute_pr_auc(y_test.values, y_prob_test)
        roc_auc_test = evaluator.compute_roc_auc(y_test.values, y_prob_test)

        # Phase C2: fit isotonic calibration on VALIDATION only, then apply
        # (never refit) to test — same contract as train_xgb.py. Calibration
        # is monotonic so PR-AUC/ROC-AUC (rank-based) are unaffected; it
        # corrects the probability VALUES the cost-based threshold assumes
        # are calibrated.
        calibrator = evaluator.fit_calibrator(y_val.values, y_prob_val, method="isotonic")
        y_prob_val_cal = evaluator.apply_calibration(calibrator, y_prob_val)
        y_prob_test_cal = evaluator.apply_calibration(calibrator, y_prob_test)

        brier_val_before = evaluator.compute_brier_score(y_val.values, y_prob_val)
        brier_val_after = evaluator.compute_brier_score(y_val.values, y_prob_val_cal)
        brier_test_before = evaluator.compute_brier_score(y_test.values, y_prob_test)
        brier_test_after = evaluator.compute_brier_score(y_test.values, y_prob_test_cal)

        mlflow.log_metric("tft_calibration_brier_score_val_before", brier_val_before)
        mlflow.log_metric("tft_calibration_brier_score_val_after", brier_val_after)
        mlflow.log_metric("tft_calibration_brier_score_test_before", brier_test_before)
        mlflow.log_metric("tft_calibration_brier_score_test_after", brier_test_after)

        logger.info(
            f"Calibration (val):  Brier {brier_val_before:.4f} -> {brier_val_after:.4f}"
        )
        logger.info(
            f"Calibration (test): Brier {brier_test_before:.4f} -> {brier_test_after:.4f}"
        )
        if brier_test_after >= brier_test_before:
            logger.warning(
                "⚠️  Calibration did not improve test Brier score — "
                "the isotonic fit on val may not generalize to test."
            )

        # Overfitting analysis
        overfit_gap = pr_auc_train - pr_auc_test
        logger.info("=" * 60)
        logger.info("TFT OVERFITTING ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"  Train PR-AUC:  {pr_auc_train:.4f}")
        logger.info(f"  Val PR-AUC:    {pr_auc_val:.4f}")
        logger.info(f"  Test PR-AUC:   {pr_auc_test:.4f}")
        logger.info(f"  Gap (train-test): {overfit_gap:.4f}")
        if overfit_gap > 0.15:
            logger.warning("⚠️  SIGNIFICANT OVERFITTING DETECTED (gap > 0.15)")
        elif overfit_gap > 0.10:
            logger.warning("⚠️  Moderate overfitting detected (gap > 0.10)")
        else:
            logger.info("✓ Overfitting gap within acceptable range (< 0.10)")
        logger.info("=" * 60)

        # Optimal threshold
        thresh_cfg = config.get("thresholds", {})
        cost_fn = thresh_cfg.get("cost_fn", 500)
        cost_fp = thresh_cfg.get("cost_fp", 5)
        revenue_tp = thresh_cfg.get("revenue_tp", 480)

        # Phase C1 (found and fixed alongside C2 wiring): threshold must be
        # selected on VALIDATION, never test — this previously read
        # y_test.values directly, the exact leak C1 exists to close.
        # train_xgb.py already had this right; train_tft.py did not.
        optimal_t = evaluator.find_optimal_threshold(y_val.values, y_prob_val, cost_fn, cost_fp, revenue_tp)
        metrics = evaluator.compute_metrics_at_threshold(y_test.values, y_prob_test, optimal_t)

        # Best F1 threshold: also selected on validation, then reported on test.
        from sklearn.metrics import precision_recall_curve
        precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_val.values, y_prob_val)
        f1_scores = np.divide(
            2 * precision_arr * recall_arr,
            precision_arr + recall_arr,
            out=np.zeros_like(precision_arr),
            where=(precision_arr + recall_arr) != 0,
        )
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = thresholds_arr[best_f1_idx] if best_f1_idx < len(thresholds_arr) else 0.5
        if best_f1_idx >= len(thresholds_arr):
            logger.warning(
                "⚠️  F1-optimal index fell on precision_recall_curve's trailing "
                "point (no corresponding threshold) — falling back to 0.5."
            )
        best_f1 = evaluator.compute_metrics_at_threshold(y_test.values, y_prob_test, best_f1_threshold)["f1"]

        # Log metrics
        mlflow.log_metric("tft_pr_auc_train", pr_auc_train)
        mlflow.log_metric("tft_pr_auc_val", pr_auc_val)
        mlflow.log_metric("tft_pr_auc_test", pr_auc_test)
        mlflow.log_metric("tft_roc_auc_test", roc_auc_test)
        mlflow.log_metric("tft_overfit_gap", overfit_gap)
        mlflow.log_metric("tft_optimal_threshold", optimal_t)
        mlflow.log_metric("tft_best_f1", best_f1)
        mlflow.log_metric("tft_best_f1_threshold", best_f1_threshold)
        for k, v in metrics.items():
            mlflow.log_metric(f"tft_test_{k}", v)

        logger.info(f"PR-AUC (test): {pr_auc_test:.4f}")
        logger.info(f"ROC-AUC (test): {roc_auc_test:.4f}")
        logger.info(f"Best F1: {best_f1:.4f} at threshold {best_f1_threshold:.4f}")
        logger.info(f"Optimal cost-based threshold: {optimal_t:.4f}")
        logger.info(f"Metrics at optimal threshold: {metrics}")

        # Generate and log plots
        reports_dir = Path("reports/figures")
        reports_dir.mkdir(parents=True, exist_ok=True)

        pr_path = reports_dir / "tft_pr_curve.png"
        evaluator.plot_pr_curve(y_test.values, y_prob_test, "TFT", str(pr_path))
        mlflow.log_artifact(str(pr_path))

        roc_path = reports_dir / "tft_roc_curve.png"
        evaluator.plot_roc_curve(y_test.values, y_prob_test, "TFT", str(roc_path))
        mlflow.log_artifact(str(roc_path))

        cm_path = reports_dir / "tft_confusion_matrix.png"
        y_pred = (y_prob_test >= optimal_t).astype(int)
        evaluator.plot_confusion_matrix(y_test.values, y_pred, str(cm_path))
        mlflow.log_artifact(str(cm_path))

        val_path = reports_dir / "tft_threshold_value.png"
        evaluator.plot_threshold_vs_business_value(
            y_test.values, y_prob_test, cost_fn, cost_fp, revenue_tp, str(val_path), optimal_threshold=optimal_t
        )
        mlflow.log_artifact(str(val_path))

        # Phase C2: reliability curve, uncalibrated vs. calibrated, on test.
        reliability_path = reports_dir / "tft_reliability_curve.png"
        evaluator.plot_reliability_curve(
            y_test.values, y_prob_test, str(reliability_path), y_prob_calibrated=y_prob_test_cal
        )
        mlflow.log_artifact(str(reliability_path))

        # Phase C4: freeze the validation-selected threshold and the
        # validation-fitted calibrator into the checkpoint before saving.
        trainer.set_threshold(optimal_t)
        trainer.set_calibrator(calibrator)

        # Save model checkpoint
        model_path = config.get("serving", {}).get("tft_model_path", "models/tft_model.ckpt")
        trainer.save(model_path)
        # Phase D6: save() no longer writes the literal `model_path` — it's
        # a stem `_artifact_paths()` derives three real files from (weights,
        # metadata, checksums). Log each of those, not the nonexistent stem.
        for artifact_path in _artifact_paths(Path(model_path)).values():
            mlflow.log_artifact(str(artifact_path))

        # Phase D4: write a manifest (config hash, git SHA, dataset hash,
        # metrics, timestamp) next to the artifact so it can be traced back
        # to the exact run that produced it, without retraining.
        manifest = build_manifest(
            model_type="tft",
            mlflow_run_id=run_id,
            settings=settings,
            dataset_dir=processed_dir,
            dataset_files=[
                "train_features.parquet",
                "train_labels.parquet",
                "val_features.parquet",
                "val_labels.parquet",
                "test_features.parquet",
                "test_labels.parquet",
            ],
            random_seed=seed,
            metrics={
                "pr_auc_train": pr_auc_train,
                "pr_auc_val": pr_auc_val,
                "pr_auc_test": pr_auc_test,
                "roc_auc_test": roc_auc_test,
                "overfit_gap": overfit_gap,
                "optimal_threshold": optimal_t,
                "best_f1": best_f1,
                "best_f1_threshold": best_f1_threshold,
            },
        )
        manifest_path = write_manifest(model_path, manifest)
        mlflow.log_artifact(str(manifest_path))

        # Classification report
        logger.info("\n" + evaluator.generate_classification_report(y_test.values, y_pred))

        logger.info("TFT training complete!")


if __name__ == "__main__":
    main()
