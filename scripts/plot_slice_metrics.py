"""
scripts/plot_slice_metrics.py

Phase C5: renders `reports/slice_metrics.csv` (produced by
`scripts/run_slice_metrics.py`) into one grouped bar chart per slice
dimension — precision vs. recall at the frozen validation threshold, with
each slice's row count in the x-axis label so a thin/unreliable slice is
visible at a glance rather than reading as equally trustworthy.

Colors follow the ecc dataviz reference palette (categorical slots 1/2:
blue for recall, orange for precision — validated colorblind-safe pair,
see dataviz skill references/palette.md).

Usage:
    conda run -n fraudx python scripts/plot_slice_metrics.py
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")

# ecc dataviz reference palette — categorical slots 1 (blue) & 2 (orange).
COLOR_RECALL = "#2a78d6"
COLOR_PRECISION = "#eb6834"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_TEXT_MUTED = "#898781"


def plot_dimension(table: pd.DataFrame, dimension: str, save_path: Path) -> None:
    """Grouped bar chart: precision vs. recall per slice, x-labels annotated
    with `n=<count>` so a reader sees sample size without a second table."""
    table = table.sort_values("count", ascending=False).reset_index(drop=True)
    slices = table["slice"].astype(str).tolist()
    x_labels = [f"{s}\n(n={c:,})" for s, c in zip(slices, table["count"])]
    x = np.arange(len(slices))
    bar_width = 0.32

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(slices)), 5))
    ax.bar(x - bar_width / 2, table["recall"], bar_width, label="Recall", color=COLOR_RECALL)
    ax.bar(x + bar_width / 2, table["precision"], bar_width, label="Precision", color=COLOR_PRECISION)

    # Unreliable slices (< min_slice_size rows) get a muted asterisk cue
    # rather than being silently indistinguishable from well-populated ones.
    for i, reliable in enumerate(table["reliable"]):
        if not reliable:
            ax.text(x[i], 0.02, "*thin slice", ha="center", va="bottom",
                     fontsize=8, color=COLOR_TEXT_MUTED, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, color=COLOR_TEXT_SECONDARY)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", color=COLOR_TEXT_SECONDARY)
    ax.set_title(f"Validation precision & recall by {dimension}", color="#0b0b0b")
    ax.grid(axis="y", color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_TEXT_MUTED)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def main() -> None:
    csv_path = Path("reports/slice_metrics.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found — run scripts/run_slice_metrics.py first."
        )
    combined = pd.read_csv(csv_path)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for dimension, table in combined.groupby("dimension"):
        save_path = FIGURES_DIR / f"slice_{dimension.lower()}.png"
        plot_dimension(table, dimension, save_path)


if __name__ == "__main__":
    main()
