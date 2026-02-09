"""
viz.py

Plotting helpers for the pipeline, including tissue-level concordance
between species and ranking of plasma biomarkers, producing simple
PNG figures for downstream inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


def plot_tissue_concordance(
    primate_effect: pd.DataFrame,
    mouse_effect: pd.DataFrame,
    outpath: Path,
    title: str = "Tissue concordance: cells (primate) vs exosomes (mouse)"
) -> None:
    """
    Scatter plot of tissue-level mean effects.

    Expects:
      - primate_effect indexed by tissue with column 'mean_effect'
      - mouse_effect indexed by tissue with column 'mean_effect'
    """
    if primate_effect.empty or mouse_effect.empty:
        logger.warning("Empty effect tables; skipping concordance plot.")
        return

    common = primate_effect.index.intersection(mouse_effect.index)
    if len(common) < 3:
        logger.warning("Not enough common tissues for concordance plot.")
        return

    df = pd.DataFrame({
        "cells_primate": primate_effect.loc[common, "mean_effect"].astype(float),
        "exosomes_mouse": mouse_effect.loc[common, "mean_effect"].astype(float)
    }, index=common).sort_index()

    plt.figure()
    plt.scatter(df["cells_primate"], df["exosomes_mouse"])

    # Add tissue labels (lightweight annotation)
    for tissue, row in df.iterrows():
        plt.text(row["cells_primate"], row["exosomes_mouse"], str(tissue), fontsize=8)

    plt.axhline(0)
    plt.axvline(0)

    plt.xlabel("Mean effect (treated - control) | Primate cells")
    plt.ylabel("Mean effect (treated - control) | Mouse exosomes")
    plt.title(title)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    logger.info("Saved tissue concordance plot: %s", outpath)


def plot_plasma_biomarker_ranking(
    plasma_biomarkers: pd.DataFrame,
    outpath: Path,
    top_n: int = 20,
    title: str = "Top plasma biomarker candidates"
) -> None:
    """
    Horizontal bar plot of top plasma biomarker candidates by |Spearman r|.

    Expects columns:
      - protein
      - spearman_r
      - qval (optional)
    """
    if plasma_biomarkers.empty:
        logger.warning("Empty plasma biomarker table; skipping plot.")
        return

    df = plasma_biomarkers.copy()
    if "abs_r" not in df.columns:
        df["abs_r"] = df["spearman_r"].abs()

    df = df.sort_values("abs_r", ascending=True).tail(top_n)

    plt.figure()
    plt.barh(df["protein"], df["abs_r"])
    plt.xlabel("|Spearman r| with outcome")
    plt.title(title)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    logger.info("Saved plasma biomarker ranking plot: %s", outpath)

def plot_age_scatter(
    meta: pd.DataFrame,
    chrono_col: str,
    pred_col: str,
    out_path: Path,
) -> None:
    """
    Scatter plot of chronological vs predicted age
    (typically using cross-validated predictions).

    Parameters
    ----------
    meta : DataFrame
        Must contain `chrono_col` and `pred_col`.
    chrono_col : str
        Column with chronological age in years.
    pred_col : str
        Column with predicted age (e.g. CV predictions).
    out_path : Path
        Where to save the PNG.
    """
    import matplotlib.pyplot as plt

    df = meta[[chrono_col, pred_col]].dropna().copy()
    if df.empty:
        logger.warning("plot_age_scatter: no data after dropping NaNs.")
        return

    x = df[chrono_col].astype(float).values
    y = df[pred_col].astype(float).values

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6)

    # Refference diagonal 1:1
    xy_min = min(x.min(), y.min())
    xy_max = max(x.max(), y.max())
    plt.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--")

    plt.xlabel("Chronological age (years)")
    plt.ylabel("Predicted age (years, CV)")
    plt.title("Aging clock: chronological vs predicted age")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_rejuvenation_by_group(
    meta: pd.DataFrame,
    group_col: str,
    rejuvenation_col: str,
    out_path: Path,
    tissue_col: Optional[str] = None,
    group_order: Optional[list] = None,
    min_per_group: int = 2,
) -> None:
    """
    Boxplot + jitter of rejuvenation scores by group.

    Parameters
    ----------
    meta : DataFrame
        Must contain `group_col` and `rejuvenation_col`.
    group_col : str
        Column with treatment / group labels (e.g. Y_C, O_C, O_V, SRC, etc.).
    rejuvenation_col : str
        Column with rejuvenation score in "delta years" (or normalized units).
        Convention: negative = biologically younger than chronological.
    out_path : Path
        Where to save the PNG.
    tissue_col : str, optional
        Reserved for future faceting / coloring by tissue. Currently unused.
    group_order : list, optional
        Explicit order of groups to display on the x-axis. If not provided,
        groups are sorted alphabetically.
    min_per_group : int, default 2
        Minimum number of samples required for a group to be plotted.
        Groups with fewer samples are dropped (with a warning).
    """

    # Basic filtering
    if group_col not in meta.columns or rejuvenation_col not in meta.columns:
        logger.warning(
            "plot_rejuvenation_by_group: required columns missing "
            f"({group_col}, {rejuvenation_col})."
        )
        return

    df = meta[[group_col, rejuvenation_col]].dropna().copy()
    if df.empty:
        logger.warning("plot_rejuvenation_by_group: no data after dropping NaNs.")
        return

    # Group summaries (for logging and sanity check)
    summary = (
        df.groupby(group_col)[rejuvenation_col]
        .agg(["count", "median", "mean"])
        .sort_values("median")
    )
    logger.info("Rejuvenation by group summary:\n%s", summary)

    # Drop groups with too few samples
    valid_groups = summary[summary["count"] >= min_per_group].index.tolist()
    if not valid_groups:
        logger.warning(
            "plot_rejuvenation_by_group: all groups have < %d samples; skipping plot.",
            min_per_group,
        )
        return

    # Determine group order
    if group_order is not None:
        # Keep only those in valid_groups and in the requested order
        groups = [g for g in group_order if g in valid_groups]
        # Add any remaining valid groups not listed in group_order
        groups += [g for g in valid_groups if g not in groups]
    else:
        groups = sorted(valid_groups)

    # Prepare data for plotting
    data_per_group = [df.loc[df[group_col] == g, rejuvenation_col].values for g in groups]
    counts_per_group = [len(vals) for vals in data_per_group]

    # Defensive: if everything collapsed to empty
    if all(len(vals) == 0 for vals in data_per_group):
        logger.warning(
            "plot_rejuvenation_by_group: no non-empty groups after filtering; skipping plot."
        )
        return

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Boxplots
    bp = ax.boxplot(
        data_per_group,
        labels=None,  # we will set labels with n later
        showfliers=True,
    )

    # Jitter of individual points
    for i, (g, y) in enumerate(zip(groups, data_per_group), start=1):
        if len(y) == 0:
            continue
        x = np.random.normal(loc=i, scale=0.08, size=len(y))
        ax.scatter(x, y, alpha=0.4)

    # X tick labels with sample sizes: e.g. "O_V (n=8)"
    xtick_labels = [f"{g} (n={n})" for g, n in zip(groups, counts_per_group)]
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(xtick_labels, rotation=30, ha="right")

    ax.set_xlabel(group_col)
    ax.set_ylabel("Rejuvenation score (Δ units; negative = younger)")

    # Horizontal reference line at 0 (no rejuvenation)
    ax.axhline(0.0, linestyle="--", linewidth=1)

    # Make y-limits symmetric around 0 for easier visual comparison
    all_vals = np.concatenate([vals for vals in data_per_group if len(vals) > 0])
    if all_vals.size > 0:
        max_abs = float(np.max(np.abs(all_vals)))
        if max_abs > 0:
            ax.set_ylim(-1.1 * max_abs, 1.1 * max_abs)

    title = "Rejuvenation score by group"
    ax.set_title(title)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    logger.info("Saved rejuvenation-by-group plot: %s", out_path)

def plot_mediation_effects_bar(
    med: pd.DataFrame,
    out_path: Path,
    title: str = "Decomposition of treatment effect (cells vs exosomes)",
) -> None:
    """
    Simple bar plot for mediation results.

    Expects a DataFrame `med` with at least the columns:
      - 'total_effect'
      - 'direct_effect'
      - 'indirect_effect'

    Typically this would be the summary row from simple_mediation_bootstrap.
    """
    if med is None or (isinstance(med, pd.DataFrame) and med.empty):
        logger.warning("plot_mediation_effects_bar: empty mediation results; skipping.")
        return

    # Accept Series, dict or DataFrame
    if isinstance(med, pd.Series):
        df = med.to_frame().T
    elif isinstance(med, dict):
        df = pd.DataFrame([med])
    else:
        df = med.copy()

    required = ["total_effect", "direct_effect", "indirect_effect"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            "plot_mediation_effects_bar: missing columns %s; expected %s. Skipping.",
            missing,
            required,
        )
        return

    row = df.iloc[0]
    effects = [
        float(row["total_effect"]),
        float(row["direct_effect"]),
        float(row["indirect_effect"]),
    ]
    labels = ["Total", "Direct (cells)", "Indirect (exo)"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, effects)
    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.ylabel("Effect size (Δ years or standardized units)")
    plt.title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    logger.info("Saved mediation effects bar plot: %s", out_path)