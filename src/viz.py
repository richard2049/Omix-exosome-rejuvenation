"""
viz.py

Plotting helpers for the pipeline, including tissue-level concordance
between species and ranking of plasma biomarkers, producing simple
PNG figures for downstream inspection.
"""

from __future__ import annotations

import ast
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
    title: str = "Top plasma biomarker candidates",
    requires_spearman: bool = False,
    fallback_metric: str = "variance",  # "variance" | "abs_mean" | "abs_diff"
) -> None:
    if plasma_biomarkers is None or plasma_biomarkers.empty:
        logger.warning("Empty plasma biomarker table; skipping plot.")
        return

    df = plasma_biomarkers.copy()

    value_col = None
    # Current contract: prespecified treatment contrasts on log2 abundance.
    if "log2_fold_change" in df.columns:
        if "is_primary" in df.columns:
            primary = df["is_primary"].astype("string").str.lower().isin(["true", "1"])
            if primary.any():
                df = df.loc[primary].copy()
        df["abs_log2_fold_change"] = pd.to_numeric(
            df["log2_fold_change"], errors="coerce"
        ).abs()
        score_col = "abs_log2_fold_change"
        value_col = "log2_fold_change"
        xlabel = "Adjusted log2 fold-change (GES - vehicle)"
        title = "Treatment-associated plasma protein signals"
    # Legacy contract retained only so historical outputs remain renderable.
    elif "spearman_r" in df.columns:
        if "abs_r" not in df.columns:
            df["abs_r"] = pd.to_numeric(df["spearman_r"], errors="coerce").abs()
        score_col = "abs_r"
        value_col = score_col
        xlabel = "|Spearman r|"
    else:
        if requires_spearman:
            logger.warning(
                "plot_plasma_biomarker_ranking: 'spearman_r' missing and requires_spearman=True; skipping plot."
            )
            return
        # Case 2: fallback ranking
        logger.info("plot_plasma_biomarker_ranking: 'spearman_r' missing; using fallback=%s", fallback_metric)

        # Heuristics: figure out a numeric score column
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if fallback_metric == "variance" and "variance" in df.columns:
            score_col = "variance"
            xlabel = "Variance"
        elif fallback_metric == "abs_mean" and "mean" in df.columns:
            df["abs_mean"] = df["mean"].abs()
            score_col = "abs_mean"
            xlabel = "|Mean|"
        elif fallback_metric == "abs_diff" and {"mean_treated", "mean_control"}.issubset(df.columns):
            df["abs_diff"] = (df["mean_treated"] - df["mean_control"]).abs()
            score_col = "abs_diff"
            xlabel = "|Delta mean|"
        elif numeric_cols:
            # last resort: use the first numeric col
            score_col = numeric_cols[0]
            xlabel = score_col
        else:
            logger.warning("No numeric columns available for fallback ranking; skipping plot.")
            return

    # Label column fallback
    if value_col is None:
        value_col = score_col
    label_col = next(
        (col for col in ["gene_name", "protein", "protein_accession", "Gene name"] if col in df.columns),
        None,
    )
    if label_col is None:
        logger.warning("No label column ('protein'/'Gene name') found; skipping plot.")
        return

    df = df.dropna(subset=[score_col])
    if df.empty:
        logger.warning("No usable scores for plasma ranking plot after dropna; skipping.")
        return

    df = df.sort_values(score_col, ascending=True).tail(top_n)

    plt.figure()
    labels = df[label_col].astype("string").fillna("").str.strip()
    if "protein_accession" in df.columns:
        accessions = df["protein_accession"].astype("string").fillna("").str.strip()
        labels = labels.where(labels.ne(""), accessions)
    plt.barh(labels.astype(str), pd.to_numeric(df[value_col], errors="coerce"))
    if value_col == "log2_fold_change":
        plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(xlabel)
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

    # Reference 1:1 diagonal
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
        Convention: positive = younger-like than chronological. This is the
        sign inverse of ``delta_age``.
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

    # Set tick labels separately to support both older and current Matplotlib.
    ax.boxplot(data_per_group, showfliers=True)

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
    ax.set_ylabel("Rejuvenation score (years; positive = younger-like)")

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
    title: str = "Linked mediation estimates (diagnostic)",
) -> None:
    """
    Plot total, direct, and indirect mediation estimates.

    The current schema uses ``Total``, ``ADE``, and ``ACME``. Historical
    ``*_effect`` columns remain accepted so older result tables can still be
    inspected. Bootstrap confidence intervals are shown when available.
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

    metric_specs = [
        ("Total", "total_effect", "Total_CI", "total_ci_low", "total_ci_high", "Total effect"),
        ("ADE", "direct_effect", "ADE_CI", "direct_ci_low", "direct_ci_high", "Direct effect (ADE)"),
        ("ACME", "indirect_effect", "ACME_CI", "indirect_ci_low", "indirect_ci_high", "Indirect effect (ACME)"),
    ]
    missing_metrics = [
        canonical
        for canonical, legacy, *_ in metric_specs
        if canonical not in df.columns and legacy not in df.columns
    ]
    if missing_metrics:
        logger.warning(
            "plot_mediation_effects_bar: missing mediation metrics %s; accepted schemas are "
            "Total/ADE/ACME or total_effect/direct_effect/indirect_effect. Skipping.",
            missing_metrics,
        )
        return

    row = df.iloc[0]

    def parse_ci(value: object) -> tuple[float, float]:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return np.nan, np.nan
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2:
            low = pd.to_numeric(pd.Series([value[0]]), errors="coerce").iloc[0]
            high = pd.to_numeric(pd.Series([value[1]]), errors="coerce").iloc[0]
            return float(low), float(high)
        return np.nan, np.nan

    effects = []
    intervals = []
    labels = []
    for canonical, legacy, ci_col, low_col, high_col, label in metric_specs:
        effect_col = canonical if canonical in df.columns else legacy
        effect = pd.to_numeric(pd.Series([row.get(effect_col)]), errors="coerce").iloc[0]
        if ci_col in df.columns:
            low, high = parse_ci(row.get(ci_col))
        elif low_col in df.columns and high_col in df.columns:
            low = pd.to_numeric(pd.Series([row.get(low_col)]), errors="coerce").iloc[0]
            high = pd.to_numeric(pd.Series([row.get(high_col)]), errors="coerce").iloc[0]
            low, high = float(low), float(high)
        else:
            low, high = np.nan, np.nan
        effects.append(float(effect))
        intervals.append((low, high))
        labels.append(label)

    if not np.isfinite(effects).all():
        logger.warning("plot_mediation_effects_bar: non-finite mediation estimates; skipping.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(effects))
    ax.bar(x, effects, color=["#496A81", "#2A9D8F", "#E9A23B"], width=0.68)
    ax.axhline(0.0, color="#343A40", linestyle="--", linewidth=1)

    intervals_available = 0
    intervals_crossing_zero = 0
    for idx, (effect, (low, high)) in enumerate(zip(effects, intervals)):
        if np.isfinite(low) and np.isfinite(high) and low <= effect <= high:
            ax.errorbar(
                idx,
                effect,
                yerr=[[effect - low], [high - effect]],
                fmt="none",
                ecolor="#20252B",
                elinewidth=1.2,
                capsize=4,
            )
            intervals_available += 1
            intervals_crossing_zero += int(low <= 0 <= high)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Estimated effect on rejuvenation score")
    fig.suptitle(title, x=0.12, y=0.98, ha="left", fontsize=14)
    if intervals_available:
        fig.text(
            0.12,
            0.925,
            f"{intervals_crossing_zero}/{intervals_available} bootstrap 95% CIs cross zero",
            fontsize=9,
            color="#555B61",
        )
    fig.text(
        0.12,
        0.025,
        "Diagnostic association estimates; they do not establish exosome causality.",
        fontsize=8.5,
        color="#555B61",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved mediation effects bar plot: %s", out_path)
