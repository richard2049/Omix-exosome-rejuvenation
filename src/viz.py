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