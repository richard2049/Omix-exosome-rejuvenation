# src/rejuvenation.py

"""
rejuvenation.py

Helpers to compute rejuvenation-related metrics from the clock output.

This module provides:
  - delta_age = predicted_age - chronological_age
  - global rejuvenation effect (treated vs control, with bootstrap CI)
  - tissue-level rejuvenation summary (per-tissue effect sizes)
  - simple expression-based tissue effects (mean treated–control shift)

It assumes:
  - Sample-level metadata includes group labels, tissues, and ages
  - Clock predictions are already computed and stored in the metadata
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_delta_age(
    meta: pd.DataFrame,
    pred_age_col: str,
    chrono_age_col: str,
    out_col: str = "delta_age",
) -> pd.DataFrame:
    """
    Add a delta_age column = predicted_age - chronological_age.

    Fails cleanly if required columns are missing or too many values are NaN.
    """
    if pred_age_col not in meta.columns:
        raise ValueError(f"Missing predicted age column: {pred_age_col}")
    if chrono_age_col not in meta.columns:
        raise ValueError(f"Missing chronological age column: {chrono_age_col}")

    df = meta.copy()
    df[out_col] = df[pred_age_col] - df[chrono_age_col]

    # Comprobación básica de NaNs
    valid_frac = df[out_col].notna().mean()
    if valid_frac < 0.5:
        raise ValueError(
            f"Too few valid delta_age values ({valid_frac:.2%}). "
            "Check age prediction and metadata."
        )
    return df


def _group_effect(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    control_labels: List[str],
    treated_labels: List[str],
    min_per_group: int = 4,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> Optional[Dict]:
    """
    Overall or tissue-level rejuvenation effect:
    difference in medians (treated - control) and bootstrap CI.
    Returns None if there is no minimum power.
    """
    if group_col not in df.columns:
        return None

    g = df.dropna(subset=[group_col, value_col])

    is_control = g[group_col].isin(control_labels)
    is_treated = g[group_col].isin(treated_labels)

    ctrl = g.loc[is_control, value_col].values
    trt = g.loc[is_treated, value_col].values

    if len(ctrl) < min_per_group or len(trt) < min_per_group:
        return None

    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(n_bootstrap):
        s_ctrl = rng.choice(ctrl, size=len(ctrl), replace=True)
        s_trt = rng.choice(trt, size=len(trt), replace=True)
        diffs.append(np.median(s_trt) - np.median(s_ctrl))

    diffs = np.array(diffs)
    effect = float(np.median(diffs))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    return {
        "n_ctrl": int(len(ctrl)),
        "n_trt": int(len(trt)),
        "effect_median": effect,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def summarize_rejuvenation_by_tissue(
    meta_with_delta: pd.DataFrame,
    tissue_col: str,
    group_col: str,
    value_col: str = "delta_age",
    control_labels: List[str] = ("Control", "WTC", "Saline"),
    treated_labels: List[str] = ("SRC", "V", "GES"),
    min_per_group: int = 4,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Returns a rejuvenation by tissue table.
    Filters automatically tissues without effect.
    """
    if tissue_col not in meta_with_delta.columns:
        raise ValueError(f"Missing tissue column: {tissue_col}")
    if group_col not in meta_with_delta.columns:
        raise ValueError(f"Missing group column: {group_col}")

    rows = []
    for tissue, sub in meta_with_delta.groupby(tissue_col):
        eff = _group_effect(
            sub,
            group_col=group_col,
            value_col=value_col,
            control_labels=list(control_labels),
            treated_labels=list(treated_labels),
            min_per_group=min_per_group,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        if eff is None:
            continue
        eff_row = {
            "tissue": tissue,
            **eff,
        }
        rows.append(eff_row)

    return pd.DataFrame(rows)


def summarise_global_rejuvenation(
    meta_with_delta: pd.DataFrame,
    group_col: str,
    value_col: str = "delta_age",
    control_labels: List[str] = ("Control", "WTC", "Saline"),
    treated_labels: List[str] = ("SRC", "V", "GES"),
    min_per_group: int = 6,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> Optional[Dict]:
    """
    Global effect size (not stratified by tissue).

    Returns None if the minimum power requirements are not met.
    """
    return _group_effect(
        meta_with_delta,
        group_col=group_col,
        value_col=value_col,
        control_labels=list(control_labels),
        treated_labels=list(treated_labels),
        min_per_group=min_per_group,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

def summarize_tissue_expression_effects(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    tissue_col: str,
    group_col: str,
    control_labels: List[str],
    treated_labels: List[str],
    min_per_group: int = 4,
) -> pd.DataFrame:
    """
    Expression-based tissue signal.

    For each tissue, compute a simple concordance measure:
    mean treated-vs-control effect across genes and overall sign.
    Lightweight Phase 2 version; GSEA and richer analyses are not yet included.
    """
    if tissue_col not in meta.columns or group_col not in meta.columns:
        raise ValueError("Missing tissue or group columns in metadata.")

    # Aseguramos misma ordenación de columnas que en meta
    expr = expr.loc[:, meta["sample_id"].values]

    rows = []
    for tissue, sub_meta in meta.groupby(tissue_col):
        is_ctrl = sub_meta[group_col].isin(control_labels)
        is_trt = sub_meta[group_col].isin(treated_labels)

        if is_ctrl.sum() < min_per_group or is_trt.sum() < min_per_group:
            continue

        cols_ctrl = sub_meta.loc[is_ctrl, "sample_id"].values
        cols_trt = sub_meta.loc[is_trt, "sample_id"].values

        # medias por grupo
        mean_ctrl = expr[cols_ctrl].mean(axis=1)
        mean_trt = expr[cols_trt].mean(axis=1)
        diff = mean_trt - mean_ctrl  # efecto por gen

        rows.append(
            {
                "tissue": tissue,
                "n_ctrl": int(is_ctrl.sum()),
                "n_trt": int(is_trt.sum()),
                "mean_effect": float(diff.mean()),
                "median_effect": float(diff.median()),
            }
        )

    return pd.DataFrame(rows)