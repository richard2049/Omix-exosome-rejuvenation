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
from scipy import stats
from sklearn.linear_model import LinearRegression
from .logging_utils import get_logger

logger = get_logger(__name__)


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
    control_labels: Sequence[str],
    treated_labels: Sequence[str],
    min_per_group: int = 2,
) -> pd.DataFrame:
    """
    Summarize tissue-wise expression effects using group labels.

    For each tissue, compare the mean expression profile between
    control and treated samples and return:

      - tissue
      - n_ctrl
      - n_trt
      - top_genes   (comma-separated string of top-effect genes)
      - mean_effect (mean (treated - control) across all genes)
      - median_effect

    Parameters
    ----------
    expr : DataFrame
        Gene expression matrix (genes x samples).
    meta : DataFrame
        Sample metadata. Must contain tissue_col, group_col and sample_id.
    tissue_col : str
        Column name for tissue/organ.
    group_col : str
        Column name for experimental group (e.g. Y_C, O_C, O_GES...).
    control_labels : list-like
        Labels considered as control.
    treated_labels : list-like
        Labels considered as treated.
    min_per_group : int
        Minimum number of samples per group within a tissue to compute effects.

    Returns
    -------
    DataFrame
        One row per tissue with effect size summary.
    """

    required_cols = {tissue_col, group_col, "sample_id"}
    missing = required_cols - set(meta.columns)
    if missing:
        logger.warning(
            "summarize_tissue_expression_effects: missing required "
            "metadata columns: %s. Returning empty DataFrame.",
            ", ".join(sorted(missing)),
        )
        return pd.DataFrame(
            columns=["tissue", "n_ctrl", "n_trt", "top_genes", "mean_effect", "median_effect"]
        )

    # Drop rows without tissue / group
    meta = meta.copy()
    meta = meta.loc[meta[tissue_col].notna() & meta[group_col].notna()]
    if meta.empty:
        logger.warning("summarize_tissue_expression_effects: no rows with both tissue and group. Returning empty.")
        return pd.DataFrame(
            columns=["tissue", "n_ctrl", "n_trt", "top_genes", "mean_effect", "median_effect"]
        )

    # Keep only control + treated labels
    valid_labels = list(control_labels) + list(treated_labels)
    meta = meta.loc[meta[group_col].isin(valid_labels)]
    if meta.empty:
        logger.warning(
            "summarize_tissue_expression_effects: no rows with group in %s. Returning empty.",
            valid_labels,
        )
        return pd.DataFrame(
            columns=["tissue", "n_ctrl", "n_trt", "top_genes", "mean_effect", "median_effect"]
        )

    # Use sample_id as index to align with expr columns
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta = meta.set_index("sample_id")

    # Align expression columns to metadata
    common_ids = expr.columns.intersection(meta.index)
    if len(common_ids) < (2 * min_per_group):
        logger.warning(
            "summarize_tissue_expression_effects: only %d common samples between expr and meta. Returning empty.",
            len(common_ids),
        )
        return pd.DataFrame(
            columns=["tissue", "n_ctrl", "n_trt", "top_genes", "mean_effect", "median_effect"]
        )

    expr = expr.loc[:, common_ids]
    meta = meta.loc[common_ids]

    rows = []
    for tissue, sub_meta in meta.groupby(tissue_col):
        # Boolean masks in this tissue
        is_ctrl = sub_meta[group_col].isin(control_labels)
        is_trt = sub_meta[group_col].isin(treated_labels)

        n_ctrl = int(is_ctrl.sum())
        n_trt = int(is_trt.sum())

        if n_ctrl < min_per_group or n_trt < min_per_group:
            continue

        # Sample IDs in each group (now they are the index)
        ctrl_cols = sub_meta.index[is_ctrl].tolist()
        trt_cols = sub_meta.index[is_trt].tolist()

        # Make sure they are all in the expression matrix
        ctrl_cols = [c for c in ctrl_cols if c in expr.columns]
        trt_cols = [c for c in trt_cols if c in expr.columns]

        if len(ctrl_cols) < min_per_group or len(trt_cols) < min_per_group:
            continue

        # Mean expression per gene in each group
        expr_ctrl = expr.loc[:, ctrl_cols].mean(axis=1)
        expr_trt = expr.loc[:, trt_cols].mean(axis=1)

        diff = expr_trt - expr_ctrl

        # Top genes by absolute effect
        top_genes = diff.abs().sort_values(ascending=False).head(20).index.tolist()
        top_genes_str = ",".join(map(str, top_genes))

        rows.append(
            {
                "tissue": tissue,
                "n_ctrl": len(ctrl_cols),
                "n_trt": len(trt_cols),
                "top_genes": top_genes_str,
                "mean_effect": float(diff.mean()),
                "median_effect": float(diff.median()),
            }
        )

    if not rows:
        logger.warning(
            "summarize_tissue_expression_effects: no tissues passed the filters (min_per_group=%d). "
            "Returning empty DataFrame.",
            min_per_group,
        )
        return pd.DataFrame(
            columns=["tissue", "n_ctrl", "n_trt", "top_genes", "mean_effect", "median_effect"]
        )

    return pd.DataFrame(rows)

def compute_plasma_biomarkers(
    plasma_expr: pd.DataFrame,
    plasma_meta: pd.DataFrame,
    outcome: pd.Series,
    min_non_nan_frac: float = 0.7,
) -> pd.DataFrame:
    """
    Rank plasma proteins by association with an outcome (e.g. rejuvenation score).

    Parameters
    ----------
    plasma_expr : DataFrame
        Rows = proteins, cols = samples (matching plasma_meta index or sample_id).
    plasma_meta : DataFrame
        Must be index-aligned or contain a column to align with plasma_expr columns.
    outcome : Series
        Numeric phenotype per sample (e.g. rejuvenation score, state index, etc.).
        Index MUST be sample IDs matching plasma_expr columns.
    min_non_nan_frac : float
        Minimum fraction of non-NaN values per protein to include in the analysis.

    Returns
    -------
    DataFrame with columns: protein, spearman_r, pval, qval, abs_r, direction.
    """
    # Ensuring alignment
    common = plasma_expr.columns.intersection(outcome.index)
    if len(common) < 3:
        raise ValueError(f"Too few samples with both plasma and outcome: {len(common)}")

    expr = plasma_expr.loc[:, common]
    y = outcome.loc[common].astype(float)

    # Filtered by missingness
    non_nan_frac = expr.notna().mean(axis=1)
    expr = expr.loc[non_nan_frac >= min_non_nan_frac]
    if expr.empty:
        raise ValueError("All plasma features dropped due to missingness.")

    records = []
    for protein, row in expr.iterrows():
        x = row.values.astype(float)
        mask = ~np.isnan(x) & ~np.isnan(y.values)
        if mask.sum() < 3:
            continue
        r, p = stats.spearmanr(x[mask], y.values[mask])
        if np.isnan(r):
            continue
        records.append((protein, r, p))

    if not records:
        return pd.DataFrame(columns=["protein", "spearman_r", "pval", "qval", "abs_r", "direction"])

    df = pd.DataFrame(records, columns=["protein", "spearman_r", "pval"])

    # Simple FDR (Benjamini–Hochberg)
    df = df.sort_values("pval").reset_index(drop=True)
    m = len(df)
    df["qval"] = df["pval"] * m / (df.index + 1)
    df["qval"] = df["qval"].clip(upper=1.0)

    df["abs_r"] = df["spearman_r"].abs()
    df["direction"] = np.where(df["spearman_r"] > 0, "pro-aging", "pro-rejuvenation")

    return df

def simple_mediation_bootstrap(
    df: pd.DataFrame,
    x_col: str,
    m_col: str,
    y_col: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Simple (non-parametric) mediation bootstrap:
    X -> M -> Y, con Y ajustado también por X.

    Devuelve:
      - total_effect
      - direct_effect
      - indirect_effect
      - bootstrap CIs (95%) de cada uno
    """
    df = df[[x_col, m_col, y_col]].dropna().copy()
    if df.shape[0] < 10:
        logger.warning(
            "simple_mediation_bootstrap: too few samples (n=%d). Returning NaNs.",
            df.shape[0],
        )
        return {
            "n_samples": int(df.shape[0]),
            "total_effect": np.nan,
            "direct_effect": np.nan,
            "indirect_effect": np.nan,
            "total_ci_low": np.nan,
            "total_ci_high": np.nan,
            "direct_ci_low": np.nan,
            "direct_ci_high": np.nan,
            "indirect_ci_low": np.nan,
            "indirect_ci_high": np.nan,
        }

    rng = np.random.default_rng(seed)

    X = df[[x_col]].to_numpy(dtype=float)
    M = df[[m_col]].to_numpy(dtype=float)
    Y = df[[y_col]].to_numpy(dtype=float)

    # a) Efecto total: Y ~ X
    reg_total = LinearRegression().fit(X, Y)
    total_effect = float(reg_total.coef_[0, 0])

    # b) Efecto directo y mediado:
    #    1) M ~ X  -> coef a
    reg_a = LinearRegression().fit(X, M)
    a = float(reg_a.coef_[0, 0])

    #    2) Y ~ X + M -> coef b (M), coef c' (X)
    XM = np.concatenate([X, M], axis=1)
    reg_b = LinearRegression().fit(XM, Y)
    b = float(reg_b.coef_[0, 1])   # coef de M
    direct_effect = float(reg_b.coef_[0, 0])  # coef de X
    indirect_effect = a * b

    total_samples = []
    direct_samples = []
    indirect_samples = []

    for _ in range(n_boot):
        idx = rng.integers(0, df.shape[0], size=df.shape[0])
        Xb = X[idx]
        Mb = M[idx]
        Yb = Y[idx]

        reg_total_b = LinearRegression().fit(Xb, Yb)
        total_b = float(reg_total_b.coef_[0, 0])

        reg_a_b = LinearRegression().fit(Xb, Mb)
        a_b = float(reg_a_b.coef_[0, 0])

        XM_b = np.concatenate([Xb, Mb], axis=1)
        reg_b_b = LinearRegression().fit(XM_b, Yb)
        b_b = float(reg_b_b.coef_[0, 1])
        direct_b = float(reg_b_b.coef_[0, 0])
        indirect_b = a_b * b_b

        total_samples.append(total_b)
        direct_samples.append(direct_b)
        indirect_samples.append(indirect_b)

    def ci(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return float(lo), float(hi)

    t_lo, t_hi = ci(total_samples)
    d_lo, d_hi = ci(direct_samples)
    i_lo, i_hi = ci(indirect_samples)

    out = {
        "n_samples": int(df.shape[0]),
        "total_effect": total_effect,
        "direct_effect": direct_effect,
        "indirect_effect": indirect_effect,
        "total_ci_low": t_lo,
        "total_ci_high": t_hi,
        "direct_ci_low": d_lo,
        "direct_ci_high": d_hi,
        "indirect_ci_low": i_lo,
        "indirect_ci_high": i_hi,
    }
    logger.info("Mediation bootstrap done: %s", out)
    return out

def run_simple_causal_model(
    prim_meta: pd.DataFrame,
    prim_plasma_meta: pd.DataFrame,
    cfg,
) -> Optional[pd.DataFrame]:
    """
    Phase 4 helper: build a simple causal model:
    group_binary -> plasma_state_score -> rejuvenation_score
    on matched animals.
    """
    if "animal_id" not in prim_meta.columns or "animal_id" not in prim_plasma_meta.columns:
        logger.warning("run_simple_causal_model: animal_id missing; cannot align plasma and tissue.")
        return None

    if "plasma_state_score" not in prim_plasma_meta.columns:
        logger.warning("run_simple_causal_model: plasma_state_score missing; skipping mediation.")
        return None

    if "rejuvenation_score" not in prim_meta.columns:
        logger.warning("run_simple_causal_model: rejuvenation_score missing; skipping mediation.")
        return None

    meta = prim_meta.copy()
    if "group_binary" not in meta.columns:
        meta["group_binary"] = (meta["group"] == cfg.primate_treated_label).astype(int)

    merged = meta.merge(
        prim_plasma_meta[["animal_id", "plasma_state_score"]],
        on="animal_id",
        how="inner",
    )
    merged = merged.dropna(subset=["group_binary", "plasma_state_score", "rejuvenation_score"])

    if merged.shape[0] < cfg.min_samples_for_mediation:
        logger.warning(
            "run_simple_causal_model: too few matched samples for mediation (n=%d).",
            merged.shape[0],
        )
        return None

    med = simple_mediation_bootstrap(
        merged,
        x_col="group_binary",
        m_col="plasma_state_score",
        y_col="rejuvenation_score",
        n_boot=cfg.mediation_bootstrap,
        seed=cfg.random_seed,
    )

    return pd.DataFrame([med])
