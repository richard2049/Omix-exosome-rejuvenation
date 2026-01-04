"""
exosome_effect.py

Functions to quantify tissue-level treatment effects, derive plasma /
exosome-like state scores, compare bulk–plasma patterns and estimate the
exosome-attributable fraction of rejuvenation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from .logging_utils import get_logger

logger = get_logger(__name__)


def compute_group_effect_by_tissue(
    meta: pd.DataFrame,
    outcome_col: str,
    treated_label: str,
    control_labels: List[str],
    group_col: str = "group",
    tissue_col: str = "tissue"
) -> pd.DataFrame:
    """
    Compute treated - control mean outcome per tissue.

    Returns
    -------
    pd.DataFrame
        Indexed by tissue with mean_effect, n_treated, n_control.
    """
    if outcome_col not in meta.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found.")

    results = []
    df = meta.copy()

    if tissue_col not in df.columns:
        df[tissue_col] = "ALL"

    for tissue, d in df.groupby(tissue_col):
        treated = d[d[group_col] == treated_label][outcome_col].dropna()
        control = d[d[group_col].isin(control_labels)][outcome_col].dropna()

        if treated.empty or control.empty:
            continue

        effect = float(treated.mean() - control.mean())
        results.append((tissue, effect, len(treated), len(control)))

    out = pd.DataFrame(results, columns=[tissue_col, "mean_effect", "n_treated", "n_control"])
    return out.set_index(tissue_col).sort_values("mean_effect") if not out.empty else out


def compare_effect_patterns(effect_a: pd.DataFrame, effect_b: pd.DataFrame) -> Dict:
    """
    Compare tissue-level effect patterns using Spearman + Pearson.

    Returns
    -------
    dict
    """
    if effect_a.empty or effect_b.empty:
        return {"n_common_tissues": 0, "spearman_rho": np.nan, "pearson_r": np.nan}

    common = effect_a.index.intersection(effect_b.index)
    if len(common) < 3:
        return {"n_common_tissues": int(len(common)), "spearman_rho": np.nan, "pearson_r": np.nan}

    a = effect_a.loc[common, "mean_effect"].astype(float)
    b = effect_b.loc[common, "mean_effect"].astype(float)

    return {
        "n_common_tissues": int(len(common)),
        "spearman_rho": float(a.corr(b, method="spearman")),
        "pearson_r": float(a.corr(b, method="pearson"))
    }


def estimate_exosome_fraction(effect_cells: pd.DataFrame, effect_exosomes: pd.DataFrame) -> Dict:
    """
    Rough magnitude-based estimate:
        median(|exo_effect|) / median(|cell_effect|)
    over common tissues.

    Returns
    -------
    dict
    """
    if effect_cells.empty or effect_exosomes.empty:
        return {"n_common_tissues": 0, "cells_median_abs": np.nan, "exo_median_abs": np.nan, "ratio": np.nan}

    common = effect_cells.index.intersection(effect_exosomes.index)
    if len(common) < 3:
        return {"n_common_tissues": int(len(common)), "cells_median_abs": np.nan, "exo_median_abs": np.nan, "ratio": np.nan}

    cells_mag = effect_cells.loc[common, "mean_effect"].abs().median()
    exo_mag = effect_exosomes.loc[common, "mean_effect"].abs().median()
    ratio = float(exo_mag / cells_mag) if cells_mag != 0 else np.nan

    return {
        "n_common_tissues": int(len(common)),
        "cells_median_abs": float(cells_mag),
        "exo_median_abs": float(exo_mag),
        "ratio": ratio
    }


def build_plasma_state_score(
    plasma_matrix: pd.DataFrame,
    n_top_proteins: int = 50,
    score_name: str = "plasma_state_score"
) -> pd.Series:
    """
    Build a systemic plasma state score (PC1 of top variable proteins).

    This can serve as a proxy mediator capturing systemic secretory shifts.
    """
    variances = plasma_matrix.var(axis=1).sort_values(ascending=False)
    top = variances.head(n_top_proteins).index
    sub = plasma_matrix.loc[top].T

    Xz = StandardScaler().fit_transform(sub.values)
    pc1 = PCA(n_components=1).fit_transform(Xz).flatten()

    return pd.Series(pc1, index=sub.index, name=score_name)


def simple_mediation_bootstrap(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    covariates: Optional[List[str]] = None,
    n_boot: int = 2000,
    random_state: int = 42
) -> Dict:
    """
    Simple linear mediation with bootstrap.

    Models:
      1) Y ~ T + C   (total effect c)
      2) M ~ T + C   (a)
      3) Y ~ T + M + C   (direct c', b)

    ACME ~ a*b
    ADE ~ c'
    Prop mediated ~ ACME / total

    Returns
    -------
    dict
    """
    rng = np.random.default_rng(random_state)

    cols = [outcome, treatment, mediator] + (covariates or [])
    d0 = df[cols].dropna().copy()
    if d0.empty:
        raise ValueError("No complete cases for mediation.")

    def fit_ols(y: pd.Series, X: pd.DataFrame):
        Xc = sm.add_constant(X, has_constant="add")
        return sm.OLS(y.astype(float).values, Xc.astype(float).values).fit()

    C = d0[covariates] if covariates else pd.DataFrame(index=d0.index)

    f1 = fit_ols(d0[outcome], pd.concat([d0[[treatment]], C], axis=1))
    c_hat = f1.params[1]

    f2 = fit_ols(d0[mediator], pd.concat([d0[[treatment]], C], axis=1))
    a_hat = f2.params[1]

    f3 = fit_ols(d0[outcome], pd.concat([d0[[treatment, mediator]], C], axis=1))
    c_prime_hat = f3.params[1]
    b_hat = f3.params[2]

    acme_hat = a_hat * b_hat
    ade_hat = c_prime_hat
    total_hat = c_hat
    prop_hat = acme_hat / total_hat if total_hat != 0 else np.nan

    n = len(d0)
    idx = np.arange(n)

    acme_bs, ade_bs, total_bs, prop_bs = [], [], [], []

    for _ in range(n_boot):
        bi = rng.choice(idx, size=n, replace=True)
        d = d0.iloc[bi]
        Cb = d[covariates] if covariates else pd.DataFrame(index=d.index)

        f1b = fit_ols(d[outcome], pd.concat([d[[treatment]], Cb], axis=1))
        f2b = fit_ols(d[mediator], pd.concat([d[[treatment]], Cb], axis=1))
        f3b = fit_ols(d[outcome], pd.concat([d[[treatment, mediator]], Cb], axis=1))

        c_b = f1b.params[1]
        a_b = f2b.params[1]
        c_prime_b = f3b.params[1]
        b_b = f3b.params[2]

        acme_bs.append(a_b * b_b)
        ade_bs.append(c_prime_b)
        total_bs.append(c_b)
        prop_bs.append((a_b * b_b) / c_b if c_b != 0 else np.nan)

    def ci(x):
        return (float(np.nanpercentile(x, 2.5)),
                float(np.nanpercentile(x, 97.5)))

    return {
        "ACME": float(acme_hat), "ACME_CI": ci(acme_bs),
        "ADE": float(ade_hat), "ADE_CI": ci(ade_bs),
        "Total": float(total_hat), "Total_CI": ci(total_bs),
        "PropMediated": float(prop_hat) if not np.isnan(prop_hat) else np.nan,
        "PropMediated_CI": ci(prop_bs),
    }