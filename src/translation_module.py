"""
translation_module.py

Higher-level translational routines that summarize primate and mouse
results into quantifiable insights, including candidate biomarkers, targets
and qualitative interpretation of exosome-related effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


# -----------------------------
# Multiple-testing
# -----------------------------

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : np.ndarray

    Returns
    -------
    np.ndarray
        q-values
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)

    q_sorted = pvals[order] * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    out = np.empty(n, dtype=float)
    out[order] = q_sorted
    return out


def safe_ttest(a: np.ndarray, b: np.ndarray) -> float:
    """
    Two-sample Welch t-test p-value.

    Uses scipy if available; otherwise returns 1.0 as conservative fallback.
    """
    try:
        from scipy.stats import ttest_ind
        res = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(res.pvalue)
    except Exception:
        return 1.0


# -----------------------------
# Lightweight DE
# -----------------------------

def differential_expression_ttest(
    expr_log: pd.DataFrame,
    meta: pd.DataFrame,
    treated_label: str,
    control_labels: List[str],
    group_col: str = "group",
    tissue: Optional[str] = None,
    tissue_col: str = "tissue",
    min_samples_per_group: int = 2
) -> pd.DataFrame:
    """
    Lightweight DE for processed OMIX matrices.

    Intended for mechanistic prioritization and cross-condition comparisons.
    Not a replacement for raw-based DE pipelines.

    Returns
    -------
    pd.DataFrame
        gene, logFC, pval, qval, n_treated, n_control, tissue
    """
    if group_col not in meta.columns:
        return pd.DataFrame()

    dfm = meta.copy().set_index("sample_id")

    if tissue is not None and tissue_col in dfm.columns:
        dfm = dfm[dfm[tissue_col] == tissue]

    samples = [s for s in expr_log.columns if s in dfm.index]
    if not samples:
        return pd.DataFrame()

    dfm = dfm.loc[samples]
    expr = expr_log.loc[:, samples]

    treated_samples = dfm[dfm[group_col] == treated_label].index.tolist()
    control_samples = dfm[dfm[group_col].isin(control_labels)].index.tolist()

    if len(treated_samples) < min_samples_per_group or len(control_samples) < min_samples_per_group:
        return pd.DataFrame()

    t_mat = expr[treated_samples].values
    c_mat = expr[control_samples].values

    mean_t = np.nanmean(t_mat, axis=1)
    mean_c = np.nanmean(c_mat, axis=1)

    logfc = mean_t - mean_c

    pvals = np.array([safe_ttest(t_mat[i, :], c_mat[i, :]) for i in range(expr.shape[0])])
    qvals = benjamini_hochberg(pvals)

    out = pd.DataFrame({
        "gene": expr.index.astype(str),
        "logFC": logfc,
        "pval": pvals,
        "qval": qvals,
        "n_treated": len(treated_samples),
        "n_control": len(control_samples),
        "tissue": tissue if tissue is not None else "ALL"
    }).sort_values("qval")

    return out


def get_top_genes(
    de: pd.DataFrame,
    n: int = 200,
    direction: str = "up",
    qval_max: float = 0.2
) -> Set[str]:
    """
    Select top genes given a lightweight DE table.

    direction:
      - "up" | "down" | "any"
    """
    if de.empty:
        return set()

    d = de.copy()
    if "qval" in d.columns:
        d = d[d["qval"] <= qval_max]

    if d.empty:
        return set()

    if direction == "up":
        d = d[d["logFC"] > 0].sort_values("logFC", ascending=False)
    elif direction == "down":
        d = d[d["logFC"] < 0].sort_values("logFC", ascending=True)
    else:
        d["absFC"] = d["logFC"].abs()
        d = d.sort_values("absFC", ascending=False)

    return set(d["gene"].head(n).tolist())


# -----------------------------
# GMT + ORA
# -----------------------------

def load_gmt(path: Path) -> Dict[str, Set[str]]:
    """
    Load gene sets from GMT.

    Format:
      set_name<TAB>description<TAB>gene1<TAB>gene2...

    Returns
    -------
    dict
    """
    gene_sets: Dict[str, Set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = set(parts[2:])
            gene_sets[name] = genes
    return gene_sets


def ora_hypergeom(
    query_genes: Set[str],
    background_genes: Set[str],
    gene_sets: Dict[str, Set[str]],
    min_overlap: int = 3
) -> pd.DataFrame:
    """
    ORA via hypergeometric test.

    Uses scipy if available; otherwise returns empty.

    Returns
    -------
    pd.DataFrame
        pathway, overlap, set_size, bg_size, pval, qval
    """
    if not query_genes or not background_genes or not gene_sets:
        return pd.DataFrame()

    try:
        from scipy.stats import hypergeom
    except Exception:
        logger.warning("scipy not available: ORA skipped.")
        return pd.DataFrame()

    query = set(g for g in query_genes if g in background_genes)

    results = []
    M = len(background_genes)
    n = len(query)

    for pname, pgenes in gene_sets.items():
        pset = set(g for g in pgenes if g in background_genes)
        K = len(pset)
        if K == 0:
            continue

        k = len(query.intersection(pset))
        if k < min_overlap:
            continue

        pval = float(hypergeom.sf(k - 1, M, K, n))
        results.append((pname, k, K, M, pval))

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results, columns=["pathway", "overlap", "set_size", "bg_size", "pval"])
    out["qval"] = benjamini_hochberg(out["pval"].values)
    return out.sort_values("qval")


# -----------------------------
# Plasma biomarkers (response-oriented)
# -----------------------------

def rank_plasma_biomarkers(
    plasma_mat: pd.DataFrame,
    plasma_meta: pd.DataFrame,
    outcome_meta: pd.DataFrame,
    outcome_col: str,
    treated_label: str,
    control_labels: List[str],
    group_col: str = "group",
    sample_id_col: str = "sample_id",
    top_n: int = 50
) -> pd.DataFrame:
    """
    Rank plasma proteins as candidate systemic biomarkers.

    Evidence:
      1) Spearman correlation with outcome
      2) Treated-control delta

    Returns
    -------
    pd.DataFrame
    """
    pm = plasma_meta.copy()
    om = outcome_meta.copy()

    pm[sample_id_col] = pm[sample_id_col].astype(str)
    om[sample_id_col] = om[sample_id_col].astype(str)

    common = sorted(
        set(plasma_mat.columns).intersection(pm[sample_id_col]).intersection(om[sample_id_col])
    )
    if len(common) < 3:
        return pd.DataFrame()

    plasma = plasma_mat.loc[:, common]
    pm = pm.set_index(sample_id_col).loc[common].reset_index()
    om = om.set_index(sample_id_col).loc[common].reset_index()

    merged = pm[[sample_id_col, group_col]].merge(
        om[[sample_id_col, outcome_col]],
        on=sample_id_col,
        how="inner"
    )
    y = merged[outcome_col].astype(float).values

    try:
        from scipy.stats import spearmanr
        use_scipy = True
    except Exception:
        use_scipy = False

    treated_ids = merged[merged[group_col] == treated_label][sample_id_col].tolist()
    control_ids = merged[merged[group_col].isin(control_labels)][sample_id_col].tolist()

    results = []
    for protein in plasma.index:
        x = plasma.loc[protein, merged[sample_id_col]].astype(float).values

        if use_scipy:
            r, p = spearmanr(x, y, nan_policy="omit")
            rho = float(r)
            pval = float(p)
        else:
            rho = float(pd.Series(x).corr(pd.Series(y), method="pearson"))
            pval = 1.0

        dt = plasma.loc[protein, treated_ids].astype(float).mean() if treated_ids else np.nan
        dc = plasma.loc[protein, control_ids].astype(float).mean() if control_ids else np.nan
        delta = float(dt - dc) if (not np.isnan(dt) and not np.isnan(dc)) else np.nan

        results.append((protein, rho, pval, delta))

    out = pd.DataFrame(results, columns=["protein", "spearman_r", "pval", "delta_treated_control"])
    out["qval"] = benjamini_hochberg(out["pval"].values)
    out["abs_r"] = out["spearman_r"].abs()
    out = out.sort_values(["qval", "abs_r"], ascending=[True, False])

    return out.head(top_n)


# -----------------------------
# High-level container
# -----------------------------

@dataclass
class TranslationalResults:
    shared_genes_by_tissue: pd.DataFrame
    shared_genes_global: pd.DataFrame
    shared_pathways_global: pd.DataFrame
    organ_priority: pd.DataFrame
    plasma_biomarkers: pd.DataFrame


def generate_translational_insights(
    prim_expr_log: pd.DataFrame,
    prim_meta: pd.DataFrame,
    mouse_expr_log: Optional[pd.DataFrame],
    mouse_meta: Optional[pd.DataFrame],
    prim_outcomes: pd.DataFrame,
    outcome_col_prim: str,
    prim_treated_label: str,
    prim_control_labels: List[str],
    mouse_treated_label: str,
    mouse_control_labels: List[str],
    plasma_mat: Optional[pd.DataFrame] = None,
    plasma_meta: Optional[pd.DataFrame] = None,
    gmt_path: Optional[Path] = None,
    top_genes_per_tissue: int = 200
) -> TranslationalResults:
    """
    Generate translational outputs without relying on raw data.

    Outputs
    -------
    TranslationalResults
    """
    # 1) Organ priority from primate outcome magnitude
    organ_priority = []
    if "tissue" in prim_outcomes.columns:
        for tissue, d in prim_outcomes.groupby("tissue"):
            treated = d[d["group"] == prim_treated_label][outcome_col_prim].dropna()
            control = d[d["group"].isin(prim_control_labels)][outcome_col_prim].dropna()

            if treated.empty or control.empty:
                continue

            effect = float(treated.mean() - control.mean())
            organ_priority.append((tissue, effect, len(treated), len(control)))

    organ_priority_df = pd.DataFrame(
        organ_priority, columns=["tissue", "primate_cell_effect", "n_treated", "n_control"]
    ).sort_values("primate_cell_effect") if organ_priority else pd.DataFrame(
        columns=["tissue", "primate_cell_effect", "n_treated", "n_control"]
    )

    # 2) Shared genes per tissue (cells vs exosomes)
    shared_rows = []
    global_shared_up: Set[str] = set()
    global_shared_down: Set[str] = set()

    if mouse_expr_log is not None and mouse_meta is not None:
        if "tissue" in prim_meta.columns and "tissue" in mouse_meta.columns:
            common_tissues = sorted(
                set(prim_meta["tissue"].dropna().astype(str)).intersection(
                    set(mouse_meta["tissue"].dropna().astype(str))
                )
            )
        else:
            common_tissues = []

        for tissue in common_tissues:
            de_prim = differential_expression_ttest(
                prim_expr_log, prim_meta,
                treated_label=prim_treated_label,
                control_labels=prim_control_labels,
                tissue=tissue
            )
            de_mouse = differential_expression_ttest(
                mouse_expr_log, mouse_meta,
                treated_label=mouse_treated_label,
                control_labels=mouse_control_labels,
                tissue=tissue
            )

            up_prim = get_top_genes(de_prim, n=top_genes_per_tissue, direction="up")
            down_prim = get_top_genes(de_prim, n=top_genes_per_tissue, direction="down")
            up_mouse = get_top_genes(de_mouse, n=top_genes_per_tissue, direction="up")
            down_mouse = get_top_genes(de_mouse, n=top_genes_per_tissue, direction="down")

            shared_up = up_prim.intersection(up_mouse)
            shared_down = down_prim.intersection(down_mouse)

            global_shared_up |= shared_up
            global_shared_down |= shared_down

            shared_rows.append({
                "tissue": tissue,
                "shared_up_n": len(shared_up),
                "shared_down_n": len(shared_down),
                "shared_up_genes": ";".join(sorted(shared_up)) if shared_up else "",
                "shared_down_genes": ";".join(sorted(shared_down)) if shared_down else ""
            })

    shared_by_tissue_df = pd.DataFrame(shared_rows)

    shared_global_df = pd.DataFrame([
        {"direction": "up", "n": len(global_shared_up), "genes": ";".join(sorted(global_shared_up))},
        {"direction": "down", "n": len(global_shared_down), "genes": ";".join(sorted(global_shared_down))}
    ])

    # 3) ORA on global shared genes (optional)
    shared_pathways_df = pd.DataFrame()
    if gmt_path is not None and gmt_path.exists():
        try:
            gene_sets = load_gmt(gmt_path)
            background = set(prim_expr_log.index.astype(str))
            query = global_shared_up.union(global_shared_down)

            shared_pathways_df = ora_hypergeom(
                query_genes=query,
                background_genes=background,
                gene_sets=gene_sets,
                min_overlap=3
            )
        except Exception as e:
            logger.warning("ORA failed: %s", str(e))

    # 4) Plasma biomarkers (optional)
    plasma_biomarkers_df = pd.DataFrame()
    if plasma_mat is not None and plasma_meta is not None:
        try:
            plasma_biomarkers_df = rank_plasma_biomarkers(
                plasma_mat=plasma_mat,
                plasma_meta=plasma_meta,
                outcome_meta=prim_outcomes,
                outcome_col=outcome_col_prim,
                treated_label=prim_treated_label,
                control_labels=prim_control_labels
            )
        except Exception as e:
            logger.warning("Plasma biomarker ranking failed: %s", str(e))

    return TranslationalResults(
        shared_genes_by_tissue=shared_by_tissue_df,
        shared_genes_global=shared_global_df,
        shared_pathways_global=shared_pathways_df,
        organ_priority=organ_priority_df,
        plasma_biomarkers=plasma_biomarkers_df
    )