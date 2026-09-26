from __future__ import annotations

"""Reference-trained, oriented plasma age-state axis utilities.

The axis is learned only from young and vehicle-treated old reference samples,
then applied without refitting to WTC/SRC samples. Its sign is oriented so that
higher values are older-like when vehicle controls have a higher median score
than young controls. This makes PC1 interpretable without turning it into
causal evidence.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _clean_group(value: object) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("O_"):
        return text.split("_", 1)[1]
    if text.endswith("_C") and "_" in text:
        return text.split("_", 1)[0]
    return text


def _mode_text(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return ""
    mode = clean.mode()
    return str(mode.iloc[0]) if not mode.empty else str(clean.iloc[0])


def _result_stub(reason: str, method: str, extra: dict[str, Any] | None = None) -> pd.DataFrame:
    row = {
        "available": False,
        "estimable": False,
        "reason": reason,
        "reason_code": "",
        "missing_author_key": "",
        "n_used": 0,
        "method": method,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "evidence_level": 0,
    }
    row.update(extra or {})
    return pd.DataFrame([row])


def _bootstrap_axis_summary(
    scores: pd.DataFrame,
    *,
    young_groups: set[str],
    old_control_groups: set[str],
    treated_groups: set[str],
    score_col: str,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    young = scores.loc[scores["group_clean"].isin(young_groups), score_col].dropna().to_numpy(float)
    old = scores.loc[scores["group_clean"].isin(old_control_groups), score_col].dropna().to_numpy(float)
    treated = scores.loc[scores["group_clean"].isin(treated_groups), score_col].dropna().to_numpy(float)
    if len(young) < 2 or len(old) < 2 or len(treated) < 2:
        return {}

    rows = []
    for _ in range(int(n_bootstrap)):
        young_bs = rng.choice(young, size=len(young), replace=True)
        old_bs = rng.choice(old, size=len(old), replace=True)
        treated_bs = rng.choice(treated, size=len(treated), replace=True)
        young_m = float(np.median(young_bs))
        old_m = float(np.median(old_bs))
        treated_m = float(np.median(treated_bs))
        gap = old_m - young_m
        rows.append(
            {
                "old_vs_young_gap": gap,
                "treated_vs_old_control": treated_m - old_m,
                "treated_vs_young_control": treated_m - young_m,
                "treated_fraction_of_old_young_gap": (treated_m - young_m) / gap if abs(gap) > 1e-12 else np.nan,
            }
        )
    boot = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for col in boot.columns:
        values = pd.to_numeric(boot[col], errors="coerce").dropna()
        if values.empty:
            continue
        out[f"{col}_ci_low"] = float(np.percentile(values, 2.5))
        out[f"{col}_ci_high"] = float(np.percentile(values, 97.5))
    return out


def _permutation_median_diff_p_value(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    n_permutations: int,
    random_state: int,
) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan
    observed = float(np.median(group_a) - np.median(group_b))
    combined = np.concatenate([group_a, group_b]).astype(float)
    n_a = len(group_a)
    rng = np.random.default_rng(random_state)
    permuted = []
    for _ in range(int(n_permutations)):
        shuffled = rng.permutation(combined)
        permuted.append(float(np.median(shuffled[:n_a]) - np.median(shuffled[n_a:])))
    permuted_arr = np.asarray(permuted, dtype=float)
    n_extreme = int(np.sum(np.abs(permuted_arr) >= abs(observed)))
    return float((n_extreme + 1) / (len(permuted_arr) + 1))


def build_oriented_plasma_aging_axis(
    plasma_expr: pd.DataFrame,
    plasma_meta: pd.DataFrame,
    *,
    feature_annotations: pd.DataFrame | None = None,
    sample_col: str = "sample_id",
    group_col: str = "group",
    young_groups: Sequence[str] = ("Y",),
    old_control_groups: Sequence[str] = ("V",),
    treated_groups: Sequence[str] = ("GES",),
    n_top_proteins: int = 50,
    min_group_samples: int = 2,
    min_non_nan_frac: float = 0.8,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build scores, loadings, and a summary for an oriented plasma PC1 axis."""
    method = "reference_trained_oriented_plasma_pc1_age_axis"
    if plasma_expr.empty or plasma_meta.empty:
        stub = _result_stub("Plasma expression matrix or metadata is empty.", method)
        return stub.copy(), stub.copy(), stub
    if sample_col not in plasma_meta.columns or group_col not in plasma_meta.columns:
        stub = _result_stub("Plasma metadata is missing sample_id or group columns.", method)
        return stub.copy(), stub.copy(), stub

    meta = plasma_meta.copy()
    meta[sample_col] = meta[sample_col].astype(str)
    meta["group_clean"] = meta[group_col].map(_clean_group)
    common = [sample for sample in meta[sample_col].tolist() if sample in plasma_expr.columns]
    if len(common) < 4:
        stub = _result_stub("Too few plasma samples overlap metadata for PC1 orientation.", method)
        return stub.copy(), stub.copy(), stub

    meta = meta.set_index(sample_col).loc[common].reset_index()
    expr = plasma_expr.loc[:, common].apply(pd.to_numeric, errors="coerce")
    finite_values = expr.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0 or np.any(finite_values <= 0):
        stub = _result_stub(
            "Plasma age-state axis requires strictly positive finite abundances for log2.",
            method,
        )
        return meta, pd.DataFrame(), stub
    expr = np.log2(expr)
    young_set = {_clean_group(group) for group in young_groups}
    old_set = {_clean_group(group) for group in old_control_groups}
    treated_set = {_clean_group(group) for group in treated_groups}
    reference_groups = young_set | old_set
    reference_samples = meta.loc[meta["group_clean"].isin(reference_groups), sample_col].tolist()
    if len(reference_samples) < 2 * int(min_group_samples):
        stub = _result_stub(
            "Cannot train plasma PC1: insufficient young and vehicle-reference samples.",
            method,
        )
        return meta, pd.DataFrame(), stub

    reference_expr = expr.loc[:, reference_samples]
    keep = (
        reference_expr.notna().mean(axis=1) >= float(min_non_nan_frac)
    ) & (reference_expr.var(axis=1, skipna=True) > 0)
    expr = expr.loc[keep]
    if expr.shape[0] < 2:
        stub = _result_stub("Too few non-missing variable plasma proteins for PC1.", method)
        return stub.copy(), stub.copy(), stub

    # Public plasma exports can contain repeated or missing protein labels.
    # Select by stable internal row IDs to avoid duplicate-label expansion.
    expr = expr.copy()
    source_feature_ids = pd.Index(expr.index).astype(str)
    feature_ids = [f"plasma_feature_{i:05d}" for i in range(expr.shape[0])]
    expr.index = feature_ids
    source_id_lookup = pd.Series(source_feature_ids.to_numpy(), index=feature_ids)

    annotation_lookup = pd.DataFrame(index=source_feature_ids)
    if feature_annotations is not None and not feature_annotations.empty:
        annotation_lookup = feature_annotations.copy()
        if "feature_id" in annotation_lookup.columns:
            annotation_lookup = annotation_lookup.set_index("feature_id", drop=False)
        annotation_lookup.index = annotation_lookup.index.astype(str)
        if annotation_lookup.index.has_duplicates:
            stub = _result_stub("Plasma feature annotation IDs must be unique.", method)
            return meta, pd.DataFrame(), stub
        annotation_lookup = annotation_lookup.reindex(source_feature_ids)

    n_top = int(min(max(2, int(n_top_proteins)), expr.shape[0]))
    reference_internal = expr.loc[:, reference_samples]
    top_features = (
        reference_internal.var(axis=1, skipna=True)
        .sort_values(ascending=False)
        .head(n_top)
        .index
    )
    top_source_ids = source_id_lookup.loc[top_features].astype(str)
    train_sub = expr.loc[top_features, reference_samples].T
    train_medians = train_sub.median(axis=0)
    train_sub = train_sub.fillna(train_medians)
    all_sub = expr.loc[top_features, common].T.fillna(train_medians)
    if train_sub.isna().any().any() or all_sub.isna().any().any():
        stub = _result_stub(
            "Cannot train plasma PC1: reference-derived imputation left missing values.",
            method,
        )
        return meta, pd.DataFrame(), stub

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_sub.to_numpy(dtype=float))
    x_all = scaler.transform(all_sub.to_numpy(dtype=float))
    pca = PCA(n_components=1, random_state=random_state)
    pca.fit(x_train)
    raw_scores = pca.transform(x_all).flatten()
    raw_loadings = pca.components_[0].astype(float)
    explained = float(pca.explained_variance_ratio_[0])

    score_df = meta.copy()
    score_df["raw_pc1_score"] = raw_scores

    young_raw = score_df.loc[score_df["group_clean"].isin(young_set), "raw_pc1_score"].dropna()
    old_raw = score_df.loc[score_df["group_clean"].isin(old_set), "raw_pc1_score"].dropna()
    treated_raw = score_df.loc[score_df["group_clean"].isin(treated_set), "raw_pc1_score"].dropna()
    if len(young_raw) < min_group_samples or len(old_raw) < min_group_samples:
        reason = "Cannot orient plasma PC1: insufficient young or old-control plasma samples."
        stub = _result_stub(reason, method)
        return score_df, pd.DataFrame(), stub

    young_raw_median = float(young_raw.median())
    old_raw_median = float(old_raw.median())
    orientation_sign = 1.0 if old_raw_median >= young_raw_median else -1.0
    score_df["plasma_age_axis_score"] = score_df["raw_pc1_score"] * orientation_sign
    score_df["axis_orientation"] = "higher_older_like"
    score_df["n_top_proteins"] = n_top
    score_df["pc1_explained_variance_ratio"] = explained
    score_df["analysis_scale"] = "log2_positive_abundance"

    top_annotations = annotation_lookup.reindex(top_source_ids.to_numpy()).copy()
    protein_accessions = (
        top_annotations["protein_accession"].astype("string").fillna("").to_numpy()
        if "protein_accession" in top_annotations.columns
        else top_source_ids.to_numpy(dtype=str)
    )
    gene_names = (
        top_annotations["gene_name"].astype("string").fillna("").to_numpy()
        if "gene_name" in top_annotations.columns
        else np.repeat("", len(top_source_ids))
    )
    protein_labels = [
        str(gene).strip() or str(accession).strip() or str(feature_id)
        for gene, accession, feature_id in zip(gene_names, protein_accessions, top_source_ids)
    ]
    loading_df = pd.DataFrame(
        {
            "feature_id": top_source_ids.to_numpy(dtype=str),
            "protein_accession": protein_accessions,
            "gene_name": gene_names,
            "protein": protein_labels,
            "raw_loading": raw_loadings,
            "loading": raw_loadings * orientation_sign,
            "abs_loading": np.abs(raw_loadings),
            "n_top_proteins": n_top,
            "pc1_explained_variance_ratio": explained,
            "axis_orientation": "higher_older_like",
            "analysis_scale": "log2_positive_abundance",
        }
    ).sort_values("abs_loading", ascending=False).reset_index(drop=True)
    loading_df.insert(0, "loading_rank", np.arange(1, len(loading_df) + 1))

    young = score_df.loc[score_df["group_clean"].isin(young_set), "plasma_age_axis_score"].dropna()
    old = score_df.loc[score_df["group_clean"].isin(old_set), "plasma_age_axis_score"].dropna()
    treated = score_df.loc[score_df["group_clean"].isin(treated_set), "plasma_age_axis_score"].dropna()
    young_median = float(young.median())
    old_median = float(old.median())
    treated_median = float(treated.median()) if len(treated) > 0 else np.nan
    old_young_gap = old_median - young_median
    treated_vs_old = treated_median - old_median if np.isfinite(treated_median) else np.nan
    treated_vs_young = treated_median - young_median if np.isfinite(treated_median) else np.nan
    fraction = treated_vs_young / old_young_gap if abs(old_young_gap) > 1e-12 and np.isfinite(treated_vs_young) else np.nan
    shift_label = (
        "young_like_shift_vs_old_controls"
        if np.isfinite(treated_vs_old) and treated_vs_old < 0
        else ("older_like_or_no_young_shift_vs_old_controls" if np.isfinite(treated_vs_old) else "treated_group_unavailable")
    )

    ci = _bootstrap_axis_summary(
        score_df,
        young_groups=young_set,
        old_control_groups=old_set,
        treated_groups=treated_set,
        score_col="plasma_age_axis_score",
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    old_gap_ci_low = ci.get("old_vs_young_gap_ci_low", np.nan)
    old_gap_ci_high = ci.get("old_vs_young_gap_ci_high", np.nan)
    old_gap_crosses_zero = bool(
        np.isfinite(old_gap_ci_low)
        and np.isfinite(old_gap_ci_high)
        and old_gap_ci_low <= 0 <= old_gap_ci_high
    )
    p_value = _permutation_median_diff_p_value(
        treated.to_numpy(float),
        old.to_numpy(float),
        n_permutations=n_permutations,
        random_state=random_state,
    )
    summary = {
        "n_samples": int(len(score_df)),
        "n_reference_training_samples": int(len(reference_samples)),
        "reference_training_groups": ",".join(sorted(reference_groups)),
        "n_top_proteins": n_top,
        "n_young": int(len(young)),
        "n_old_controls": int(len(old)),
        "n_treated": int(len(treated)),
        "pc1_explained_variance_ratio": explained,
        "orientation_sign": orientation_sign,
        "axis_orientation": "higher_older_like",
        "analysis_scale": "log2_positive_abundance",
        "young_control_median": young_median,
        "old_control_median": old_median,
        "treated_median": treated_median,
        "old_vs_young_gap": old_young_gap,
        "treated_vs_old_control": treated_vs_old,
        "treated_vs_young_control": treated_vs_young,
        "treated_fraction_of_old_young_gap": fraction,
        "old_young_gap_ci_crosses_zero": old_gap_crosses_zero,
        "treated_fraction_interpretation": (
            "unstable_denominator_old_young_gap_ci_crosses_zero"
            if old_gap_crosses_zero
            else "old_young_gap_bootstrap_ci_excludes_zero"
        ),
        "treated_shift_label": shift_label,
        "treated_vs_old_permutation_p_value": p_value,
        "available": True,
        "estimable": True,
        "reason": (
            "Small plasma cohort; axis is an age-state orientation of PC1, not causal evidence."
            if len(score_df) < 50
            else ""
        ),
        "n_used": int(len(score_df)),
        "method": method,
        "ci_low": ci.get("treated_vs_old_control_ci_low", np.nan),
        "ci_high": ci.get("treated_vs_old_control_ci_high", np.nan),
        "evidence_level": 1,
    }
    summary.update(ci)
    score_df["available"] = True
    score_df["estimable"] = True
    score_df["reason"] = ""
    score_df["n_used"] = int(len(score_df))
    score_df["method"] = method
    score_df["ci_low"] = np.nan
    score_df["ci_high"] = np.nan
    score_df["evidence_level"] = 1
    loading_df["available"] = True
    loading_df["estimable"] = True
    loading_df["reason"] = ""
    loading_df["n_used"] = n_top
    loading_df["method"] = method
    loading_df["ci_low"] = np.nan
    loading_df["ci_high"] = np.nan
    loading_df["evidence_level"] = 1
    return score_df, loading_df, pd.DataFrame([summary])


def correlate_plasma_axis_with_delta_age(
    prim_meta: pd.DataFrame,
    plasma_meta: pd.DataFrame,
    *,
    axis_col: str = "plasma_age_axis_score",
    delta_age_col: str = "delta_age",
    animal_col: str = "animal_id",
    confidence_col: str = "animal_id_confidence",
    high_conf_values: Sequence[str] = ("high", "metadata_exact"),
    min_animals: int = 8,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Correlate oriented plasma age-state scores with tissue delta-age by animal."""
    method = "linked_plasma_age_axis_delta_age_spearman"
    required_prim = {animal_col, delta_age_col}
    required_plasma = {animal_col, axis_col}
    if not required_prim.issubset(prim_meta.columns) or not required_plasma.issubset(plasma_meta.columns):
        return _result_stub(
            "Cannot correlate plasma axis with delta_age: animal_id, delta_age, or axis score is missing.",
            method,
            extra={"n_animals": 0, "spearman_rho": np.nan},
        )

    prim = prim_meta.dropna(subset=[animal_col, delta_age_col]).copy()
    prim[animal_col] = prim[animal_col].astype(str).str.strip()
    prim[delta_age_col] = pd.to_numeric(prim[delta_age_col], errors="coerce")
    prim = prim.dropna(subset=[delta_age_col])
    prim_animal = (
        prim.groupby(animal_col, as_index=False)
        .agg(
            delta_age=(delta_age_col, "median"),
            bulk_group=("group", _mode_text) if "group" in prim.columns else (delta_age_col, "size"),
            n_bulk_samples=(delta_age_col, "size"),
        )
    )

    plasma = plasma_meta.dropna(subset=[animal_col, axis_col]).copy()
    if confidence_col in plasma.columns:
        allowed = {str(value).strip().lower() for value in high_conf_values}
        plasma = plasma.loc[
            plasma[confidence_col].astype(str).str.strip().str.lower().isin(allowed)
        ].copy()
    plasma[animal_col] = plasma[animal_col].astype(str).str.strip()
    plasma[axis_col] = pd.to_numeric(plasma[axis_col], errors="coerce")
    plasma = plasma.dropna(subset=[axis_col])
    plasma_animal = (
        plasma.groupby(animal_col, as_index=False)
        .agg(
            plasma_age_axis_score=(axis_col, "median"),
            plasma_group=("group", _mode_text) if "group" in plasma.columns else (axis_col, "size"),
            n_plasma_samples=(axis_col, "size"),
        )
    )

    linked = prim_animal.merge(plasma_animal, on=animal_col, how="inner")
    n_animals = int(linked[animal_col].nunique()) if not linked.empty else 0
    if n_animals < int(min_animals):
        return _result_stub(
            f"Too few linked animals for plasma axis to delta_age correlation (n={n_animals}).",
            method,
            extra={"n_animals": n_animals, "spearman_rho": np.nan},
        )

    x = linked["plasma_age_axis_score"].to_numpy(float)
    y = linked["delta_age"].to_numpy(float)
    rho = float(stats.spearmanr(x, y).correlation)

    rng = np.random.default_rng(random_state)
    boot = []
    idx = np.arange(len(linked))
    for _ in range(int(n_bootstrap)):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        xb = x[sample_idx]
        yb = y[sample_idx]
        if np.nanstd(xb) == 0 or np.nanstd(yb) == 0:
            continue
        rb = stats.spearmanr(xb, yb).correlation
        if np.isfinite(rb):
            boot.append(float(rb))
    ci_low = float(np.percentile(boot, 2.5)) if boot else np.nan
    ci_high = float(np.percentile(boot, 97.5)) if boot else np.nan

    perm = []
    for _ in range(int(n_permutations)):
        shuffled = rng.permutation(x)
        rp = stats.spearmanr(shuffled, y).correlation
        if np.isfinite(rp):
            perm.append(float(rp))
    if perm:
        n_extreme = int(np.sum(np.abs(np.asarray(perm, dtype=float)) >= abs(rho)))
        permutation_p = float((n_extreme + 1) / (len(perm) + 1))
    else:
        permutation_p = np.nan
    interpretation = (
        "older_like_plasma_tracks_higher_delta_age"
        if rho > 0
        else ("older_like_plasma_tracks_lower_delta_age" if rho < 0 else "no_monotonic_association")
    )

    return pd.DataFrame(
        [
            {
                "n_animals": n_animals,
                "spearman_rho": rho,
                "rho_ci_low": ci_low,
                "rho_ci_high": ci_high,
                "permutation_p_value": permutation_p,
                "interpretation_label": interpretation,
                "available": True,
                "estimable": True,
                "reason": (
                    f"Linked animal count is modest (n={n_animals}); interpret correlation as hypothesis-generating."
                    if n_animals < 30
                    else ""
                ),
                "n_used": n_animals,
                "method": method,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "evidence_level": 2,
            }
        ]
    )
