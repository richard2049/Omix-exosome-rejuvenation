"""
rejuvenation.py

Helpers to compute rejuvenation-related metrics from the clock output.

This module provides:
  - delta_age = predicted_age - chronological_age
  - global rejuvenation effect (treated vs control, with bootstrap CI)
  - tissue-level rejuvenation summary (per-tissue effect sizes)
  - simple expression-based tissue effects (mean treated-control shift)

It assumes:
  - Sample-level metadata includes group labels, tissues, and ages
  - Clock predictions are already computed and stored in the metadata
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from .effect_sizes import hedges_g
from .logging_utils import get_logger

logger = get_logger(__name__)


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return pd.Series(out, index=p_values.index, dtype=float)

    pv = p[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    q = ranked * m / (np.arange(1, len(ranked) + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    out_idx = np.where(mask)[0][order]
    out[out_idx] = q
    return pd.Series(out, index=p_values.index, dtype=float)


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

    # Basic NaN sanity check
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


def annotate_effect_uncertainty(
    df: pd.DataFrame,
    effect_col: str = "effect_median",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
) -> pd.DataFrame:
    """
    Add conservative interpretation fields for delta-age effect summaries.

    Effects remain in their original units. The added signal-to-uncertainty
    score is only a prioritization aid: it must not be interpreted as a
    p-value, posterior probability, or normalized treatment effect.
    """
    out = df.copy()
    if out.empty:
        return out

    default = pd.Series(np.nan, index=out.index, dtype=float)
    effect = pd.to_numeric(out[effect_col] if effect_col in out.columns else default, errors="coerce")
    ci_low = pd.to_numeric(out[ci_low_col] if ci_low_col in out.columns else default, errors="coerce")
    ci_high = pd.to_numeric(out[ci_high_col] if ci_high_col in out.columns else default, errors="coerce")

    ci_width = ci_high - ci_low
    finite_ci = ci_low.notna() & ci_high.notna() & (ci_width > 0)
    ci_crosses_zero = finite_ci & (ci_low <= 0) & (ci_high >= 0)
    ci_excludes_zero = finite_ci & ((ci_high < 0) | (ci_low > 0))

    direction = pd.Series("unresolved", index=out.index, dtype="object")
    direction.loc[effect < 0] = "younger_shift"
    direction.loc[effect > 0] = "older_shift"

    interpretation = pd.Series("uncertain_missing_ci", index=out.index, dtype="object")
    interpretation.loc[ci_crosses_zero & (direction == "younger_shift")] = "nominal_younger_shift_ci_crosses_zero"
    interpretation.loc[ci_crosses_zero & (direction == "older_shift")] = "nominal_older_shift_ci_crosses_zero"
    interpretation.loc[ci_excludes_zero & (direction == "younger_shift")] = "supported_younger_shift_ci_excludes_zero"
    interpretation.loc[ci_excludes_zero & (direction == "older_shift")] = "supported_older_shift_ci_excludes_zero"
    interpretation.loc[finite_ci & (direction == "unresolved")] = "near_zero_effect"

    signal_to_uncertainty = effect.abs() / ci_width.replace(0, np.nan)
    signal_to_uncertainty = signal_to_uncertainty.where(finite_ci)

    out["ci_width"] = ci_width
    out["ci_crosses_zero"] = ci_crosses_zero.fillna(False)
    out["effect_direction"] = direction
    out["interpretation_label"] = interpretation
    out["signal_to_uncertainty"] = signal_to_uncertainty
    return out


TreatmentContrast = Tuple[str, str, str]


def _holm_adjust(p_values: pd.Series, family_size: Optional[int] = None) -> pd.Series:
    """Holm-adjust finite p-values, optionally padding the planned family with p=1."""
    numeric = pd.to_numeric(p_values, errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric)
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    if not finite.any():
        return out

    observed = numeric.loc[finite].to_numpy(dtype=float)
    m = max(int(family_size or len(observed)), len(observed))
    order = np.argsort(observed)
    ranked = observed[order]
    adjusted_ranked = np.maximum.accumulate(
        np.asarray([(m - i) * value for i, value in enumerate(ranked)], dtype=float)
    )
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    out.loc[finite] = adjusted
    return out


def _bh_adjust_with_family_size(p_values: pd.Series, family_size: int) -> pd.Series:
    """BH-adjust finite p-values while retaining unavailable planned hypotheses."""
    numeric = pd.to_numeric(p_values, errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric)
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    if not finite.any():
        return out

    observed = numeric.loc[finite].to_numpy(dtype=float)
    m = max(int(family_size), len(observed))
    order = np.argsort(observed)
    ranked = observed[order]
    adjusted_ranked = ranked * m / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    out.loc[finite] = adjusted
    return out


def _prepare_animal_outcomes(
    data: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    animal_col: str,
    tissue_col: Optional[str] = None,
    sex_col: Optional[str] = None,
) -> pd.DataFrame:
    """Validate identifiers and return a deterministic animal-outcome frame."""
    required = {group_col, value_col, animal_col}
    if tissue_col:
        required.add(tissue_col)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required animal-outcome columns: {', '.join(missing)}")

    columns = list(required)
    if sex_col and sex_col in data.columns:
        columns.append(sex_col)
    frame = data.loc[:, list(dict.fromkeys(columns))].copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=[group_col, animal_col, value_col])
    frame[group_col] = frame[group_col].astype(str)
    frame[animal_col] = frame[animal_col].astype(str)
    if tissue_col:
        frame = frame.dropna(subset=[tissue_col])
        frame[tissue_col] = frame[tissue_col].astype(str)

    group_conflicts = frame.groupby(animal_col, observed=True)[group_col].nunique()
    if (group_conflicts > 1).any():
        animals = ", ".join(group_conflicts.loc[group_conflicts > 1].index.astype(str)[:5])
        raise ValueError(f"Animals assigned to multiple treatment groups: {animals}")

    if sex_col and sex_col in frame.columns:
        sex_text = frame[sex_col].astype("string").str.strip()
        known = frame.loc[sex_text.notna() & sex_text.ne(""), [animal_col, sex_col]].copy()
        sex_conflicts = known.groupby(animal_col, observed=True)[sex_col].nunique()
        if (sex_conflicts > 1).any():
            animals = ", ".join(sex_conflicts.loc[sex_conflicts > 1].index.astype(str)[:5])
            raise ValueError(f"Animals assigned to multiple sex values: {animals}")
        frame[sex_col] = sex_text.fillna("unknown").replace("", "unknown").astype(str)
    else:
        sex_col = sex_col or "__permutation_stratum__"
        frame[sex_col] = "all"

    if tissue_col:
        duplicate = frame.duplicated([animal_col, tissue_col], keep=False)
        if duplicate.any():
            examples = frame.loc[duplicate, [animal_col, tissue_col]].head(5).astype(str)
            text = "; ".join(examples.agg("/".join, axis=1))
            raise ValueError(f"Duplicate animal-tissue outcome rows are not permitted: {text}")

    sort_cols = [group_col, animal_col]
    if tissue_col:
        sort_cols.insert(0, tissue_col)
    return frame.sort_values(sort_cols).reset_index(drop=True)


def _bootstrap_mean_difference(
    data: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    treated_label: str,
    control_label: str,
    strata_col: str,
    n_bootstrap: int,
    random_state: int,
) -> Tuple[float, float]:
    """Bootstrap animals within treatment-by-stratum cells."""
    rng = np.random.default_rng(random_state)
    cells: Dict[Tuple[str, str], np.ndarray] = {}
    for (group, stratum), sub in data.groupby([group_col, strata_col], sort=True, observed=True):
        if group in {treated_label, control_label}:
            cells[(str(group), str(stratum))] = sub[value_col].to_numpy(dtype=float)

    boot = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        sampled: Dict[str, List[np.ndarray]] = {treated_label: [], control_label: []}
        for (group, _), values in cells.items():
            sampled[group].append(rng.choice(values, size=len(values), replace=True))
        treated = np.concatenate(sampled[treated_label])
        control = np.concatenate(sampled[control_label])
        boot[index] = float(treated.mean() - control.mean())
    return tuple(float(x) for x in np.quantile(boot, [0.025, 0.975]))


def _stratified_permutation_p_value(
    data: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    treated_label: str,
    control_label: str,
    strata_col: str,
    n_permutations: int,
    random_state: int,
) -> float:
    """Two-sided animal-label permutation, preserving arm counts within sex strata."""
    rng = np.random.default_rng(random_state)
    values = data[value_col].to_numpy(dtype=float)
    labels = data[group_col].astype(str).to_numpy()
    observed = abs(
        float(values[labels == treated_label].mean() - values[labels == control_label].mean())
    )
    stratum_indices = [
        np.asarray(index, dtype=int)
        for index in data.groupby(strata_col, sort=True, observed=True).indices.values()
    ]
    exceedances = 0
    for _ in range(int(n_permutations)):
        permuted = labels.copy()
        for index in stratum_indices:
            permuted[index] = rng.permutation(permuted[index])
        statistic = abs(
            float(
                values[permuted == treated_label].mean()
                - values[permuted == control_label].mean()
            )
        )
        exceedances += int(statistic >= observed - 1e-12)
    return float((exceedances + 1) / (int(n_permutations) + 1))


def _animal_contrast(
    data: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    animal_col: str,
    sex_col: str,
    contrast: TreatmentContrast,
    min_per_group: int,
    n_bootstrap: int,
    n_permutations: int,
    random_state: int,
) -> Optional[Dict[str, object]]:
    name, treated_label, control_label = contrast
    subset = data.loc[data[group_col].isin([treated_label, control_label])].copy()
    treated = subset.loc[subset[group_col].eq(treated_label), value_col].to_numpy(dtype=float)
    control = subset.loc[subset[group_col].eq(control_label), value_col].to_numpy(dtype=float)
    n_treated = int(subset.loc[subset[group_col].eq(treated_label), animal_col].nunique())
    n_control = int(subset.loc[subset[group_col].eq(control_label), animal_col].nunique())
    if n_treated < int(min_per_group) or n_control < int(min_per_group):
        return None
    if len(treated) != n_treated or len(control) != n_control:
        raise ValueError("Animal contrast input must contain exactly one row per animal")

    effect = float(treated.mean() - control.mean())
    standardized_effect = hedges_g(treated, control)
    ci_low, ci_high = _bootstrap_mean_difference(
        subset,
        group_col=group_col,
        value_col=value_col,
        treated_label=treated_label,
        control_label=control_label,
        strata_col=sex_col,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    p_value = _stratified_permutation_p_value(
        subset,
        group_col=group_col,
        value_col=value_col,
        treated_label=treated_label,
        control_label=control_label,
        strata_col=sex_col,
        n_permutations=n_permutations,
        random_state=random_state + 1,
    )
    return {
        "contrast": name,
        "treated_group": treated_label,
        "control_group": control_label,
        "n_trt": n_treated,
        "n_ctrl": n_control,
        "n_treated_animals": n_treated,
        "n_control_animals": n_control,
        "n_used": n_treated + n_control,
        "mean_effect": effect,
        "standardized_effect": standardized_effect,
        "standardized_effect_method": "hedges_g_pooled_sd",
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_type": "pointwise_percentile_bootstrap",
        "ci_level": 0.95,
        "p_value": p_value,
        "estimand": "difference_in_animal_means",
        "bootstrap_unit": "animal",
        "permutation_unit": "animal",
        "permutation_strata": sex_col,
    }


def summarize_animal_treatment_contrasts_by_tissue(
    meta_with_delta: pd.DataFrame,
    *,
    tissue_col: str,
    group_col: str,
    animal_col: str,
    sex_col: str,
    value_col: str = "delta_age",
    contrasts: Sequence[TreatmentContrast] = (
        ("SRC_vs_vehicle", "O_GES", "O_V"),
        ("WTC_vs_vehicle", "O_WT", "O_V"),
        ("SRC_vs_WTC", "O_GES", "O_WT"),
    ),
    primary_contrast: str = "SRC_vs_vehicle",
    min_per_group: int = 2,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate prespecified tissue contrasts with animals as independent units."""
    frame = _prepare_animal_outcomes(
        meta_with_delta,
        group_col=group_col,
        value_col=value_col,
        animal_col=animal_col,
        tissue_col=tissue_col,
        sex_col=sex_col,
    )
    planned_tissues = sorted(frame[tissue_col].dropna().astype(str).unique())
    rows: List[Dict[str, object]] = []
    for tissue_index, tissue in enumerate(planned_tissues):
        tissue_data = frame.loc[frame[tissue_col].eq(tissue)].copy()
        for contrast_index, contrast in enumerate(contrasts):
            seed = int(random_state + 1009 * tissue_index + 97 * contrast_index)
            result = _animal_contrast(
                tissue_data,
                group_col=group_col,
                value_col=value_col,
                animal_col=animal_col,
                sex_col=sex_col,
                contrast=contrast,
                min_per_group=min_per_group,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                random_state=seed,
            )
            if result is None:
                name, treated_label, control_label = contrast
                n_treated = int(
                    tissue_data.loc[tissue_data[group_col].eq(treated_label), animal_col].nunique()
                )
                n_control = int(
                    tissue_data.loc[tissue_data[group_col].eq(control_label), animal_col].nunique()
                )
                rows.append(
                    {
                        "tissue": tissue,
                        "contrast": name,
                        "treated_group": treated_label,
                        "control_group": control_label,
                        "is_primary": name == primary_contrast,
                        "n_trt": n_treated,
                        "n_ctrl": n_control,
                        "n_treated_animals": n_treated,
                        "n_control_animals": n_control,
                        "n_used": n_treated + n_control,
                        "mean_effect": np.nan,
                        "standardized_effect": np.nan,
                        "standardized_effect_method": "hedges_g_pooled_sd",
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "ci_type": "pointwise_percentile_bootstrap",
                        "ci_level": 0.95,
                        "p_value": np.nan,
                        "estimand": "difference_in_animal_means",
                        "bootstrap_unit": "animal",
                        "permutation_unit": "animal",
                        "permutation_strata": sex_col,
                        "n_tissues_in_family": len(planned_tissues),
                        "available": True,
                        "estimable": False,
                        "reason": (
                            f"Insufficient animal support: {n_treated} {treated_label} and "
                            f"{n_control} {control_label}; minimum {int(min_per_group)} per arm."
                        ),
                        "method": "animal_mean_difference_bootstrap_permutation_by_tissue",
                        "evidence_level": 0,
                    }
                )
                continue
            result.update(
                {
                    "tissue": tissue,
                    "is_primary": contrast[0] == primary_contrast,
                    "n_tissues_in_family": len(planned_tissues),
                    "available": True,
                    "estimable": True,
                    "reason": "",
                    "method": "animal_mean_difference_bootstrap_permutation_by_tissue",
                    "evidence_level": 1,
                }
            )
            rows.append(result)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    n_tissues = len(planned_tissues)
    out["primary_fwer_p_value"] = np.nan
    primary = out["contrast"].eq(primary_contrast)
    out.loc[primary, "primary_fwer_p_value"] = np.minimum(
        1.0, pd.to_numeric(out.loc[primary, "p_value"], errors="coerce") * n_tissues
    )
    out["holm_all_contrasts_p_value"] = _holm_adjust(
        out["p_value"], family_size=n_tissues * len(contrasts)
    )
    out["bh_by_contrast_q_value"] = np.nan
    for _, index in out.groupby("contrast", sort=False).groups.items():
        out.loc[index, "bh_by_contrast_q_value"] = _bh_adjust_with_family_size(
            out.loc[index, "p_value"], family_size=n_tissues
        )
    out["p_value_adjusted"] = out["holm_all_contrasts_p_value"]
    out.loc[primary, "p_value_adjusted"] = out.loc[primary, "primary_fwer_p_value"]
    return annotate_effect_uncertainty(out, effect_col="mean_effect")


def summarize_global_animal_treatment_contrasts(
    meta_with_delta: pd.DataFrame,
    *,
    tissue_col: str,
    group_col: str,
    animal_col: str,
    sex_col: str,
    value_col: str = "delta_age",
    contrasts: Sequence[TreatmentContrast] = (
        ("SRC_vs_vehicle", "O_GES", "O_V"),
        ("WTC_vs_vehicle", "O_WT", "O_V"),
        ("SRC_vs_WTC", "O_GES", "O_WT"),
    ),
    primary_contrast: str = "SRC_vs_vehicle",
    min_per_group: int = 4,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Secondary global summary after reducing repeated tissues to one row per animal."""
    frame = _prepare_animal_outcomes(
        meta_with_delta,
        group_col=group_col,
        value_col=value_col,
        animal_col=animal_col,
        tissue_col=tissue_col,
        sex_col=sex_col,
    )
    animal = (
        frame.groupby([group_col, animal_col, sex_col], as_index=False, observed=True)
        .agg(**{value_col: (value_col, "mean"), "n_tissues_per_animal": (tissue_col, "nunique")})
        .sort_values([group_col, animal_col])
        .reset_index(drop=True)
    )
    rows: List[Dict[str, object]] = []
    for contrast_index, contrast in enumerate(contrasts):
        result = _animal_contrast(
            animal,
            group_col=group_col,
            value_col=value_col,
            animal_col=animal_col,
            sex_col=sex_col,
            contrast=contrast,
            min_per_group=min_per_group,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            random_state=int(random_state + 97 * contrast_index),
        )
        if result is None:
            continue
        used_groups = {contrast[1], contrast[2]}
        used = animal.loc[animal[group_col].isin(used_groups)]
        result.update(
            {
                "scope": "global_secondary",
                "is_primary": contrast[0] == primary_contrast,
                "n_tissues_per_animal_min": int(used["n_tissues_per_animal"].min()),
                "n_tissues_per_animal_max": int(used["n_tissues_per_animal"].max()),
                "available": True,
                "estimable": True,
                "reason": (
                    "Secondary across-tissue summary; each animal contributes one mean over its observed tissues."
                ),
                "method": "animal_aggregated_global_bootstrap_permutation",
                "evidence_level": 1,
            }
        )
        rows.append(result)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["holm_three_contrasts_p_value"] = _holm_adjust(
        out["p_value"], family_size=len(contrasts)
    )
    return annotate_effect_uncertainty(out, effect_col="mean_effect")


def summarize_clustered_treatment_sensitivity(
    meta_with_delta: pd.DataFrame,
    *,
    tissue_col: str,
    group_col: str,
    animal_col: str,
    sex_col: str,
    age_col: str,
    value_col: str = "delta_age",
    contrasts: Sequence[TreatmentContrast] = (
        ("SRC_vs_vehicle", "O_GES", "O_V"),
        ("WTC_vs_vehicle", "O_WT", "O_V"),
        ("SRC_vs_WTC", "O_GES", "O_WT"),
    ),
    primary_contrast: str = "SRC_vs_vehicle",
    min_animals_per_group: int = 4,
) -> pd.DataFrame:
    """Tissue-adjusted OLS sensitivity with animal-clustered uncertainty."""
    frame = _prepare_animal_outcomes(
        meta_with_delta,
        group_col=group_col,
        value_col=value_col,
        animal_col=animal_col,
        tissue_col=tissue_col,
        sex_col=sex_col,
    )
    if frame.empty:
        return pd.DataFrame()
    if age_col not in meta_with_delta.columns:
        raise ValueError(f"Missing required clustered-sensitivity age column: {age_col}")
    age_map = meta_with_delta[[animal_col, age_col]].copy()
    age_map = age_map.dropna(subset=[animal_col])
    age_map[animal_col] = age_map[animal_col].astype(str)
    valid_animals = set(frame[animal_col].astype(str))
    age_map = age_map.loc[age_map[animal_col].isin(valid_animals)]
    age_map[age_col] = pd.to_numeric(age_map[age_col], errors="coerce")
    age_conflicts = age_map.dropna().groupby(animal_col, observed=True)[age_col].nunique()
    if (age_conflicts > 1).any():
        raise ValueError("Animals have conflicting chronological ages")
    age_map = age_map.dropna().drop_duplicates(animal_col).set_index(animal_col)[age_col]
    frame[age_col] = frame[animal_col].map(age_map)

    rows: List[Dict[str, object]] = []
    for name, treated_label, control_label in contrasts:
        sub = frame.loc[frame[group_col].isin([treated_label, control_label])].copy()
        n_treated = int(sub.loc[sub[group_col].eq(treated_label), animal_col].nunique())
        n_control = int(sub.loc[sub[group_col].eq(control_label), animal_col].nunique())
        if n_treated < min_animals_per_group or n_control < min_animals_per_group:
            continue
        sub["treated_binary"] = sub[group_col].eq(treated_label).astype(float)
        design = pd.DataFrame(
            {"intercept": 1.0, "treated_binary": sub["treated_binary"]}, index=sub.index
        )
        covariates = ["tissue"]
        tissue_terms = pd.get_dummies(
            sub[tissue_col].astype(str), prefix="tissue", drop_first=True, dtype=float
        )
        design = pd.concat([design, tissue_terms], axis=1)
        age = pd.to_numeric(sub[age_col], errors="coerce")
        if age.notna().all() and age.nunique() > 1:
            design["age_centered"] = age - age.mean()
            covariates.append("age")
        sex_terms = pd.get_dummies(
            sub[sex_col].astype(str), prefix="sex", drop_first=True, dtype=float
        )
        if not sex_terms.empty:
            design = pd.concat([design, sex_terms], axis=1)
            covariates.append("sex")
        design["__outcome__"] = sub[value_col].to_numpy(dtype=float)
        design = design.dropna(axis=0, how="any")
        sub = sub.loc[design.index]
        x = design.drop(columns="__outcome__").to_numpy(dtype=float)
        y = design["__outcome__"].to_numpy(dtype=float)
        if np.linalg.matrix_rank(x) != x.shape[1]:
            raise ValueError(f"Rank-deficient clustered sensitivity design for {name}")
        fitted = sm.OLS(y, x).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub[animal_col].to_numpy(), "use_correction": True},
            use_t=True,
        )
        effect = float(fitted.params[1])
        ci_low, ci_high = (float(x) for x in fitted.conf_int(alpha=0.05)[1])
        rows.append(
            {
                "contrast": name,
                "treated_group": treated_label,
                "control_group": control_label,
                "is_primary": name == primary_contrast,
                "mean_effect": effect,
                "effect_se": float(fitted.bse[1]),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_type": "pointwise_cluster_robust_t",
                "ci_level": 0.95,
                "p_value": float(fitted.pvalues[1]),
                "n_trt": n_treated,
                "n_ctrl": n_control,
                "n_used": int(len(sub)),
                "n_animals": int(sub[animal_col].nunique()),
                "n_tissues": int(sub[tissue_col].nunique()),
                "cluster_unit": animal_col,
                "covariates_used": ",".join(covariates),
                "available": True,
                "estimable": True,
                "reason": "Sensitivity analysis; not the primary treatment-effect estimator.",
                "method": "ols_tissue_age_sex_animal_clustered",
                "evidence_level": 1,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["holm_three_contrasts_p_value"] = _holm_adjust(
        out["p_value"], family_size=len(contrasts)
    )
    return annotate_effect_uncertainty(out, effect_col="mean_effect")


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

    return annotate_effect_uncertainty(pd.DataFrame(rows), effect_col="effect_median")


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
    outcome_col: str = "delta_age",
    min_per_group: int = 2,
    covariate_cols: Optional[Sequence[str]] = ("age", "sex", "batch"),
) -> pd.DataFrame:
    """
    Summarize tissue-wise treatment effects on a sample-level outcome using
    per-tissue linear models:

        outcome ~ treated + covariates

    This avoids pseudo-replication from treating genes as independent units
    for tissue-level inference.

    Parameters
    ----------
    expr : DataFrame
        Gene expression matrix (genes x samples). Used for sample alignment and
        gene-count metadata; inference is performed at sample level.
    meta : DataFrame
        Sample metadata. Must contain tissue_col, group_col, sample_id and outcome_col.
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

    Returns one row per tissue with covariate-adjusted treatment-effect summaries.
    """

    out_cols = [
        "tissue",
        "n_ctrl",
        "n_trt",
        "n_samples",
        "n_used",
        "n_genes_modeled",
        "method",
        "covariates_used",
        "top_genes",
        "mean_effect",
        "median_effect",
        "effect_se",
        "ci_low",
        "ci_high",
        "p_value",
        "fdr_q_value",
        "available",
        "estimable",
        "reason",
    ]

    required_cols = {tissue_col, group_col, "sample_id", outcome_col}
    missing = required_cols - set(meta.columns)
    if missing:
        logger.warning(
            "summarize_tissue_expression_effects: missing required "
            "metadata columns: %s. Returning empty DataFrame.",
            ", ".join(sorted(missing)),
        )
        return pd.DataFrame(columns=out_cols)

    # Drop rows without tissue/group/outcome
    meta = meta.copy()
    meta = meta.loc[
        meta[tissue_col].notna()
        & meta[group_col].notna()
        & pd.to_numeric(meta[outcome_col], errors="coerce").notna()
    ]
    if meta.empty:
        logger.warning(
            "summarize_tissue_expression_effects: no rows with tissue/group/outcome. Returning empty."
        )
        return pd.DataFrame(columns=out_cols)

    # Keep only control + treated labels
    valid_labels = list(control_labels) + list(treated_labels)
    meta = meta.loc[meta[group_col].isin(valid_labels)]
    if meta.empty:
        logger.warning(
            "summarize_tissue_expression_effects: no rows with group in %s. Returning empty.",
            valid_labels,
        )
        return pd.DataFrame(columns=out_cols)

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
        return pd.DataFrame(columns=out_cols)

    expr = expr.loc[:, common_ids]
    meta = meta.loc[common_ids]

    def _encode_covariate(name: str, s: pd.Series) -> Optional[pd.DataFrame]:
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().mean() >= 0.8 and s_num.nunique(dropna=True) > 1:
            return pd.DataFrame({name: s_num.astype(float)})

        name_low = name.lower()
        if ("sex" in name_low) or ("gender" in name_low):
            s_low = s.astype("string").str.strip().str.lower()
            mapped = s_low.map(
                {
                    "m": 1.0,
                    "male": 1.0,
                    "f": 0.0,
                    "female": 0.0,
                }
            )
            if mapped.notna().mean() >= 0.8 and mapped.nunique(dropna=True) > 1:
                return pd.DataFrame({name: mapped.astype(float)})

        # Generic categorical encoding (e.g., batch)
        s_cat = s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        if s_cat.notna().mean() < 0.8 or s_cat.nunique(dropna=True) <= 1:
            return None
        if s_cat.nunique(dropna=True) > 12:
            return None
        dummies = pd.get_dummies(s_cat, prefix=name, drop_first=True, dtype=float)
        if dummies.shape[1] == 0:
            return None
        return dummies

    rows = []
    for tissue, sub_meta in meta.groupby(tissue_col):
        # Boolean masks in this tissue
        is_ctrl = sub_meta[group_col].isin(control_labels)
        is_trt = sub_meta[group_col].isin(treated_labels)

        n_ctrl = int(is_ctrl.sum())
        n_trt = int(is_trt.sum())

        if n_ctrl < min_per_group or n_trt < min_per_group:
            continue

        ids = [s for s in sub_meta.index.tolist() if s in expr.columns]
        if len(ids) < (2 * min_per_group):
            continue
        sub = sub_meta.loc[ids].copy()
        sub["treated_binary"] = sub[group_col].isin(treated_labels).astype(float)
        sub[outcome_col] = pd.to_numeric(sub[outcome_col], errors="coerce")

        design = pd.DataFrame(index=sub.index)
        design["intercept"] = 1.0
        design["treated_binary"] = sub["treated_binary"].astype(float)
        used_covariates: List[str] = []

        for cov in list(covariate_cols or []):
            if cov not in sub.columns:
                continue
            encoded = _encode_covariate(cov, sub[cov])
            if encoded is None:
                continue
            for cname in encoded.columns:
                design[cname] = encoded[cname].astype(float)
            used_covariates.append(cov)

        design["__y__"] = sub[outcome_col].astype(float)
        design = design.dropna(axis=0, how="any")
        if design.empty:
            continue
        if design["treated_binary"].nunique() < 2:
            continue

        keep_ids = design.index
        if len(keep_ids) < (2 * min_per_group):
            continue

        n_ctrl_model = int((design["treated_binary"] == 0).sum())
        n_trt_model = int((design["treated_binary"] == 1).sum())
        if n_ctrl_model < min_per_group or n_trt_model < min_per_group:
            continue

        X_df = design.drop(columns=["__y__"])
        y = design["__y__"].to_numpy(dtype=float)
        X = X_df.to_numpy(dtype=float)
        n, p = X.shape
        if n <= (p + 1):
            continue

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat
        dof = n - p
        if dof <= 0:
            continue

        sigma2 = float(np.dot(resid, resid) / dof)
        xtx_inv = np.linalg.pinv(X.T @ X)
        var_treat = float(max(sigma2 * xtx_inv[1, 1], 0.0))
        effect = float(beta[1])
        effect_se = float(np.sqrt(var_treat)) if np.isfinite(var_treat) else np.nan

        if np.isfinite(effect_se) and effect_se > 0:
            t_stat = effect / effect_se
            p_value = float(2.0 * stats.t.sf(np.abs(t_stat), dof))
            t_crit = float(stats.t.ppf(0.975, dof))
            ci_low = float(effect - t_crit * effect_se)
            ci_high = float(effect + t_crit * effect_se)
        else:
            p_value = np.nan
            ci_low, ci_high = np.nan, np.nan

        rows.append(
            {
                "tissue": tissue,
                "n_ctrl": n_ctrl_model,
                "n_trt": n_trt_model,
                "n_samples": int(n),
                "n_used": int(n),
                "n_genes_modeled": int(expr.shape[0]),
                "method": "ols_sample_level_outcome_by_tissue",
                "covariates_used": ",".join(used_covariates) if used_covariates else "none",
                "top_genes": "",
                "mean_effect": effect,
                "median_effect": effect,
                "effect_se": effect_se,
                "ci_low": float(ci_low) if not np.isnan(ci_low) else np.nan,
                "ci_high": float(ci_high) if not np.isnan(ci_high) else np.nan,
                "p_value": p_value,
                "fdr_q_value": np.nan,
                "available": True,
                "estimable": True,
                "reason": "",
            }
        )

    if not rows:
        logger.warning(
            "summarize_tissue_expression_effects: no tissues passed the filters (min_per_group=%d). "
            "Returning empty DataFrame.",
            min_per_group,
        )
        return pd.DataFrame(columns=out_cols)

    out = pd.DataFrame(rows, columns=out_cols)
    out["fdr_q_value"] = _benjamini_hochberg(out["p_value"])
    return out


def _plasma_pair_design(
    groups: pd.Series,
    sex: pd.Series,
    *,
    group_a: str,
) -> tuple[np.ndarray, list[str]] | tuple[None, list[str]]:
    """Construct a full-rank pairwise group model with optional sex adjustment."""
    treatment = groups.astype(str).eq(str(group_a)).astype(float).to_numpy()
    columns = [np.ones(len(groups), dtype=float), treatment]
    names = ["intercept", "group_effect"]

    clean_sex = sex.astype("string").str.strip().str.upper()
    observed_sexes = sorted(clean_sex.dropna().unique().tolist())
    if len(observed_sexes) > 1:
        columns.append(clean_sex.eq(observed_sexes[-1]).astype(float).to_numpy())
        names.append(f"sex_{observed_sexes[-1]}")

    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None, names
    return design, names


def compute_plasma_protein_contrasts(
    plasma_expr: pd.DataFrame,
    plasma_meta: pd.DataFrame,
    *,
    feature_annotations: Optional[pd.DataFrame] = None,
    contrasts: Sequence[Tuple[str, str, str]] = (
        ("GES_vs_V", "GES", "V"),
        ("WT_vs_V", "WT", "V"),
        ("GES_vs_WT", "GES", "WT"),
    ),
    primary_contrast: str = "GES_vs_V",
    min_per_group: int = 3,
    n_bootstrap: int = 200,
    random_state: int = 42,
    min_sign_agreement: float = 0.8,
    stability_top_k: int = 500,
) -> pd.DataFrame:
    """Estimate prespecified within-plasma treatment contrasts.

    Each protein is analysed on a log2 scale with a pairwise OLS model adjusted
    for sex when both sexes are represented. HC3 covariance and t-based
    intervals protect the small-cohort inference from equal-variance
    assumptions. The plasma sample is the statistical unit; no plasma-to-tissue
    or plasma-to-animal linkage is required or implied.
    """
    required_meta = {"sample_id", "group", "sex"}
    missing_meta = required_meta - set(plasma_meta.columns)
    if missing_meta:
        raise ValueError(f"Plasma metadata missing required columns: {sorted(missing_meta)}")
    if plasma_expr.empty or plasma_meta.empty:
        raise ValueError("Plasma expression matrix and metadata must be non-empty.")
    if plasma_expr.index.has_duplicates:
        raise ValueError("Plasma feature identifiers must be unique; use protein accessions.")

    meta = plasma_meta.copy()
    meta["sample_id"] = meta["sample_id"].astype(str).str.strip()
    if meta["sample_id"].duplicated().any():
        raise ValueError("Plasma sample_id values must be unique.")
    meta = meta.set_index("sample_id")
    common = [sample for sample in plasma_expr.columns.astype(str) if sample in meta.index]
    if len(common) < 2 * int(min_per_group):
        raise ValueError(f"Too few plasma samples overlap metadata: {len(common)}")

    expr = plasma_expr.loc[:, common].apply(pd.to_numeric, errors="coerce")
    expr.index = expr.index.astype(str)
    finite_values = expr.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Plasma matrix contains no finite abundance values.")
    if np.any(finite_values <= 0):
        raise ValueError("Log2 plasma contrasts require strictly positive finite abundances.")
    expr_log2 = np.log2(expr)

    annotations = pd.DataFrame(index=expr.index)
    if feature_annotations is not None and not feature_annotations.empty:
        annotations = feature_annotations.copy()
        if "feature_id" in annotations.columns:
            annotations = annotations.set_index("feature_id", drop=False)
        annotations.index = annotations.index.astype(str)
        if annotations.index.has_duplicates:
            raise ValueError("Plasma feature annotation IDs must be unique.")
        annotations = annotations.reindex(expr.index.astype(str))

    def annotation(feature_id: object) -> tuple[str, str, str]:
        key = str(feature_id)
        accession = key
        gene_name = ""
        if key in annotations.index:
            row = annotations.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            accession_value = row.get("protein_accession", key)
            gene_value = row.get("gene_name", "")
            if pd.notna(accession_value) and str(accession_value).strip():
                accession = str(accession_value).strip()
            if pd.notna(gene_value) and str(gene_value).strip():
                gene_name = str(gene_value).strip()
        protein_label = gene_name or accession
        return accession, gene_name, protein_label

    records: list[dict[str, object]] = []
    for contrast_name, group_a, group_b in contrasts:
        pair_samples = meta.index[meta["group"].astype(str).isin([group_a, group_b])].tolist()
        pair_samples = [sample for sample in common if sample in pair_samples]
        if not pair_samples:
            continue
        pair_meta = meta.loc[pair_samples]

        for feature_id, row in expr_log2.loc[:, pair_samples].iterrows():
            values = pd.to_numeric(row, errors="coerce")
            valid = values.notna()
            model_meta = pair_meta.loc[valid.index[valid]].copy()
            model_values = values.loc[model_meta.index].to_numpy(dtype=float)
            counts = model_meta["group"].astype(str).value_counts()
            n_a = int(counts.get(group_a, 0))
            n_b = int(counts.get(group_b, 0))
            if min(n_a, n_b) < int(min_per_group):
                continue

            design, design_names = _plasma_pair_design(
                model_meta["group"],
                model_meta["sex"],
                group_a=group_a,
            )
            if design is None or len(model_values) <= design.shape[1]:
                continue

            # HC3 divides residuals by (1 - leverage). A leverage of one can
            # occur in sparse group-by-sex patterns and makes robust inference
            # undefined, so those feature/contrast rows are not reported as
            # estimable.
            hat = design @ np.linalg.pinv(design.T @ design) @ design.T
            if np.any(np.diag(hat) >= 1.0 - 1e-10):
                continue

            fit = sm.OLS(model_values, design).fit(cov_type="HC3", use_t=True)
            effect = float(fit.params[1])
            effect_se = float(fit.bse[1])
            p_value = float(fit.pvalues[1])
            ci_low, ci_high = (float(value) for value in fit.conf_int(alpha=0.05)[1])
            if not np.all(np.isfinite([effect, effect_se, p_value, ci_low, ci_high])):
                continue
            accession, gene_name, protein_label = annotation(feature_id)
            records.append(
                {
                    "feature_id": str(feature_id),
                    "protein_accession": accession,
                    "gene_name": gene_name,
                    "protein": protein_label,
                    "contrast": str(contrast_name),
                    "group_a": str(group_a),
                    "group_b": str(group_b),
                    "is_primary": str(contrast_name) == str(primary_contrast),
                    "log2_fold_change": effect,
                    "abs_log2_fold_change": abs(effect),
                    "effect_se": effect_se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "mean_log2_group_a": float(
                        values.loc[
                            model_meta.index[model_meta["group"].astype(str).eq(group_a)]
                        ].mean()
                    ),
                    "mean_log2_group_b": float(
                        values.loc[
                            model_meta.index[model_meta["group"].astype(str).eq(group_b)]
                        ].mean()
                    ),
                    "n_group_a": n_a,
                    "n_group_b": n_b,
                    "n_used": int(len(model_values)),
                    "covariates_used": (
                        "sex" if any(name.startswith("sex_") for name in design_names) else "none"
                    ),
                    "direction": (
                        f"higher_in_{group_a}"
                        if effect > 0
                        else (f"lower_in_{group_a}" if effect < 0 else "no_difference")
                    ),
                    "bootstrap_ci_low": np.nan,
                    "bootstrap_ci_high": np.nan,
                    "sign_agreement": np.nan,
                    "stable_association": False,
                    "stability_tested": False,
                    "fdr_significant": False,
                    "analysis_scale": "log2_positive_abundance",
                    "source_scale_status": "SOURCE_VERIFICATION_PENDING",
                    "claim_class": "observed_association",
                    "available": True,
                    "estimable": True,
                    "reason": "",
                    "method": "pairwise_ols_hc3_log2_abundance_group_sex",
                    "evidence_level": 1,
                }
            )

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame.from_records(records)
    result["q_value"] = np.nan
    for contrast_name, indices in result.groupby("contrast", sort=False).groups.items():
        result.loc[indices, "q_value"] = _benjamini_hochberg(
            result.loc[indices, "p_value"]
        ).to_numpy()
    result["fdr_significant"] = result["q_value"].lt(0.05)

    rng = np.random.default_rng(random_state)
    if int(n_bootstrap) > 0 and int(stability_top_k) > 0:
        for _, contrast_frame in result.groupby("contrast", sort=False):
            top_indices = (
                contrast_frame.sort_values(
                    ["p_value", "abs_log2_fold_change"],
                    ascending=[True, False],
                )
                .head(int(stability_top_k))
                .index
            )
            for idx in top_indices:
                row = result.loc[idx]
                pair_samples = meta.index[
                    meta["group"].astype(str).isin([row["group_a"], row["group_b"]])
                ].tolist()
                pair_samples = [sample for sample in common if sample in pair_samples]
                values = expr_log2.loc[row["feature_id"], pair_samples]
                valid = values.notna()
                boot_meta = meta.loc[valid.index[valid], ["group", "sex"]].copy()
                boot_meta["value"] = values.loc[boot_meta.index].to_numpy(dtype=float)
                strata = (
                    ["group", "sex"]
                    if boot_meta["sex"].nunique(dropna=True) > 1
                    else ["group"]
                )
                grouper: str | list[str] = strata if len(strata) > 1 else strata[0]
                stratum_indices = [
                    frame.index.to_numpy()
                    for _, frame in boot_meta.groupby(grouper, dropna=False)
                ]
                boot_effects: list[float] = []
                for _ in range(int(n_bootstrap)):
                    sampled_ids = np.concatenate(
                        [rng.choice(ids, size=len(ids), replace=True) for ids in stratum_indices]
                    )
                    sampled = boot_meta.loc[sampled_ids]
                    design, _ = _plasma_pair_design(
                        sampled["group"],
                        sampled["sex"],
                        group_a=str(row["group_a"]),
                    )
                    if design is None:
                        continue
                    coefficients = np.linalg.lstsq(
                        design,
                        sampled["value"].to_numpy(dtype=float),
                        rcond=None,
                    )[0]
                    if len(coefficients) > 1 and np.isfinite(coefficients[1]):
                        boot_effects.append(float(coefficients[1]))
                if not boot_effects:
                    continue
                boot_array = np.asarray(boot_effects, dtype=float)
                boot_low, boot_high = np.percentile(boot_array, [2.5, 97.5])
                sign_agreement = float(
                    max(np.mean(boot_array > 0), np.mean(boot_array < 0))
                )
                result.at[idx, "bootstrap_ci_low"] = float(boot_low)
                result.at[idx, "bootstrap_ci_high"] = float(boot_high)
                result.at[idx, "sign_agreement"] = sign_agreement
                result.at[idx, "stable_association"] = bool(
                    (boot_low > 0 or boot_high < 0)
                    and sign_agreement >= float(min_sign_agreement)
                )
                result.at[idx, "stability_tested"] = True

    return result.sort_values(
        ["is_primary", "contrast", "q_value", "abs_log2_fold_change"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
