"""
Cross-species attribution and validation helpers for the OMIX Exosome Rejuvenation pipeline.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .clocks import predict_biological_age, train_transcriptomic_clock
from .effect_sizes import hedges_g
from .logging_utils import get_logger
from .reason_codes import annotate_reason_fields

logger = get_logger(__name__)


def _normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _resolve_tissue_mapping(value: Any, mapping: Optional[Mapping[str, str]] = None) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    if not mapping:
        return raw
    normalized_map = {_normalize_token(k): v for k, v in mapping.items()}
    return normalized_map.get(_normalize_token(raw), raw)


def parse_mouse_exosome_sample_id(sample_id: str) -> Dict[str, Any]:
    text = str(sample_id).strip()
    match = re.match(
        r"^(?P<age>\d+)_(?P<sex>[A-Za-z])_(?P<tissue>[^_]+)_(?P<arm>[^_]+)_(?P<replicate>\d+)$",
        text,
    )
    if not match:
        raise ValueError(f"Unrecognized OMIX009283 sample ID format: {sample_id}")

    arm_map = {
        "baseline": "Baseline",
        "ctrl": "Ctrl",
        "veh": "Veh",
        "wt": "WT",
        "ges": "GES",
    }
    sex_raw = match.group("sex").upper()
    sex = {"W": "F", "F": "F", "M": "M"}.get(sex_raw, sex_raw)
    arm_raw = match.group("arm")

    return {
        "sample_id": text,
        "age": float(match.group("age")),
        "sex_raw": sex_raw,
        "sex": sex,
        "tissue": match.group("tissue"),
        "arm_raw": arm_raw,
        "arm": arm_map.get(arm_raw.lower(), arm_raw),
        "replicate": int(match.group("replicate")),
    }


def build_mouse_exosome_metadata(sample_ids: Iterable[str]) -> pd.DataFrame:
    rows = []
    for sample_id in sample_ids:
        try:
            parsed = parse_mouse_exosome_sample_id(str(sample_id))
            parsed["parse_ok"] = True
            parsed["reason"] = ""
        except Exception as exc:
            parsed = {
                "sample_id": str(sample_id),
                "age": np.nan,
                "sex_raw": pd.NA,
                "sex": pd.NA,
                "tissue": pd.NA,
                "arm_raw": pd.NA,
                "arm": pd.NA,
                "replicate": np.nan,
                "parse_ok": False,
                "reason": str(exc),
            }
        rows.append(parsed)
    meta = pd.DataFrame(rows)
    if "sample_id" in meta.columns:
        meta["sample_id"] = meta["sample_id"].astype(str)
    return meta


def _effect_with_uncertainty(
    treated: Sequence[float],
    control: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    treated_arr = np.asarray(list(treated), dtype=float)
    control_arr = np.asarray(list(control), dtype=float)
    treated_arr = treated_arr[np.isfinite(treated_arr)]
    control_arr = control_arr[np.isfinite(control_arr)]

    base = {
        "mean_effect": np.nan,
        "standardized_effect": np.nan,
        "standardized_ci_low": np.nan,
        "standardized_ci_high": np.nan,
        "standardized_effect_method": "hedges_g_pooled_sd",
        "ci_low": np.nan,
        "ci_high": np.nan,
        "permutation_p_value": np.nan,
        "n_treated": int(len(treated_arr)),
        "n_control": int(len(control_arr)),
    }
    if len(treated_arr) == 0 or len(control_arr) == 0:
        return base

    rng = np.random.default_rng(random_state)
    observed = float(np.mean(treated_arr) - np.mean(control_arr))
    base["mean_effect"] = observed
    base["standardized_effect"] = hedges_g(treated_arr, control_arr)

    boot = []
    standardized_boot = []
    for _ in range(int(n_bootstrap)):
        bt = rng.choice(treated_arr, size=len(treated_arr), replace=True)
        bc = rng.choice(control_arr, size=len(control_arr), replace=True)
        boot.append(float(np.mean(bt) - np.mean(bc)))
        boot_g = hedges_g(bt, bc)
        if np.isfinite(boot_g):
            standardized_boot.append(float(boot_g))
    if boot:
        base["ci_low"] = float(np.percentile(boot, 2.5))
        base["ci_high"] = float(np.percentile(boot, 97.5))
    if standardized_boot:
        base["standardized_ci_low"] = float(np.percentile(standardized_boot, 2.5))
        base["standardized_ci_high"] = float(np.percentile(standardized_boot, 97.5))

    pooled = np.concatenate([treated_arr, control_arr])
    n_t = len(treated_arr)
    perm = []
    for _ in range(int(n_permutations)):
        shuffled = rng.permutation(pooled)
        perm.append(float(np.mean(shuffled[:n_t]) - np.mean(shuffled[n_t:])))
    if perm:
        perm_arr = np.asarray(perm, dtype=float)
        base["permutation_p_value"] = float(np.mean(np.abs(perm_arr) >= abs(observed)))

    return base


def compute_mouse_exosome_tissue_effects(
    expr_log: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    reference_arms: Sequence[str] = ("Baseline",),
    contrasts: Sequence[Tuple[str, str]] = (("GES", "Veh"), ("WT", "Veh"), ("GES", "WT")),
    sample_id_col: str = "sample_id",
    tissue_col: str = "tissue",
    arm_col: str = "arm",
    age_col: str = "age",
    min_reference_samples: int = 20,
    min_reference_ages: int = 4,
    min_samples_per_group: int = 3,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    random_state: int = 42,
    return_sample_outcomes: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    outcome_frames = []

    if expr_log is None or meta is None or expr_log.empty or meta.empty:
        empty_result = annotate_reason_fields(pd.DataFrame(
            [
                {
                    "tissue": "NA",
                    "contrast": "NA",
                    "treated_arm": "NA",
                    "control_arm": "NA",
                    "available": False,
                    "estimable": False,
                    "reason": "Mouse exosome matrix and/or metadata unavailable.",
                    "n_used": 0,
                    "method": "mouse_tissue_clock_contrast",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }
            ]
        ))
        if return_sample_outcomes:
            return empty_result, pd.DataFrame()
        return empty_result

    meta = meta.copy()
    meta[sample_id_col] = meta[sample_id_col].astype(str)

    for tissue, sub in meta.groupby(tissue_col):
        sample_ids = [sid for sid in sub[sample_id_col].tolist() if sid in expr_log.columns]
        if not sample_ids:
            continue

        tissue_expr = expr_log.loc[:, sample_ids]
        sub = sub.set_index(sample_id_col).loc[sample_ids].reset_index()
        sub[age_col] = pd.to_numeric(sub[age_col], errors="coerce")
        ref = sub.loc[sub[arm_col].isin(reference_arms) & sub[age_col].notna()].copy()

        if len(ref) < int(min_reference_samples) or ref[age_col].nunique() < int(min_reference_ages):
            reason = (
                f"Insufficient reference samples for tissue clock in {tissue} "
                f"(n={len(ref)}, unique ages={ref[age_col].nunique()})."
            )
            for treated_arm, control_arm in contrasts:
                rows.append(
                    {
                        "tissue": tissue,
                        "contrast": f"{treated_arm}_vs_{control_arm}",
                        "treated_arm": treated_arm,
                        "control_arm": control_arm,
                        "n_reference": int(len(ref)),
                        "n_reference_ages": int(ref[age_col].nunique()),
                        "n_treated": 0,
                        "n_control": 0,
                        "mean_effect": np.nan,
                        "permutation_p_value": np.nan,
                        "clock_cv_mae": np.nan,
                        "clock_cv_spearman": np.nan,
                        "available": False,
                        "estimable": False,
                        "reason": reason,
                        "n_used": 0,
                        "method": "mouse_tissue_clock_contrast",
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "evidence_level": 0,
                    }
                )
            continue

        try:
            clock, _, metrics = train_transcriptomic_clock(
                tissue_expr,
                ref[[sample_id_col, age_col]].copy(),
                age_col=age_col,
                model="ridge",
                n_splits=min(5, max(2, int(ref[age_col].nunique()))),
                random_state=random_state,
                cv_group_col=None,
            )
            pred = predict_biological_age(clock, tissue_expr, sub[[sample_id_col]].copy())
            pred = pred.rename(columns={"predicted_age": "predicted_age_mouse"})
            sub = sub.merge(pred, on=sample_id_col, how="left")
            outcome_frames.append(
                sub[
                    [
                        column
                        for column in (
                            sample_id_col,
                            tissue_col,
                            arm_col,
                            "sex",
                            age_col,
                            "predicted_age_mouse",
                        )
                        if column in sub.columns
                    ]
                ].copy()
            )
        except Exception as exc:
            reason = f"Mouse tissue clock failed for {tissue}: {exc}"
            for treated_arm, control_arm in contrasts:
                rows.append(
                    {
                        "tissue": tissue,
                        "contrast": f"{treated_arm}_vs_{control_arm}",
                        "treated_arm": treated_arm,
                        "control_arm": control_arm,
                        "n_reference": int(len(ref)),
                        "n_reference_ages": int(ref[age_col].nunique()),
                        "n_treated": 0,
                        "n_control": 0,
                        "mean_effect": np.nan,
                        "permutation_p_value": np.nan,
                        "clock_cv_mae": np.nan,
                        "clock_cv_spearman": np.nan,
                        "available": False,
                        "estimable": False,
                        "reason": reason,
                        "n_used": 0,
                        "method": "mouse_tissue_clock_contrast",
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "evidence_level": 0,
                    }
                )
            continue

        for treated_arm, control_arm in contrasts:
            treated = sub.loc[sub[arm_col] == treated_arm, "predicted_age_mouse"].dropna().tolist()
            control = sub.loc[sub[arm_col] == control_arm, "predicted_age_mouse"].dropna().tolist()
            if len(treated) < int(min_samples_per_group) or len(control) < int(min_samples_per_group):
                rows.append(
                    {
                        "tissue": tissue,
                        "contrast": f"{treated_arm}_vs_{control_arm}",
                        "treated_arm": treated_arm,
                        "control_arm": control_arm,
                        "n_reference": int(len(ref)),
                        "n_reference_ages": int(ref[age_col].nunique()),
                        "n_treated": int(len(treated)),
                        "n_control": int(len(control)),
                        "mean_effect": np.nan,
                        "permutation_p_value": np.nan,
                        "clock_cv_mae": float(metrics.get("MAE", np.nan)),
                        "clock_cv_spearman": float(metrics.get("spearman_r", np.nan)),
                        "available": False,
                        "estimable": False,
                        "reason": (
                            f"Too few samples for {treated_arm} vs {control_arm} in {tissue} "
                            f"(n_treated={len(treated)}, n_control={len(control)})."
                        ),
                        "n_used": int(len(treated) + len(control)),
                        "method": "mouse_tissue_clock_contrast",
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "evidence_level": 0,
                    }
                )
                continue

            res = _effect_with_uncertainty(
                treated,
                control,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                random_state=random_state,
            )
            rows.append(
                {
                    "tissue": tissue,
                    "contrast": f"{treated_arm}_vs_{control_arm}",
                    "treated_arm": treated_arm,
                    "control_arm": control_arm,
                    "n_reference": int(len(ref)),
                    "n_reference_ages": int(ref[age_col].nunique()),
                    "n_treated": int(res["n_treated"]),
                    "n_control": int(res["n_control"]),
                    "mean_effect": float(res["mean_effect"]),
                    "standardized_effect": float(res["standardized_effect"]),
                    "standardized_ci_low": res["standardized_ci_low"],
                    "standardized_ci_high": res["standardized_ci_high"],
                    "standardized_effect_method": res["standardized_effect_method"],
                    "permutation_p_value": res["permutation_p_value"],
                    "clock_cv_mae": float(metrics.get("MAE", np.nan)),
                    "clock_cv_spearman": float(metrics.get("spearman_r", np.nan)),
                    "available": True,
                    "estimable": True,
                    "reason": "",
                    "n_used": int(res["n_treated"] + res["n_control"]),
                    "method": "mouse_tissue_clock_contrast",
                    "ci_low": res["ci_low"],
                    "ci_high": res["ci_high"],
                    "evidence_level": 1,
                }
            )

    result = annotate_reason_fields(pd.DataFrame(rows))
    if return_sample_outcomes:
        outcomes = (
            pd.concat(outcome_frames, ignore_index=True)
            if outcome_frames
            else pd.DataFrame()
        )
        return result, outcomes
    return result


def summarize_mouse_exosome_signatures(mouse_effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if mouse_effects is None or mouse_effects.empty:
        return annotate_reason_fields(pd.DataFrame(
            [
                {
                    "contrast": "NA",
                    "available": False,
                    "estimable": False,
                    "reason": "Mouse exosome effect table unavailable.",
                    "n_used": 0,
                    "method": "mouse_exosome_signature_summary",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }
            ]
        ))

    for contrast, sub in mouse_effects.groupby("contrast"):
        ok = sub.loc[sub["estimable"].astype(bool)].copy()
        if ok.empty:
            rows.append(
                {
                    "contrast": contrast,
                    "n_tissues_estimable": 0,
                    "mean_effect": np.nan,
                    "median_effect": np.nan,
                    "fraction_negative": np.nan,
                    "available": False,
                    "estimable": False,
                    "reason": "No tissues passed mouse exosome estimability filters.",
                    "n_used": 0,
                    "method": "mouse_exosome_signature_summary",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }
            )
            continue

        effects = ok["mean_effect"].astype(float)
        rows.append(
            {
                "contrast": contrast,
                "n_tissues_estimable": int(len(ok)),
                "mean_effect": float(effects.mean()),
                "median_effect": float(effects.median()),
                "fraction_negative": float((effects < 0).mean()),
                "available": True,
                "estimable": True,
                "reason": "",
                "n_used": int(len(ok)),
                "method": "mouse_exosome_signature_summary",
                "ci_low": float(ok["ci_low"].astype(float).median()),
                "ci_high": float(ok["ci_high"].astype(float).median()),
                "evidence_level": 1,
            }
        )

    return annotate_reason_fields(pd.DataFrame(rows))


def _alignment_frame(
    primate_series: pd.Series,
    external_series: pd.Series,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "macaque_effect": pd.to_numeric(primate_series, errors="coerce"),
            "external_effect": pd.to_numeric(external_series, errors="coerce"),
        }
    ).dropna()
    if df.empty:
        return df

    prim_std = float(df["macaque_effect"].std(ddof=0))
    ext_std = float(df["external_effect"].std(ddof=0))
    df["macaque_z"] = (
        (df["macaque_effect"] - df["macaque_effect"].mean()) / prim_std if prim_std > 0 else 0.0
    )
    df["external_z"] = (
        (df["external_effect"] - df["external_effect"].mean()) / ext_std if ext_std > 0 else 0.0
    )
    df["signed_concordance"] = (
        np.sign(df["macaque_effect"]).replace(0, np.nan)
        == np.sign(df["external_effect"]).replace(0, np.nan)
    ).fillna(False).astype(int)
    df["macaque_rank"] = df["macaque_effect"].rank(method="average")
    df["external_rank"] = df["external_effect"].rank(method="average")
    denom = max(1.0, float(len(df) - 1))
    df["rank_concordance"] = 1.0 - (df["macaque_rank"] - df["external_rank"]).abs() / denom
    df["standardized_effect_similarity"] = 1.0 / (
        1.0 + (df["macaque_z"] - df["external_z"]).abs()
    )
    df["residual_component"] = (df["macaque_z"] - df["external_z"]).abs()
    return df


_TISSUE_MAPPING_COLUMNS = {
    "mouse_tissue",
    "primate_tissue",
    "compatibility_tier",
    "include_in_primary",
    "source_status",
    "mapping_note",
}
_ALLOWED_TISSUE_TIERS = {"exact", "organ_compatible", "weak_context", "excluded"}


def load_cross_species_tissue_mapping(source: Path | str | pd.DataFrame) -> pd.DataFrame:
    """Load and validate the explicit mouse-to-macaque anatomical contract."""
    mapping = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    missing = sorted(_TISSUE_MAPPING_COLUMNS.difference(mapping.columns))
    if missing:
        raise ValueError(f"Cross-species tissue mapping is missing columns: {missing}")

    mapping = mapping.loc[:, sorted(_TISSUE_MAPPING_COLUMNS)].copy()
    for column in ("mouse_tissue", "primate_tissue", "compatibility_tier"):
        mapping[column] = mapping[column].astype(str).str.strip()
        if mapping[column].eq("").any():
            raise ValueError(f"Cross-species tissue mapping contains an empty {column}.")
    mapping["compatibility_tier"] = mapping["compatibility_tier"].str.lower()
    invalid_tiers = sorted(set(mapping["compatibility_tier"]) - _ALLOWED_TISSUE_TIERS)
    if invalid_tiers:
        raise ValueError(f"Unsupported cross-species tissue compatibility tiers: {invalid_tiers}")

    bool_tokens = mapping["include_in_primary"].astype(str).str.strip().str.lower()
    invalid_bool = sorted(set(bool_tokens) - {"true", "false", "1", "0"})
    if invalid_bool:
        raise ValueError(f"Invalid include_in_primary values: {invalid_bool}")
    mapping["include_in_primary"] = bool_tokens.isin({"true", "1"})
    invalid_primary = mapping["include_in_primary"] & ~mapping["compatibility_tier"].isin(
        {"exact", "organ_compatible"}
    )
    if invalid_primary.any():
        bad = mapping.loc[invalid_primary, "mouse_tissue"].tolist()
        raise ValueError(
            "Weak-context or excluded mappings cannot enter the primary summary: "
            f"{bad}"
        )

    mapping["mouse_tissue_key"] = mapping["mouse_tissue"].map(_normalize_token)
    if mapping["mouse_tissue_key"].duplicated().any():
        duplicates = mapping.loc[
            mapping["mouse_tissue_key"].duplicated(keep=False), "mouse_tissue"
        ].tolist()
        raise ValueError(f"Duplicate normalized mouse tissue mappings: {duplicates}")
    if mapping["primate_tissue"].duplicated().any():
        duplicates = mapping.loc[
            mapping["primate_tissue"].duplicated(keep=False), "primate_tissue"
        ].tolist()
        raise ValueError(f"Multiple mouse tissues map to the same primate tissue: {duplicates}")
    return mapping.reset_index(drop=True)


def _response_components(frame: pd.DataFrame) -> Dict[str, float]:
    macaque = frame["macaque_standardized_effect"].to_numpy(dtype=float)
    mouse = frame["mouse_standardized_effect"].to_numpy(dtype=float)
    macaque_norm = float(np.linalg.norm(macaque))
    mouse_norm = float(np.linalg.norm(mouse))
    if macaque_norm <= 0 or mouse_norm <= 0:
        return {
            "cosine_similarity": np.nan,
            "relative_response_norm": np.nan,
            "aligned_response_coefficient": np.nan,
        }
    cosine = float(np.dot(macaque, mouse) / (macaque_norm * mouse_norm))
    norm_ratio = float(mouse_norm / macaque_norm)
    return {
        "cosine_similarity": cosine,
        "relative_response_norm": norm_ratio,
        "aligned_response_coefficient": float(cosine * norm_ratio),
    }


def _standardized_effects_by_mapping(
    primate: pd.DataFrame,
    mouse: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    primate_group_col: str,
    primate_outcome_col: str,
    primate_tissue_col: str,
    primate_treated: str,
    primate_control: str,
    mouse_arm_col: str,
    mouse_outcome_col: str,
    mouse_tissue_col: str,
    mouse_treated: str,
    mouse_control: str,
    min_per_group: int,
) -> pd.DataFrame:
    rows = []
    for map_row in mapping.itertuples(index=False):
        primate_sub = primate.loc[
            primate[primate_tissue_col].astype(str).eq(str(map_row.primate_tissue))
            & primate[primate_group_col].astype(str).isin([primate_treated, primate_control])
        ]
        mouse_sub = mouse.loc[
            mouse[mouse_tissue_col].map(_normalize_token).eq(str(map_row.mouse_tissue_key))
            & mouse[mouse_arm_col].astype(str).isin([mouse_treated, mouse_control])
        ]
        primate_treated_values = primate_sub.loc[
            primate_sub[primate_group_col].astype(str).eq(primate_treated), primate_outcome_col
        ]
        primate_control_values = primate_sub.loc[
            primate_sub[primate_group_col].astype(str).eq(primate_control), primate_outcome_col
        ]
        mouse_treated_values = mouse_sub.loc[
            mouse_sub[mouse_arm_col].astype(str).eq(mouse_treated), mouse_outcome_col
        ]
        mouse_control_values = mouse_sub.loc[
            mouse_sub[mouse_arm_col].astype(str).eq(mouse_control), mouse_outcome_col
        ]
        counts = [
            primate_treated_values.notna().sum(),
            primate_control_values.notna().sum(),
            mouse_treated_values.notna().sum(),
            mouse_control_values.notna().sum(),
        ]
        if min(counts) < int(min_per_group):
            continue
        macaque_g = hedges_g(primate_treated_values, primate_control_values)
        mouse_g = hedges_g(mouse_treated_values, mouse_control_values)
        if not np.isfinite(macaque_g) or not np.isfinite(mouse_g):
            continue
        rows.append(
            {
                "mouse_tissue": str(map_row.mouse_tissue),
                "primate_tissue": str(map_row.primate_tissue),
                "compatibility_tier": str(map_row.compatibility_tier),
                "include_in_primary": bool(map_row.include_in_primary),
                "macaque_standardized_effect": float(macaque_g),
                "mouse_standardized_effect": float(mouse_g),
                "signed_concordance": int(np.sign(macaque_g) == np.sign(mouse_g)),
                "n_macaque_treated": int(counts[0]),
                "n_macaque_control": int(counts[1]),
                "n_mouse_treated": int(counts[2]),
                "n_mouse_control": int(counts[3]),
                "mapping_source_status": str(map_row.source_status),
                "mapping_note": str(map_row.mapping_note),
            }
        )
    return pd.DataFrame(rows)


def _resample_primate_animals(
    frame: pd.DataFrame,
    *,
    animal_col: str,
    group_col: str,
    sex_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    animal_meta = frame[[animal_col, group_col, sex_col]].drop_duplicates()
    if animal_meta[animal_col].duplicated().any():
        raise ValueError("Each macaque animal must have one treatment group and sex.")
    pieces = []
    for (group, sex), sub in animal_meta.groupby([group_col, sex_col], observed=True, sort=True):
        animals = sub[animal_col].astype(str).to_numpy()
        for draw_index, animal in enumerate(rng.choice(animals, size=len(animals), replace=True)):
            selected = frame.loc[frame[animal_col].astype(str).eq(str(animal))].copy()
            selected[animal_col] = f"bootstrap_{group}_{sex}_{draw_index}"
            pieces.append(selected)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()


def _resample_mouse_samples(
    frame: pd.DataFrame,
    *,
    tissue_col: str,
    arm_col: str,
    sex_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pieces = []
    strata = [tissue_col, arm_col] + ([sex_col] if sex_col in frame.columns else [])
    for _, sub in frame.groupby(strata, observed=True, sort=True):
        index = rng.choice(sub.index.to_numpy(), size=len(sub), replace=True)
        pieces.append(sub.loc[index].copy())
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()


def _permute_primate_groups(
    frame: pd.DataFrame,
    *,
    animal_col: str,
    group_col: str,
    sex_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    animal_meta = frame[[animal_col, group_col, sex_col]].drop_duplicates()
    if animal_meta[animal_col].duplicated().any():
        raise ValueError("Each macaque animal must have one treatment group and sex.")
    permuted = animal_meta.copy()
    for _, index in permuted.groupby(sex_col, observed=True, sort=True).groups.items():
        permuted.loc[index, group_col] = rng.permutation(
            permuted.loc[index, group_col].to_numpy()
        )
    group_map = permuted.set_index(animal_col)[group_col]
    out = frame.copy()
    out[group_col] = out[animal_col].map(group_map)
    return out


def _permute_mouse_arms(
    frame: pd.DataFrame,
    *,
    tissue_col: str,
    arm_col: str,
    sex_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = frame.copy()
    strata = [tissue_col] + ([sex_col] if sex_col in out.columns else [])
    for _, index in out.groupby(strata, observed=True, sort=True).groups.items():
        out.loc[index, arm_col] = rng.permutation(out.loc[index, arm_col].to_numpy())
    return out


def compute_cross_species_response_alignment(
    primate_outcomes: pd.DataFrame,
    mouse_outcomes: pd.DataFrame,
    tissue_mapping: Path | str | pd.DataFrame,
    *,
    primate_group_col: str = "group",
    primate_outcome_col: str = "delta_age",
    primate_tissue_col: str = "tissue",
    primate_animal_col: str = "animal_id",
    primate_sex_col: str = "sex",
    primate_treated: str = "O_GES",
    primate_control: str = "O_V",
    mouse_arm_col: str = "arm",
    mouse_outcome_col: str = "predicted_age_mouse",
    mouse_tissue_col: str = "tissue",
    mouse_sex_col: str = "sex",
    mouse_treated: str = "GES",
    mouse_control: str = "Veh",
    min_per_group: int = 2,
    min_common_tissues: int = 3,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify non-causal cross-species response alignment on Hedges-g effects."""
    method = "cross_species_hedges_g_response_alignment"
    required_primate = {
        primate_group_col,
        primate_outcome_col,
        primate_tissue_col,
        primate_animal_col,
        primate_sex_col,
    }
    required_mouse = {mouse_arm_col, mouse_outcome_col, mouse_tissue_col}
    missing = sorted(required_primate.difference(primate_outcomes.columns)) + sorted(
        required_mouse.difference(mouse_outcomes.columns)
    )
    if missing or primate_outcomes.empty or mouse_outcomes.empty:
        reason = (
            f"Response alignment inputs unavailable or missing columns: {missing}."
        )
        by_tissue_stub = annotate_reason_fields(
            pd.DataFrame(
                [{
                    "mouse_tissue": "NA",
                    "primate_tissue": "NA",
                    "compatibility_tier": "NA",
                    "include_in_primary": False,
                    "macaque_standardized_effect": np.nan,
                    "mouse_standardized_effect": np.nan,
                    "available": False,
                    "estimable": False,
                    "reason": reason,
                    "n_used": 0,
                    "method": method,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }]
            )
        )
        summary_stub = annotate_reason_fields(
            pd.DataFrame(
                [{
                    "contrast_pair": "macaque_SRC_vs_vehicle__mouse_SRC_Exo_vs_vehicle",
                    "n_common_tissues": 0,
                    "cosine_similarity": np.nan,
                    "relative_response_norm": np.nan,
                    "aligned_response_coefficient": np.nan,
                    "n_bootstrap_requested": int(n_bootstrap),
                    "n_bootstrap_valid": 0,
                    "n_permutations_requested": int(n_permutations),
                    "n_permutations_valid": 0,
                    "available": False,
                    "estimable": False,
                    "reason": reason,
                    "n_used": 0,
                    "method": method,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }]
            )
        )
        return by_tissue_stub, summary_stub

    primate = primate_outcomes.copy()
    mouse = mouse_outcomes.copy()
    primate = primate.loc[
        primate[primate_group_col].astype(str).isin([primate_treated, primate_control])
    ].copy()
    mouse = mouse.loc[
        mouse[mouse_arm_col].astype(str).isin([mouse_treated, mouse_control])
    ].copy()
    duplicate = primate.duplicated([primate_animal_col, primate_tissue_col], keep=False)
    if duplicate.any():
        raise ValueError("Macaque response alignment requires one row per animal and tissue.")
    mapping = load_cross_species_tissue_mapping(tissue_mapping)
    mapping = mapping.loc[mapping["include_in_primary"].astype(bool)].copy()
    by_tissue = _standardized_effects_by_mapping(
        primate,
        mouse,
        mapping,
        primate_group_col=primate_group_col,
        primate_outcome_col=primate_outcome_col,
        primate_tissue_col=primate_tissue_col,
        primate_treated=primate_treated,
        primate_control=primate_control,
        mouse_arm_col=mouse_arm_col,
        mouse_outcome_col=mouse_outcome_col,
        mouse_tissue_col=mouse_tissue_col,
        mouse_treated=mouse_treated,
        mouse_control=mouse_control,
        min_per_group=min_per_group,
    )
    if len(by_tissue) < int(min_common_tissues):
        reason = (
            f"Only {len(by_tissue)} primary-compatible tissue pairs have finite standardized "
            f"effects (< {int(min_common_tissues)} required)."
        )
        summary = annotate_reason_fields(
            pd.DataFrame(
                [{
                    "n_common_tissues": int(len(by_tissue)),
                    "available": not by_tissue.empty,
                    "estimable": False,
                    "reason": reason,
                    "n_used": int(len(by_tissue)),
                    "method": method,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }]
            )
        )
        if by_tissue.empty:
            by_tissue = pd.DataFrame(
                [{
                    "mouse_tissue": "NA",
                    "primate_tissue": "NA",
                    "compatibility_tier": "NA",
                    "include_in_primary": False,
                    "macaque_standardized_effect": np.nan,
                    "mouse_standardized_effect": np.nan,
                    "available": False,
                    "estimable": False,
                    "reason": reason,
                    "n_used": 0,
                    "method": "cross_species_hedges_g_response_pair",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }]
            )
        return annotate_reason_fields(by_tissue), summary

    observed = _response_components(by_tissue)
    rng = np.random.default_rng(random_state)
    bootstrap_rows = []
    for _ in range(int(n_bootstrap)):
        primate_boot = _resample_primate_animals(
            primate,
            animal_col=primate_animal_col,
            group_col=primate_group_col,
            sex_col=primate_sex_col,
            rng=rng,
        )
        mouse_boot = _resample_mouse_samples(
            mouse,
            tissue_col=mouse_tissue_col,
            arm_col=mouse_arm_col,
            sex_col=mouse_sex_col,
            rng=rng,
        )
        boot_effects = _standardized_effects_by_mapping(
            primate_boot,
            mouse_boot,
            mapping,
            primate_group_col=primate_group_col,
            primate_outcome_col=primate_outcome_col,
            primate_tissue_col=primate_tissue_col,
            primate_treated=primate_treated,
            primate_control=primate_control,
            mouse_arm_col=mouse_arm_col,
            mouse_outcome_col=mouse_outcome_col,
            mouse_tissue_col=mouse_tissue_col,
            mouse_treated=mouse_treated,
            mouse_control=mouse_control,
            min_per_group=min_per_group,
        )
        if len(boot_effects) == len(by_tissue):
            components = _response_components(boot_effects)
            if all(np.isfinite(value) for value in components.values()):
                bootstrap_rows.append(components)

    bootstrap = pd.DataFrame(bootstrap_rows)
    intervals: Dict[str, float] = {}
    for metric in observed:
        values = bootstrap.get(metric, pd.Series(dtype=float)).dropna()
        intervals[f"{metric}_ci_low"] = (
            float(values.quantile(0.025)) if not values.empty else np.nan
        )
        intervals[f"{metric}_ci_high"] = (
            float(values.quantile(0.975)) if not values.empty else np.nan
        )
        intervals[f"{metric}_sign_stability"] = (
            float((np.sign(values) == np.sign(observed[metric])).mean())
            if not values.empty and observed[metric] != 0
            else np.nan
        )

    leave_one_out = []
    for tissue in by_tissue["primate_tissue"]:
        reduced = by_tissue.loc[~by_tissue["primate_tissue"].eq(tissue)]
        if len(reduced) >= 2:
            row = {"excluded_tissue": tissue, **_response_components(reduced)}
            leave_one_out.append(row)
    loo = pd.DataFrame(leave_one_out)

    permutation_rows = []
    for _ in range(int(n_permutations)):
        primate_permuted = _permute_primate_groups(
            primate,
            animal_col=primate_animal_col,
            group_col=primate_group_col,
            sex_col=primate_sex_col,
            rng=rng,
        )
        mouse_permuted = _permute_mouse_arms(
            mouse,
            tissue_col=mouse_tissue_col,
            arm_col=mouse_arm_col,
            sex_col=mouse_sex_col,
            rng=rng,
        )
        permuted_effects = _standardized_effects_by_mapping(
            primate_permuted,
            mouse_permuted,
            mapping,
            primate_group_col=primate_group_col,
            primate_outcome_col=primate_outcome_col,
            primate_tissue_col=primate_tissue_col,
            primate_treated=primate_treated,
            primate_control=primate_control,
            mouse_arm_col=mouse_arm_col,
            mouse_outcome_col=mouse_outcome_col,
            mouse_tissue_col=mouse_tissue_col,
            mouse_treated=mouse_treated,
            mouse_control=mouse_control,
            min_per_group=min_per_group,
        )
        if len(permuted_effects) == len(by_tissue):
            components = _response_components(permuted_effects)
            if all(np.isfinite(value) for value in components.values()):
                permutation_rows.append(components)
    permutations = pd.DataFrame(permutation_rows)

    def _two_sided_permutation_p(metric: str) -> float:
        values = permutations.get(metric, pd.Series(dtype=float)).dropna()
        if values.empty or not np.isfinite(observed[metric]):
            return np.nan
        exceedances = int((values.abs() >= abs(observed[metric]) - 1e-12).sum())
        return float((exceedances + 1) / (len(values) + 1))

    summary_row: Dict[str, Any] = {
        "contrast_pair": "macaque_SRC_vs_vehicle__mouse_SRC_Exo_vs_vehicle",
        "n_common_tissues": int(len(by_tissue)),
        **observed,
        **intervals,
        "cosine_leave_one_out_min": float(loo["cosine_similarity"].min()) if not loo.empty else np.nan,
        "cosine_leave_one_out_max": float(loo["cosine_similarity"].max()) if not loo.empty else np.nan,
        "aligned_coefficient_leave_one_out_min": (
            float(loo["aligned_response_coefficient"].min()) if not loo.empty else np.nan
        ),
        "aligned_coefficient_leave_one_out_max": (
            float(loo["aligned_response_coefficient"].max()) if not loo.empty else np.nan
        ),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": int(len(bootstrap)),
        "n_permutations_requested": int(n_permutations),
        "n_permutations_valid": int(len(permutations)),
        "cosine_permutation_p_value": _two_sided_permutation_p("cosine_similarity"),
        "aligned_coefficient_permutation_p_value": _two_sided_permutation_p(
            "aligned_response_coefficient"
        ),
        "standardized_effect_method": "hedges_g_pooled_sd",
        "weighting": "equal_tissue",
        "interpretation": (
            "Exploratory cross-species response resemblance; not an exosome-attributable "
            "fraction, causal mediation estimate, or quantitative transfer estimate."
        ),
        "available": True,
        "estimable": True,
        "reason": "",
        "n_used": int(len(by_tissue)),
        "method": method,
        "ci_low": intervals["aligned_response_coefficient_ci_low"],
        "ci_high": intervals["aligned_response_coefficient_ci_high"],
        "evidence_level": 3,
    }
    by_tissue = by_tissue.assign(
        available=True,
        estimable=True,
        reason="",
        n_used=1,
        method="cross_species_hedges_g_response_pair",
        ci_low=np.nan,
        ci_high=np.nan,
        evidence_level=3,
    )
    return annotate_reason_fields(by_tissue), annotate_reason_fields(
        pd.DataFrame([summary_row])
    )


def compute_exosome_alignment_tables(
    primate_effects: pd.DataFrame,
    mouse_effects: pd.DataFrame,
    *,
    tissue_map: Optional[Mapping[str, str]] = None,
    tissue_mapping: Path | str | pd.DataFrame | None = None,
    contrasts: Sequence[str] = ("GES_vs_Veh", "WT_vs_Veh"),
    min_common_tissues: int = 3,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_tissue_rows = []
    summary_rows = []

    if tissue_mapping is not None:
        mapping_contract = load_cross_species_tissue_mapping(tissue_mapping)
    elif tissue_map:
        # Backward-compatible conversion of legacy dictionaries. Exact token
        # matches can enter the primary set; brain-to-hippocampus is always a
        # weak-context sensitivity; other explicit organ mappings remain
        # organ-compatible rather than anatomically exact.
        legacy_rows = []
        for mouse_tissue, primate_tissue in tissue_map.items():
            mouse_key = _normalize_token(mouse_tissue)
            primate_key = _normalize_token(primate_tissue)
            is_brain_hippocampus = mouse_key == "brain" and primate_key == "hippocampus"
            tier = (
                "weak_context"
                if is_brain_hippocampus
                else ("exact" if mouse_key == primate_key else "organ_compatible")
            )
            legacy_rows.append(
                {
                    "mouse_tissue": mouse_tissue,
                    "primate_tissue": primate_tissue,
                    "compatibility_tier": tier,
                    "include_in_primary": not is_brain_hippocampus,
                    "source_status": "legacy_dictionary",
                    "mapping_note": "Legacy mapping converted to the tiered contract.",
                }
            )
        mapping_contract = load_cross_species_tissue_mapping(pd.DataFrame(legacy_rows))
    else:
        mapping_contract = pd.DataFrame(columns=[*_TISSUE_MAPPING_COLUMNS, "mouse_tissue_key"])

    if primate_effects is None or primate_effects.empty or mouse_effects is None or mouse_effects.empty:
        stub = pd.DataFrame(
            [
                {
                    "contrast": "NA",
                    "mouse_tissue": "NA",
                    "primate_tissue": "NA",
                    "available": False,
                    "estimable": False,
                    "reason": "Primate and/or mouse effect tables unavailable for exosome alignment.",
                    "n_used": 0,
                    "method": "cross_species_effect_alignment",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }
            ]
        )
        summary_stub = pd.DataFrame(
            [
                {
                    "contrast": "NA",
                    "n_common_tissues": 0,
                    "available": False,
                    "estimable": False,
                    "reason": "Primate and/or mouse effect tables unavailable for exosome alignment.",
                    "n_used": 0,
                    "method": "cross_species_effect_alignment_summary",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }
            ]
        )
        return annotate_reason_fields(stub), annotate_reason_fields(summary_stub)

    prim = primate_effects.copy()
    if "tissue" in prim.columns:
        prim = prim.set_index("tissue")
    prim.index = prim.index.map(lambda x: _resolve_tissue_mapping(x, None))
    prim.index.name = "primate_tissue"

    mouse_estimable = (
        mouse_effects["estimable"].astype(bool)
        if "estimable" in mouse_effects.columns
        else pd.Series(True, index=mouse_effects.index, dtype=bool)
    )
    mouse_unavailable = not mouse_estimable.any()
    upstream_reason = ""
    upstream_reason_code = ""
    upstream_missing_key = ""
    if mouse_unavailable:
        unavailable_rows = mouse_effects.loc[~mouse_estimable]
        if not unavailable_rows.empty:
            first = unavailable_rows.iloc[0]
            upstream_reason = str(first.get("reason", "")).strip()
            upstream_reason_code = str(first.get("reason_code", "")).strip()
            upstream_missing_key = str(first.get("missing_author_key", "")).strip()

    for contrast in contrasts:
        sub = mouse_effects.loc[
            (mouse_effects["contrast"] == contrast)
            & mouse_effects["estimable"].astype(bool)
        ].copy()
        if sub.empty:
            reason = upstream_reason or f"No estimable mouse effects available for contrast {contrast}."
            reason_fields = {
                "reason_code": upstream_reason_code,
                "missing_author_key": upstream_missing_key,
            }
            summary_rows.append(
                {
                    "contrast": contrast,
                    "n_common_tissues": 0,
                    "fraction_signed_concordant": np.nan,
                    "mean_rank_concordance": np.nan,
                    "mean_standardized_effect_similarity": np.nan,
                    "residual_component": np.nan,
                    "spearman_rho": np.nan,
                    "pearson_r": np.nan,
                    "preferred_alignment": "",
                    "available": False,
                    "estimable": False,
                    "reason": reason,
                    "n_used": 0,
                    "method": "cross_species_effect_alignment_summary",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "permutation_p_value": np.nan,
                    "evidence_level": 0,
                    **reason_fields,
                }
            )
            by_tissue_rows.append(
                {
                    "contrast": contrast,
                    "mouse_tissue": "NA",
                    "primate_tissue": "NA",
                    "macaque_effect": np.nan,
                    "mouse_effect": np.nan,
                    "signed_concordance": np.nan,
                    "rank_concordance": np.nan,
                    "standardized_effect_similarity": np.nan,
                    "residual_component": np.nan,
                    "global_spearman_rho": np.nan,
                    "global_pearson_r": np.nan,
                    "mouse_permutation_p_value": np.nan,
                    "available": False,
                    "estimable": False,
                    "reason": reason,
                    "n_used": 0,
                    "method": "cross_species_effect_alignment",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                    **reason_fields,
                }
            )
            continue

        sub["mouse_tissue_key"] = sub["tissue"].map(_normalize_token)
        mapped = sub.merge(
            mapping_contract,
            on="mouse_tissue_key",
            how="inner",
            suffixes=("", "_mapping"),
        )
        join_all = mapped.merge(
            prim.reset_index(),
            on="primate_tissue",
            how="inner",
            suffixes=("_mouse", "_macaque"),
        )
        join = join_all.loc[join_all["include_in_primary"].astype(bool)].copy()
        if join.empty:
            join = pd.DataFrame(columns=["tissue", "primate_tissue", "mouse_effect", "macaque_effect"])
        else:
            if "mean_effect_mouse" in join.columns:
                join = join.rename(columns={"mean_effect_mouse": "mouse_effect"})
            elif "mean_effect" in join.columns:
                join["mouse_effect"] = pd.to_numeric(join["mean_effect"], errors="coerce")

            if "mean_effect_macaque" in join.columns:
                join = join.rename(columns={"mean_effect_macaque": "macaque_effect"})
            elif "mean_effect" in prim.columns:
                join["macaque_effect"] = pd.to_numeric(join["mean_effect"], errors="coerce")

        primate_series = pd.Series(
            pd.to_numeric(join.get("macaque_effect", pd.Series(dtype=float)), errors="coerce").values,
            index=join.get("primate_tissue", pd.Series(dtype=str)).astype(str).values,
        )
        mouse_series = pd.Series(
            pd.to_numeric(join.get("mouse_effect", pd.Series(dtype=float)), errors="coerce").values,
            index=join.get("primate_tissue", pd.Series(dtype=str)).astype(str).values,
        )
        aligned = _alignment_frame(primate_series, mouse_series)

        if len(aligned) < int(min_common_tissues):
            reason = (
                f"Only {len(aligned)} common tissues (< {int(min_common_tissues)} required) "
                f"for contrast {contrast}."
            )
            summary_rows.append(
                {
                    "contrast": contrast,
                    "n_common_tissues": int(len(aligned)),
                    "fraction_signed_concordant": np.nan,
                    "mean_rank_concordance": np.nan,
                    "mean_standardized_effect_similarity": np.nan,
                    "residual_component": np.nan,
                    "spearman_rho": np.nan,
                    "pearson_r": np.nan,
                    "preferred_alignment": "",
                    "available": False,
                    "estimable": False,
                    "reason": reason,
                    "n_used": int(len(aligned)),
                    "method": "cross_species_effect_alignment_summary",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "permutation_p_value": np.nan,
                    "evidence_level": 0,
                }
            )
            by_tissue_rows.append(
                {
                    "contrast": contrast,
                    "mouse_tissue": "NA",
                    "primate_tissue": "NA",
                    "macaque_effect": np.nan,
                    "mouse_effect": np.nan,
                    "signed_concordance": np.nan,
                    "rank_concordance": np.nan,
                    "standardized_effect_similarity": np.nan,
                    "residual_component": np.nan,
                    "global_spearman_rho": np.nan,
                    "global_pearson_r": np.nan,
                    "mouse_permutation_p_value": np.nan,
                    "available": False,
                    "estimable": False,
                    "reason": reason,
                    "n_used": int(len(aligned)),
                    "method": "cross_species_effect_alignment",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }
            )
            continue

        spearman_rho = float(aligned["macaque_effect"].corr(aligned["external_effect"], method="spearman"))
        pearson_r = float(aligned["macaque_effect"].corr(aligned["external_effect"], method="pearson"))
        observed_similarity = float(aligned["standardized_effect_similarity"].mean())

        rng = np.random.default_rng(random_state)
        boot = []
        arr = aligned[["macaque_effect", "external_effect"]].to_numpy(dtype=float)
        idx = np.arange(len(arr))
        for _ in range(int(n_bootstrap)):
            bidx = rng.choice(idx, size=len(idx), replace=True)
            frame = _alignment_frame(pd.Series(arr[bidx, 0]), pd.Series(arr[bidx, 1]))
            if not frame.empty:
                boot.append(float(frame["standardized_effect_similarity"].mean()))
        ci_low = float(np.percentile(boot, 2.5)) if boot else np.nan
        ci_high = float(np.percentile(boot, 97.5)) if boot else np.nan

        perm = []
        for _ in range(int(n_permutations)):
            shuffled = rng.permutation(arr[:, 1])
            frame = _alignment_frame(pd.Series(arr[:, 0]), pd.Series(shuffled))
            if not frame.empty:
                perm.append(float(frame["standardized_effect_similarity"].mean()))
        perm_p = float(np.mean(np.asarray(perm, dtype=float) >= observed_similarity)) if perm else np.nan

        for tissue_name, row in aligned.iterrows():
            mouse_row = join.loc[join["primate_tissue"] == tissue_name].iloc[0]
            by_tissue_rows.append(
                {
                    "contrast": contrast,
                    "mouse_tissue": str(mouse_row["tissue"]),
                    "primate_tissue": tissue_name,
                    "macaque_effect": float(row["macaque_effect"]),
                    "mouse_effect": float(row["external_effect"]),
                    "signed_concordance": int(row["signed_concordance"]),
                    "rank_concordance": float(row["rank_concordance"]),
                    "standardized_effect_similarity": float(row["standardized_effect_similarity"]),
                    "residual_component": float(row["residual_component"]),
                    "global_spearman_rho": spearman_rho,
                    "global_pearson_r": pearson_r,
                    "mouse_permutation_p_value": float(mouse_row.get("permutation_p_value", np.nan)),
                    "compatibility_tier": str(mouse_row["compatibility_tier"]),
                    "include_in_primary": True,
                    "analysis_scope": "primary_organ_compatible",
                    "mapping_source_status": str(mouse_row["source_status"]),
                    "mapping_note": str(mouse_row["mapping_note"]),
                    "available": True,
                    "estimable": True,
                    "reason": "",
                    "n_used": int(len(aligned)),
                    "method": "cross_species_effect_alignment",
                    "ci_low": float(mouse_row.get("ci_low", np.nan)),
                    "ci_high": float(mouse_row.get("ci_high", np.nan)),
                    "evidence_level": 3,
                }
            )

        weak_join = join_all.loc[~join_all["include_in_primary"].astype(bool)].copy()
        if not weak_join.empty:
            if "mean_effect_mouse" in weak_join.columns:
                weak_join = weak_join.rename(columns={"mean_effect_mouse": "mouse_effect"})
            elif "mean_effect" in weak_join.columns:
                weak_join["mouse_effect"] = pd.to_numeric(weak_join["mean_effect"], errors="coerce")
            if "mean_effect_macaque" in weak_join.columns:
                weak_join = weak_join.rename(columns={"mean_effect_macaque": "macaque_effect"})
            for _, weak_row in weak_join.iterrows():
                macaque_effect = float(weak_row["macaque_effect"])
                mouse_effect = float(weak_row["mouse_effect"])
                by_tissue_rows.append(
                    {
                        "contrast": contrast,
                        "mouse_tissue": str(weak_row["tissue"]),
                        "primate_tissue": str(weak_row["primate_tissue"]),
                        "macaque_effect": macaque_effect,
                        "mouse_effect": mouse_effect,
                        "signed_concordance": int(
                            np.sign(macaque_effect) != 0
                            and np.sign(macaque_effect) == np.sign(mouse_effect)
                        ),
                        "rank_concordance": np.nan,
                        "standardized_effect_similarity": np.nan,
                        "residual_component": np.nan,
                        "global_spearman_rho": np.nan,
                        "global_pearson_r": np.nan,
                        "mouse_permutation_p_value": float(
                            weak_row.get("permutation_p_value", np.nan)
                        ),
                        "compatibility_tier": str(weak_row["compatibility_tier"]),
                        "include_in_primary": False,
                        "analysis_scope": "weak_context_sensitivity",
                        "mapping_source_status": str(weak_row["source_status"]),
                        "mapping_note": str(weak_row["mapping_note"]),
                        "available": True,
                        "estimable": True,
                        "reason": (
                            "Reported as a pairwise weak-context sensitivity and excluded "
                            "from the primary cross-species summary."
                        ),
                        "n_used": 1,
                        "method": "cross_species_effect_alignment_sensitivity",
                        "ci_low": float(weak_row.get("ci_low", np.nan)),
                        "ci_high": float(weak_row.get("ci_high", np.nan)),
                        "evidence_level": 3,
                    }
                )

        summary_rows.append(
            {
                "contrast": contrast,
                "n_common_tissues": int(len(aligned)),
                "fraction_signed_concordant": float(aligned["signed_concordance"].mean()),
                "mean_rank_concordance": float(aligned["rank_concordance"].mean()),
                "mean_standardized_effect_similarity": observed_similarity,
                "residual_component": float(aligned["residual_component"].mean()),
                "spearman_rho": spearman_rho,
                "pearson_r": pearson_r,
                "preferred_alignment": "",
                "analysis_scope": "primary_organ_compatible",
                "n_weak_context_tissues": int(
                    (~join_all["include_in_primary"].astype(bool)).sum()
                ),
                "available": True,
                "estimable": True,
                "reason": "",
                "n_used": int(len(aligned)),
                "method": "cross_species_effect_alignment_summary",
                "ci_low": ci_low,
                "ci_high": ci_high,
                "permutation_p_value": perm_p,
                "evidence_level": 3,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        ok = summary_df.loc[summary_df["estimable"].astype(bool)].copy()
        if not ok.empty:
            preferred = str(
                ok.sort_values("mean_standardized_effect_similarity", ascending=False).iloc[0]["contrast"]
            )
            summary_df["preferred_alignment"] = preferred

    return annotate_reason_fields(pd.DataFrame(by_tissue_rows)), annotate_reason_fields(summary_df)


def assign_group_stage_age(
    groups: pd.Series,
    group_age_map: Mapping[str, float],
) -> pd.Series:
    normalized_map = {_normalize_token(k): float(v) for k, v in group_age_map.items()}

    def _map_one(value: Any) -> float:
        key = _normalize_token(value)
        if key in normalized_map:
            return normalized_map[key]
        if key.startswith("y"):
            return normalized_map.get("yc", np.nan)
        if key.startswith("m"):
            return normalized_map.get("mc", np.nan)
        if key.startswith("o"):
            return normalized_map.get("ov", np.nan)
        return np.nan

    return groups.map(_map_one).astype(float)


def summarize_multimodal_concordance(
    transcript_effects: pd.DataFrame,
    methylation_effects: pd.DataFrame,
    *,
    min_common_tissues: int = 3,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    if transcript_effects is None or transcript_effects.empty or methylation_effects is None or methylation_effects.empty:
        return annotate_reason_fields(pd.DataFrame(
            [
                {
                    "available": False,
                    "estimable": False,
                    "reason": "Transcriptomic and/or methylation rejuvenation tables unavailable.",
                    "n_used": 0,
                    "method": "multimodal_concordance",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }
            ]
        ))

    tx = transcript_effects.copy()
    me = methylation_effects.copy()
    if "tissue" in tx.columns:
        tx = tx.set_index("tissue")
    if "tissue" in me.columns:
        me = me.set_index("tissue")

    frame = _alignment_frame(
        tx["effect_median"] if "effect_median" in tx.columns else tx["mean_effect"],
        me["effect_median"] if "effect_median" in me.columns else me["mean_effect"],
    )
    if len(frame) < int(min_common_tissues):
        return annotate_reason_fields(pd.DataFrame(
            [
                {
                    "n_common_tissues": int(len(frame)),
                    "fraction_signed_concordant": np.nan,
                    "mean_rank_concordance": np.nan,
                    "mean_standardized_effect_similarity": np.nan,
                    "residual_component": np.nan,
                    "spearman_rho": np.nan,
                    "pearson_r": np.nan,
                    "available": False,
                    "estimable": False,
                    "reason": (
                        f"Only {len(frame)} common tissues (< {int(min_common_tissues)} required) "
                        "for multimodal concordance."
                    ),
                    "n_used": int(len(frame)),
                    "method": "multimodal_concordance",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "permutation_p_value": np.nan,
                    "evidence_level": 0,
                }
            ]
        ))

    observed_similarity = float(frame["standardized_effect_similarity"].mean())
    rng = np.random.default_rng(random_state)
    boot = []
    arr = frame[["macaque_effect", "external_effect"]].to_numpy(dtype=float)
    idx = np.arange(len(arr))
    for _ in range(int(n_bootstrap)):
        bidx = rng.choice(idx, size=len(idx), replace=True)
        boot_frame = _alignment_frame(pd.Series(arr[bidx, 0]), pd.Series(arr[bidx, 1]))
        if not boot_frame.empty:
            boot.append(float(boot_frame["standardized_effect_similarity"].mean()))
    ci_low = float(np.percentile(boot, 2.5)) if boot else np.nan
    ci_high = float(np.percentile(boot, 97.5)) if boot else np.nan

    perm = []
    for _ in range(int(n_permutations)):
        shuffled = rng.permutation(arr[:, 1])
        perm_frame = _alignment_frame(pd.Series(arr[:, 0]), pd.Series(shuffled))
        if not perm_frame.empty:
            perm.append(float(perm_frame["standardized_effect_similarity"].mean()))
    perm_p = float(np.mean(np.asarray(perm, dtype=float) >= observed_similarity)) if perm else np.nan

    return annotate_reason_fields(pd.DataFrame(
        [
            {
                "n_common_tissues": int(len(frame)),
                "common_tissues": ";".join(frame.index.astype(str)),
                "fraction_signed_concordant": float(frame["signed_concordance"].mean()),
                "mean_rank_concordance": float(frame["rank_concordance"].mean()),
                "mean_standardized_effect_similarity": observed_similarity,
                "residual_component": float(frame["residual_component"].mean()),
                "spearman_rho": float(frame["macaque_effect"].corr(frame["external_effect"], method="spearman")),
                "pearson_r": float(frame["macaque_effect"].corr(frame["external_effect"], method="pearson")),
                "available": True,
                "estimable": True,
                "reason": "",
                "n_used": int(len(frame)),
                "method": "multimodal_concordance",
                "ci_low": ci_low,
                "ci_high": ci_high,
                "permutation_p_value": perm_p,
                "evidence_level": 1,
            }
        ]
    ))


def read_subset_sample_info(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.endswith("sample.info.csv")]
        if not members:
            raise FileNotFoundError(f"No sample.info.csv found in {zip_path.name}")
        with zf.open(members[0]) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            return pd.read_csv(text)


def build_subset_validation_table(
    sample_info: pd.DataFrame,
    *,
    subset_name: str,
    bulk_tissue: str,
    group_map: Mapping[str, str],
    bulk_effects: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = sample_info.copy()
    raw_group = df["group"].astype(str).str.strip() if "group" in df.columns else pd.Series([], dtype=str)
    repo_group = raw_group.map(lambda x: group_map.get(x, x))
    counts = repo_group.value_counts().to_dict()

    bulk_row = None
    if bulk_effects is not None and not bulk_effects.empty:
        tmp = bulk_effects.copy()
        if "tissue" in tmp.columns:
            match = tmp.loc[tmp["tissue"].astype(str) == str(bulk_tissue)]
            if not match.empty:
                bulk_row = match.iloc[0]
        elif bulk_tissue in tmp.index:
            bulk_row = tmp.loc[bulk_tissue]

    row = {
        "subset": subset_name,
        "bulk_tissue": bulk_tissue,
        "n_samples_total": int(len(df)),
        "n_y_c": int(counts.get("Y_C", 0)),
        "n_m_c": int(counts.get("M_C", 0)),
        "n_o_c": int(counts.get("O_C", 0)),
        "n_o_v": int(counts.get("O_V", 0)),
        "n_o_wt": int(counts.get("O_WT", 0)),
        "n_o_ges": int(counts.get("O_GES", 0)),
        "group_distinction_preserved": bool((counts.get("O_C", 0) > 0) and (counts.get("O_V", 0) > 0)),
        "intervention_groups_present": bool(
            counts.get("O_V", 0) > 0 and counts.get("O_WT", 0) > 0 and counts.get("O_GES", 0) > 0
        ),
        "bulk_effect_median": np.nan,
        "bulk_ci_low": np.nan,
        "bulk_ci_high": np.nan,
        "validation_status": "metadata_only",
        "available": True,
        "estimable": False,
        "reason": (
            "Subset sample-sheet validation integrated. Expression-level pseudobulk validation "
            "for zipped single-cell matrices is not yet implemented."
        ),
        "n_used": int(len(df)),
        "method": "subset_sample_info_audit",
        "ci_low": np.nan,
        "ci_high": np.nan,
        "evidence_level": 1,
    }
    if bulk_row is not None:
        row["bulk_effect_median"] = float(
            bulk_row["effect_median"] if "effect_median" in bulk_row else bulk_row.get("mean_effect", np.nan)
        )
        row["bulk_ci_low"] = float(bulk_row.get("ci_low", np.nan))
        row["bulk_ci_high"] = float(bulk_row.get("ci_high", np.nan))
    else:
        row["reason"] = (
            "Subset sample-sheet validation integrated, but no matching bulk tissue effect "
            "was available for direct comparison."
        )
        row["evidence_level"] = 0

    return annotate_reason_fields(pd.DataFrame([row]))
