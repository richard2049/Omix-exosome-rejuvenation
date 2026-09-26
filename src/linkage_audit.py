"""
linkage_audit.py

Utilities to assess whether primate bulk metadata can be linked to plasma
metadata at the animal level, and to build a strict estimability gate for
downstream mediation/decomposition analyses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd

from .reason_codes import (
    INSUFFICIENT_LINKED_ANIMALS,
    PLASMA_ANIMAL_LINKAGE_COLLISION,
    PLASMA_BULK_ANIMAL_LINKAGE_MISSING,
    PLASMA_LINKAGE_CONFIDENCE_MISSING,
    OK,
    missing_author_key_for_reason_code,
)


def _mode_or_first(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    if s.empty:
        return ""
    m = s.mode()
    if m.empty:
        return str(s.iloc[0])
    return str(m.iloc[0])


def _canonical_group_label(x: Any) -> str:
    s = str(x).strip().upper()
    if s.startswith("O_"):
        return s.split("_", 1)[1]
    if s.endswith("_C"):
        return "C"
    if s in {"CTRL", "CONTROL"}:
        return "C"
    return s


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _confirmed_boolean(series: pd.Series, *, column: str) -> pd.Series:
    tokens = series.astype(str).str.strip().str.lower()
    allowed = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    invalid = sorted(set(tokens) - set(allowed))
    if invalid:
        raise ValueError(f"Validated linkage manifest has invalid {column} values: {invalid}")
    return tokens.map(allowed).astype(bool)


def load_validated_plasma_linkage_manifest(
    path: Path | str,
    *,
    expected_sha256: str | None,
    plasma_meta: pd.DataFrame,
    primate_meta: pd.DataFrame,
    sample_col: str = "sample_id",
    animal_col: str = "animal_id",
    group_col: str = "group",
    sex_col: str = "sex",
) -> pd.DataFrame:
    """Validate an explicitly confirmed one-to-one plasma-to-animal key."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ValueError(f"Validated linkage manifest does not exist: {manifest_path.name}")
    expected = str(expected_sha256 or "").strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("A 64-character SHA-256 is required for the validated linkage manifest.")
    observed = _sha256_file(manifest_path)
    if observed != expected:
        raise ValueError(
            "Validated linkage manifest SHA-256 mismatch "
            f"(expected {expected}, observed {observed})."
        )

    manifest = pd.read_csv(manifest_path, dtype=str)
    required = {sample_col, animal_col, "identity_confirmed", "evidence_source"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Validated linkage manifest is missing columns: {missing}")
    manifest = manifest.loc[:, [sample_col, animal_col, "identity_confirmed", "evidence_source"]].copy()
    for column in (sample_col, animal_col, "evidence_source"):
        manifest[column] = manifest[column].astype(str).str.strip()
        if manifest[column].isin({"", "nan", "None", "<NA>"}).any():
            raise ValueError(f"Validated linkage manifest contains empty {column} values.")
    if manifest.empty:
        raise ValueError("Validated linkage manifest contains no links.")
    if manifest[sample_col].duplicated().any():
        raise ValueError("Validated linkage manifest contains duplicate sample_id values.")
    if manifest[animal_col].duplicated().any():
        raise ValueError("Validated linkage manifest is not one-to-one: duplicate animal_id values.")
    confirmed = _confirmed_boolean(manifest["identity_confirmed"], column="identity_confirmed")
    if not confirmed.all():
        raise ValueError("Every validated linkage row must have identity_confirmed=true.")

    required_plasma = {sample_col, group_col, sex_col}
    required_primate = {animal_col, group_col, sex_col}
    if not required_plasma.issubset(plasma_meta.columns):
        raise ValueError(f"Plasma metadata is missing linkage-check columns: {sorted(required_plasma - set(plasma_meta.columns))}")
    if not required_primate.issubset(primate_meta.columns):
        raise ValueError(f"Primate metadata is missing linkage-check columns: {sorted(required_primate - set(primate_meta.columns))}")

    plasma_units = plasma_meta.loc[:, [sample_col, group_col, sex_col]].copy()
    if plasma_units[sample_col].duplicated().any():
        raise ValueError("Plasma metadata contains duplicate sample_id values.")
    primate_units = primate_meta.loc[:, [animal_col, group_col, sex_col]].drop_duplicates().copy()
    inconsistent = (
        primate_units.groupby(animal_col, observed=True)[[group_col, sex_col]]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if inconsistent.any():
        bad = sorted(inconsistent.index[inconsistent].astype(str).tolist())
        raise ValueError(f"Primate metadata assigns multiple group/sex values to animal_id: {bad}")
    primate_units = primate_units.drop_duplicates(animal_col)

    checked = manifest.merge(plasma_units, on=sample_col, how="left", validate="one_to_one")
    if checked[[group_col, sex_col]].isna().any(axis=None):
        missing_samples = checked.loc[checked[group_col].isna() | checked[sex_col].isna(), sample_col].tolist()
        raise ValueError(f"Validated linkage sample_id values are absent from plasma metadata: {missing_samples}")
    checked = checked.merge(
        primate_units,
        on=animal_col,
        how="left",
        suffixes=("_plasma", "_primate"),
        validate="one_to_one",
    )
    primate_group_col = f"{group_col}_primate"
    primate_sex_col = f"{sex_col}_primate"
    if checked[[primate_group_col, primate_sex_col]].isna().any(axis=None):
        missing_animals = checked.loc[
            checked[primate_group_col].isna() | checked[primate_sex_col].isna(), animal_col
        ].tolist()
        raise ValueError(f"Validated linkage animal_id values are absent from primate metadata: {missing_animals}")

    plasma_groups = checked[f"{group_col}_plasma"].map(_canonical_group_label)
    primate_groups = checked[primate_group_col].map(_canonical_group_label)
    if not plasma_groups.eq(primate_groups).all():
        bad = checked.loc[~plasma_groups.eq(primate_groups), sample_col].tolist()
        raise ValueError(f"Validated linkage group mismatch for sample_id values: {bad}")
    plasma_sex = checked[f"{sex_col}_plasma"].astype(str).str.strip().str.upper()
    primate_sex = checked[primate_sex_col].astype(str).str.strip().str.upper()
    if not plasma_sex.eq(primate_sex).all():
        bad = checked.loc[~plasma_sex.eq(primate_sex), sample_col].tolist()
        raise ValueError(f"Validated linkage sex mismatch for sample_id values: {bad}")

    result = manifest.copy()
    result["identity_confirmed"] = confirmed.to_numpy()
    result["animal_id_source"] = "validated_manifest"
    result["animal_id_confidence"] = "metadata_exact"
    result["mapping_rule"] = "validated_manifest_sha256"
    result["mapping_reason"] = result["evidence_source"].map(
        lambda source: f"Explicit biological identity confirmed by {source}"
    )
    result["linkage_manifest_sha256"] = observed
    return result


def audit_primate_plasma_linkage(
    prim_meta: pd.DataFrame,
    prim_plasma_meta: pd.DataFrame,
    animal_col: str = "animal_id",
    group_col: str = "group",
    sex_col: str = "sex",
    animal_confidence_col: str = "animal_id_confidence",
    high_conf_values: Sequence[str] = ("high", "metadata_exact"),
    treated_label: str = "O_GES",
    control_labels: Sequence[str] = ("O_V", "O_WT"),
) -> Dict[str, Any]:
    n_prim_samples = int(len(prim_meta))
    n_plasma_samples = int(len(prim_plasma_meta))

    has_prim_animal_col = animal_col in prim_meta.columns
    has_plasma_animal_col = animal_col in prim_plasma_meta.columns

    prim_non_null = 0
    plasma_non_null = 0
    prim_ids = set()
    plasma_ids = set()

    if has_prim_animal_col:
        prim_vals = prim_meta[animal_col].dropna().astype(str)
        prim_non_null = int(prim_vals.shape[0])
        prim_ids = set(prim_vals.tolist())

    if has_plasma_animal_col:
        plasma_vals = prim_plasma_meta[animal_col].dropna().astype(str)
        plasma_non_null = int(plasma_vals.shape[0])
        plasma_ids = set(plasma_vals.tolist())

    high_conf_set = {str(x).strip().lower() for x in high_conf_values}
    n_plasma_high_conf_non_null = 0
    overlap_high_conf_ids = set()
    n_overlap_treated = 0
    n_overlap_control = 0
    mapping_coverage = 0.0
    mapping_collision_count = 0

    if has_plasma_animal_col and animal_confidence_col in prim_plasma_meta.columns:
        plasma_conf = (
            prim_plasma_meta[animal_confidence_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(high_conf_set)
        )
        plasma_hc_vals = prim_plasma_meta.loc[plasma_conf, animal_col].dropna().astype(str)
        n_plasma_high_conf_non_null = int(plasma_hc_vals.shape[0])
        plasma_hc_ids = set(plasma_hc_vals.tolist())
        mapping_collision_count = int(plasma_hc_vals.duplicated().sum())
    elif has_plasma_animal_col:
        # Linkage without explicit confidence provenance is not causal-analysis ready.
        plasma_hc_ids = set()
    else:
        plasma_hc_ids = set()

    overlap_high_conf_ids = prim_ids.intersection(plasma_hc_ids)
    n_overlap_high_conf = int(len(overlap_high_conf_ids))
    mapping_coverage = (
        float(n_overlap_high_conf / n_plasma_samples) if n_plasma_samples > 0 else 0.0
    )

    if n_overlap_high_conf > 0 and group_col in prim_meta.columns:
        prim_group = (
            prim_meta[[animal_col, group_col]]
            .dropna()
            .astype({animal_col: str})
            .groupby(animal_col)[group_col]
            .agg(_mode_or_first)
        )
        overlap_idx = prim_group.index.intersection(pd.Index(list(overlap_high_conf_ids)))
        ctrl_set = {str(x) for x in control_labels}
        n_overlap_treated = int((prim_group.loc[overlap_idx] == str(treated_label)).sum())
        n_overlap_control = int(prim_group.loc[overlap_idx].isin(ctrl_set).sum())

    overlap_ids = prim_ids.intersection(plasma_ids)
    n_overlap = int(len(overlap_ids))

    overlap_fraction_prim = float(n_overlap / len(prim_ids)) if prim_ids else 0.0
    overlap_fraction_plasma = float(n_overlap / len(plasma_ids)) if plasma_ids else 0.0

    group_concordance = float("nan")
    sex_concordance = float("nan")

    if n_overlap > 0 and group_col in prim_meta.columns and group_col in prim_plasma_meta.columns:
        prim_group = (
            prim_meta[[animal_col, group_col]]
            .dropna()
            .astype({animal_col: str})
            .groupby(animal_col)[group_col]
            .agg(_mode_or_first)
        )
        plasma_group = (
            prim_plasma_meta[[animal_col, group_col]]
            .dropna()
            .astype({animal_col: str})
            .groupby(animal_col)[group_col]
            .agg(_mode_or_first)
        )
        common = prim_group.index.intersection(plasma_group.index)
        if len(common) > 0:
            pg = prim_group.loc[common].map(_canonical_group_label)
            qg = plasma_group.loc[common].map(_canonical_group_label)
            group_concordance = float((pg == qg).mean())

    if n_overlap > 0 and sex_col in prim_meta.columns and sex_col in prim_plasma_meta.columns:
        prim_sex = (
            prim_meta[[animal_col, sex_col]]
            .dropna()
            .astype({animal_col: str})
            .groupby(animal_col)[sex_col]
            .agg(_mode_or_first)
        )
        plasma_sex = (
            prim_plasma_meta[[animal_col, sex_col]]
            .dropna()
            .astype({animal_col: str})
            .groupby(animal_col)[sex_col]
            .agg(_mode_or_first)
        )
        common = prim_sex.index.intersection(plasma_sex.index)
        if len(common) > 0:
            sex_concordance = float((prim_sex.loc[common] == plasma_sex.loc[common]).mean())

    return {
        "available": True,
        "n_prim_samples": n_prim_samples,
        "n_plasma_samples": n_plasma_samples,
        "has_prim_animal_id_col": bool(has_prim_animal_col),
        "has_plasma_animal_id_col": bool(has_plasma_animal_col),
        "has_plasma_animal_id_confidence_col": bool(animal_confidence_col in prim_plasma_meta.columns),
        "n_prim_animal_id_non_null": prim_non_null,
        "n_plasma_animal_id_non_null": plasma_non_null,
        "n_plasma_high_conf_animal_id_non_null": n_plasma_high_conf_non_null,
        "n_overlap_animal_ids": n_overlap,
        "n_overlap_animal_ids_high_conf": n_overlap_high_conf,
        "n_overlap_treated_animals": n_overlap_treated,
        "n_overlap_control_animals": n_overlap_control,
        "overlap_fraction_prim_animals": overlap_fraction_prim,
        "overlap_fraction_plasma_animals": overlap_fraction_plasma,
        "mapping_coverage": mapping_coverage,
        "mapping_collision_count": mapping_collision_count,
        "group_concordance_on_overlap": group_concordance,
        "sex_concordance_on_overlap": sex_concordance,
    }


def build_estimability_report(
    linkage_audit: Dict[str, Any],
    min_samples_for_mediation: int = 12,
    min_overlap_animals: int = 20,
    min_treated_overlap: int = 6,
    min_control_overlap: int = 12,
) -> Dict[str, Any]:
    has_prim = bool(linkage_audit.get("has_prim_animal_id_col", False))
    has_plasma = bool(linkage_audit.get("has_plasma_animal_id_col", False))
    has_plasma_confidence = bool(
        linkage_audit.get("has_plasma_animal_id_confidence_col", False)
    )
    n_overlap = int(
        linkage_audit.get(
            "n_overlap_animal_ids_high_conf",
            linkage_audit.get("n_overlap_animal_ids", 0),
        )
        or 0
    )
    n_overlap_treated = int(linkage_audit.get("n_overlap_treated_animals", 0) or 0)
    n_overlap_control = int(linkage_audit.get("n_overlap_control_animals", 0) or 0)
    mapping_collision_count = int(linkage_audit.get("mapping_collision_count", 0) or 0)

    if not has_prim or not has_plasma:
        tier = "unlinked"
        can_do_mediation = False
        reason = "animal_id column missing in bulk and/or plasma metadata."
        reason_code = PLASMA_BULK_ANIMAL_LINKAGE_MISSING
    elif not has_plasma_confidence:
        tier = "unlinked"
        can_do_mediation = False
        reason = (
            "Plasma animal_id confidence column is missing; linkage confidence provenance "
            "is required for individual-level inference."
        )
        reason_code = PLASMA_LINKAGE_CONFIDENCE_MISSING
    elif mapping_collision_count > 0:
        tier = "partially_linked"
        can_do_mediation = False
        reason = (
            f"Detected {mapping_collision_count} duplicate high-confidence plasma-to-animal "
            "mapping collision(s); linked causal analysis requires one plasma row per animal."
        )
        reason_code = PLASMA_ANIMAL_LINKAGE_COLLISION
    elif n_overlap <= 0:
        tier = "unlinked"
        can_do_mediation = False
        reason = "No overlapping high-confidence animal_id values across bulk and plasma metadata."
        reason_code = PLASMA_BULK_ANIMAL_LINKAGE_MISSING
    elif n_overlap < int(min_overlap_animals):
        tier = "partially_linked"
        can_do_mediation = False
        reason = (
            f"Only {n_overlap} overlapping high-confidence animals (< {int(min_overlap_animals)} "
            "minimum required for linked analysis)."
        )
        reason_code = INSUFFICIENT_LINKED_ANIMALS
    elif n_overlap_treated < int(min_treated_overlap):
        tier = "partially_linked"
        can_do_mediation = False
        reason = (
            f"Only {n_overlap_treated} overlapping treated animals (< {int(min_treated_overlap)} "
            "minimum required)."
        )
        reason_code = INSUFFICIENT_LINKED_ANIMALS
    elif n_overlap_control < int(min_control_overlap):
        tier = "partially_linked"
        can_do_mediation = False
        reason = (
            f"Only {n_overlap_control} overlapping control animals (< {int(min_control_overlap)} "
            "minimum required)."
        )
        reason_code = INSUFFICIENT_LINKED_ANIMALS
    elif n_overlap < int(min_samples_for_mediation):
        tier = "partially_linked"
        can_do_mediation = False
        reason = (
            f"Only {n_overlap} overlapping animals (< {int(min_samples_for_mediation)} "
            "minimum for stable mediation)."
        )
        reason_code = INSUFFICIENT_LINKED_ANIMALS
    else:
        tier = "fully_linked"
        can_do_mediation = True
        reason = "Sufficient animal-level overlap for mediation/decomposition."
        reason_code = OK

    return {
        "available": True,
        "tier": tier,
        "can_do_mediation": bool(can_do_mediation),
        "can_do_linked_decomposition": bool(can_do_mediation),
        "n_overlap_animal_ids": n_overlap,
        "n_overlap_treated_animals": n_overlap_treated,
        "n_overlap_control_animals": n_overlap_control,
        "mapping_collision_count": mapping_collision_count,
        "min_overlap_animals": int(min_overlap_animals),
        "min_treated_overlap": int(min_treated_overlap),
        "min_control_overlap": int(min_control_overlap),
        "min_samples_for_mediation": int(min_samples_for_mediation),
        "reason": reason,
        "reason_code": reason_code,
        "missing_author_key": (
            "" if reason_code == OK else missing_author_key_for_reason_code(reason_code)
        ),
    }
