from __future__ import annotations

"""
run_pipeline.py

End-to-end analysis pipeline for the OMIX Exosome Rejuvenation project:
- loads primate bulk RNA-seq (OMIX007580),
- trains and cross-validates a lightweight transcriptomic clock,
- derives a proxy rejuvenation score,
- estimates tissue-level exosome-like effects,
- processes plasma proteomics (OMIX007581) using prespecified treatment
  contrasts and a reference-trained plasma age-state axis,
- loads Mammal40 methylation (OMIX007582) as a guarded validation block, while
  keeping biological sample mapping conservative when only technical IDs are available.
"""

import argparse
import sys
import inspect
import re
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import PipelineConfig, OmixPaths
from .logging_utils import get_logger
from .run_provenance import begin_run_manifest, finalize_run_manifest

from .omix_io import (
    load_omix_matrix,
    load_omix_metadata,
    align_matrix_and_metadata,
    find_first_present_column,
    assert_metadata_reasonable,
)
from .omix007580 import (
    OMIX007580_RELEASE_CONTRACT,
    load_omix007580_anonymous_counts,
    sha256_file,
)

from .preprocessing import (
    standardize_metadata_columns,
    log1p_counts,
    log2_cpm_counts,
    filter_top_variance,
    build_proxy_rejuvenation_score,
)

from .clocks import (
    train_transcriptomic_clock,
    predict_biological_age,
)
from .reference_clock import (
    apply_exact_sample_exclusions,
    frozen_fold_assignment_digest,
    train_reference_transcriptomic_clock,
)

from .exosome_effect import (
    compare_effect_patterns,
    simple_mediation_bootstrap,
)
from .plasma_axis import (
    build_oriented_plasma_aging_axis,
    correlate_plasma_axis_with_delta_age,
)
from .attribution import (
    assign_group_stage_age,
    build_mouse_exosome_metadata,
    build_subset_validation_table,
    compute_cross_species_response_alignment,
    compute_exosome_alignment_tables,
    compute_mouse_exosome_tissue_effects,
    load_cross_species_tissue_mapping,
    read_subset_sample_info,
    summarize_mouse_exosome_signatures,
    summarize_multimodal_concordance,
)

from .translation_module import generate_translational_insights
from .viz import (
    plot_tissue_concordance,
    plot_plasma_biomarker_ranking,
    plot_age_scatter,
    plot_rejuvenation_by_group,
    plot_mediation_effects_bar,
)

from .rejuvenation import (
    annotate_effect_uncertainty,
    compute_delta_age,
    summarize_animal_treatment_contrasts_by_tissue,
    summarize_clustered_treatment_sensitivity,
    summarize_global_animal_treatment_contrasts,
    summarize_rejuvenation_by_tissue,
    summarize_tissue_expression_effects,
    compute_plasma_protein_contrasts,
)
from .linkage_audit import (
    audit_primate_plasma_linkage,
    build_estimability_report,
    load_validated_plasma_linkage_manifest,
)
from .reason_codes import PLASMA_LINKAGE_CONFIDENCE_MISSING, annotate_reason_fields

logger = get_logger(__name__)

STANDARD_RESULT_COLUMNS = [
    "available",
    "estimable",
    "reason",
    "reason_code",
    "missing_author_key",
    "n_used",
    "method",
    "ci_low",
    "ci_high",
    "evidence_level",
]

PROFILE_SPECS: Dict[str, Dict[str, Any]] = {
    "full": {
        "data_root_relative": Path("data/RAW/data"),
        "bulk_matrix": "OMIX007580-01.txt",
        "bulk_metadata": "OMIX007580-02.csv",
        "bulk_input_mode": "omix007580_anonymous_counts",
        "bulk_representation": "log2_cpm",
        "clock_protocol": "reference_nested_ridge",
        "clock_excluded_sample_ids": ["58-MF-C-Trachea_4"],
        "clock_fold_file": "config/primate_clock_folds.csv",
        "clock_fold_sha256": "666731461817ec5c9e9b9cb3007a62ee328b61a805f0caab2779fc2663bcf98e",
        "plasma_matrix": "OMIX007581-01.csv",
        "plasma_metadata": None,
        "methylation_matrix": "OMIX007582_beta_matrix.csv",
        "methylation_metadata": "OMIX007582-02.csv",
        "mouse_matrix": "OMIX009283-01.txt",
        "mouse_metadata": "OMIX009283_metadata.csv",
        "subset_files": ["OMIX007583-01.zip", "OMIX007586-02.zip"],
    },
    "demo": {
        "data_root_relative": Path("data/PROCESSED"),
        "bulk_matrix": "OMIX007580_01_example.txt",
        "bulk_metadata": "OMIX007580-02_example.csv",
        "bulk_input_mode": "feature_indexed",
        "bulk_representation": "log1p_counts",
        "clock_protocol": "legacy_all_samples_ridge",
        "clock_excluded_sample_ids": [],
        "clock_fold_file": None,
        "clock_fold_sha256": None,
        "plasma_matrix": "OMIX007581-01_example.csv",
        "plasma_metadata": None,
        "methylation_matrix": "OMIX007582_beta_matrix_example.csv",
        "methylation_metadata": "OMIX007582-02_example.csv",
        "mouse_matrix": None,
        "mouse_metadata": None,
        "subset_files": [],
    },
}


def call_with_supported_kwargs(func, /, *args, **kwargs):
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return func(*args, **filtered)


def _ensure_standard_schema(
    df: pd.DataFrame,
    *,
    available: Optional[bool] = None,
    estimable: Optional[bool] = None,
    reason: Optional[str] = None,
    reason_code: Optional[str] = None,
    missing_author_key: Optional[str] = None,
    n_used: Optional[int] = None,
    method: Optional[str] = None,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    evidence_level: Optional[int] = None,
) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "available": True if available is None else bool(available),
        "estimable": True if estimable is None else bool(estimable),
        "reason": "" if reason is None else str(reason),
        "reason_code": "" if reason_code is None else str(reason_code),
        "missing_author_key": "" if missing_author_key is None else str(missing_author_key),
        "n_used": np.nan if n_used is None else int(n_used),
        "method": "" if method is None else str(method),
        "ci_low": np.nan if ci_low is None else float(ci_low),
        "ci_high": np.nan if ci_high is None else float(ci_high),
        "evidence_level": 0 if evidence_level is None else int(evidence_level),
    }

    for col in STANDARD_RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = defaults[col]

    if available is not None:
        out["available"] = out["available"].fillna(bool(available))
    if estimable is not None:
        out["estimable"] = out["estimable"].fillna(bool(estimable))
    if reason is not None:
        out["reason"] = out["reason"].replace("", str(reason)).fillna(str(reason))
    if reason_code is not None:
        out["reason_code"] = out["reason_code"].replace("", str(reason_code)).fillna(str(reason_code))
    if missing_author_key is not None:
        out["missing_author_key"] = (
            out["missing_author_key"]
            .replace("", str(missing_author_key))
            .fillna(str(missing_author_key))
        )
    if n_used is not None:
        out["n_used"] = out["n_used"].fillna(int(n_used))
    if method is not None:
        out["method"] = out["method"].replace("", str(method)).fillna(str(method))
    if evidence_level is not None:
        out["evidence_level"] = out["evidence_level"].fillna(int(evidence_level))

    return annotate_reason_fields(out)


def _attach_primary_mediation_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Populate the standard CI fields with the total-effect interval."""
    out = df.copy()
    if {"total_ci_low", "total_ci_high"}.issubset(out.columns):
        out["ci_low"] = pd.to_numeric(out["total_ci_low"], errors="coerce")
        out["ci_high"] = pd.to_numeric(out["total_ci_high"], errors="coerce")
        return out

    if "Total_CI" not in out.columns:
        return out

    def bounds(value: object) -> tuple[float, float]:
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2:
            numeric = pd.to_numeric(pd.Series([value[0], value[1]]), errors="coerce")
            return float(numeric.iloc[0]), float(numeric.iloc[1])
        return np.nan, np.nan

    intervals = out["Total_CI"].map(bounds)
    out["ci_low"] = intervals.map(lambda interval: interval[0])
    out["ci_high"] = intervals.map(lambda interval: interval[1])
    return out


def _evidence_level(
    *,
    estimable: bool,
    tier: Optional[str] = None,
    has_linkage_support: bool = False,
    has_exosome_alignment: bool = False,
    has_linked_mediation: bool = False,
) -> int:
    if bool(has_linked_mediation) and str(tier) == "fully_linked":
        return 4
    if bool(has_exosome_alignment):
        return 3
    if bool(has_linkage_support):
        return 2
    if bool(estimable):
        return 1
    return 0


def _causal_exosome_fraction_status() -> Dict[str, Any]:
    """Return the structured non-identifiability result for causal attribution."""
    return {
        "available": True,
        "estimable": False,
        "reason": (
            "A causal exosome-attributable fraction is not identified because whole-cell "
            "therapy was studied in macaques and exosome therapy in mice, without a "
            "within-macaque exosome intervention or a validated causal mediator. Quantitative "
            "response resemblance is reported separately in "
            "cross_species_response_alignment_summary.csv."
        ),
        "n_used": 0,
        "method": "causal_exosome_fraction_not_identified",
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n_common_tissues": 0,
        "cells_median_abs": np.nan,
        "exo_median_abs": np.nan,
        "ratio": np.nan,
        "empirical_p_value": np.nan,
        "evidence_level": 0,
    }


def _contrast_pairs(specs: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for spec in specs:
        left, _, right = str(spec).partition("_vs_")
        if left and right:
            pairs.append((left, right))
    return pairs


def _result_stub(
    *,
    reason: str,
    method: str,
    extra: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    row = {
        "available": False,
        "estimable": False,
        "reason": reason,
        "n_used": 0,
        "method": method,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "evidence_level": 0,
    }
    if extra:
        row.update(extra)
    return _ensure_standard_schema(pd.DataFrame([row]), method=method, evidence_level=0)


def _build_mediation_stub(
    *,
    reason: str,
    tier: str,
    n_overlap_animal_ids: int,
    n_used: int = 0,
) -> pd.DataFrame:
    """Build the canonical non-estimable mediation row."""
    return _result_stub(
        reason=reason,
        method="linear_mediation_bootstrap",
        extra={
            "tier": str(tier),
            "n_overlap_animal_ids": int(n_overlap_animal_ids),
            "n_used": int(n_used),
        },
    )


def _build_animal_level_mediation_table(
    prim_meta: pd.DataFrame,
    plasma_meta: pd.DataFrame,
    *,
    high_conf_values: Sequence[str],
) -> pd.DataFrame:
    """Return one linked mediation row per high-confidence animal."""
    output_columns = [
        "animal_id",
        "rejuvenation_score",
        "group_binary",
        "group",
        "plasma_state_score",
    ]
    prim_required = {"animal_id", "rejuvenation_score", "group_binary"}
    plasma_required = {"animal_id", "plasma_state_score", "animal_id_confidence"}
    if not prim_required.issubset(prim_meta.columns) or not plasma_required.issubset(plasma_meta.columns):
        return pd.DataFrame(columns=output_columns)

    prim_agg = {
        "rejuvenation_score": ("rejuvenation_score", "median"),
        "group_binary": ("group_binary", "max"),
    }
    if "group" in prim_meta.columns:
        prim_agg["group"] = (
            "group",
            lambda s: s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else "",
        )

    prim_animal_level = (
        prim_meta.dropna(subset=["animal_id", "rejuvenation_score"])
        .copy()
        .groupby("animal_id", as_index=False)
        .agg(**prim_agg)
    )
    if "group" not in prim_animal_level.columns:
        prim_animal_level["group"] = ""

    allowed_conf = {str(x).strip().lower() for x in high_conf_values}
    confidence = plasma_meta["animal_id_confidence"].astype(str).str.strip().str.lower()
    plasma_animal_level = (
        plasma_meta.loc[confidence.isin(allowed_conf)]
        .dropna(subset=["animal_id", "plasma_state_score"])
        .groupby("animal_id", as_index=False)
        .agg(plasma_state_score=("plasma_state_score", "median"))
    )

    merged = prim_animal_level.merge(
        plasma_animal_level,
        on="animal_id",
        how="inner",
        validate="one_to_one",
    )
    if merged["animal_id"].duplicated().any():
        raise ValueError("Animal-level mediation table contains duplicate animal_id rows.")
    return merged[output_columns]


def _resolve_profile_name(base_dir: Path, requested: str, explicit_data_root: Optional[Path] = None) -> str:
    profile = str(requested or "auto").strip().lower()
    if profile not in {"auto", "full", "demo"}:
        raise ValueError(f"Unsupported data profile: {requested}")
    if profile != "auto":
        return profile

    if explicit_data_root is not None:
        root = explicit_data_root
        for candidate in ("full", "demo"):
            bulk_name = str(PROFILE_SPECS[candidate]["bulk_matrix"])
            if bulk_name and (root / bulk_name).exists():
                return candidate
    else:
        for candidate in ("full", "demo"):
            spec = PROFILE_SPECS[candidate]
            bulk_name = str(spec["bulk_matrix"])
            if (base_dir / spec["data_root_relative"] / bulk_name).exists():
                return candidate

    raise FileNotFoundError(
        "Could not resolve a runnable data profile. Checked both full and demo layouts."
    )


def _resolve_profile_data_root(base_dir: Path, profile: str, explicit_data_root: Optional[Path] = None) -> Path:
    if explicit_data_root is not None:
        return explicit_data_root
    spec = PROFILE_SPECS[profile]
    return base_dir / spec["data_root_relative"]


def _make_omix_paths(
    data_root: Path,
    *,
    matrix_name: Optional[str],
    metadata_name: Optional[str] = None,
    required: bool = False,
) -> Optional[OmixPaths]:
    if not matrix_name:
        return None

    matrix_path = data_root / matrix_name
    if not matrix_path.exists():
        if required:
            raise FileNotFoundError(f"Required dataset file not found: {matrix_path}")
        return None

    metadata_path: Optional[Path] = None
    if metadata_name:
        metadata_candidate = data_root / metadata_name
        if metadata_candidate.exists():
            metadata_path = metadata_candidate
        elif required:
            raise FileNotFoundError(f"Required metadata file not found: {metadata_candidate}")

    return OmixPaths(matrix=matrix_path, metadata=metadata_path)


def _table_to_effect_index(
    df: pd.DataFrame,
    *,
    tissue_col: str = "tissue",
    effect_col: str = "mean_effect",
    treated_col: str = "n_treated",
    control_col: str = "n_control",
) -> pd.DataFrame:
    if df is None or df.empty or tissue_col not in df.columns or effect_col not in df.columns:
        return pd.DataFrame(columns=["mean_effect", "n_treated", "n_control"])

    out = df.copy()
    if "estimable" in out.columns:
        out = out.loc[out["estimable"].astype(bool)]
    if out.empty:
        return pd.DataFrame(columns=["mean_effect", "n_treated", "n_control"])

    keep = [tissue_col, effect_col]
    if treated_col in out.columns:
        keep.append(treated_col)
    if control_col in out.columns:
        keep.append(control_col)
    out = out[keep].dropna(subset=[effect_col]).copy()
    rename_map = {
        effect_col: "mean_effect",
        treated_col: "n_treated",
        control_col: "n_control",
    }
    out = out.rename(columns=rename_map)
    for col in ("n_treated", "n_control"):
        if col not in out.columns:
            out[col] = np.nan
    return out.set_index(tissue_col).sort_values("mean_effect")


def _control_set_variants(labels: List[str]) -> Dict[str, List[str]]:
    base = [str(x) for x in labels if str(x)]
    variants: Dict[str, List[str]] = {"primary": base}

    oc_only = [x for x in base if x == "O_C"]
    if oc_only:
        variants["oc_only"] = oc_only

    no_vehicle = [x for x in base if x not in {"O_V", "O_WT"}]
    if no_vehicle and no_vehicle != base:
        variants["no_vehicle_no_wt"] = no_vehicle

    return variants


def _canonical_group_label(label: str) -> str:
    s = str(label).strip().upper()
    if s.startswith("O_"):
        return s.split("_", 1)[1]
    if s.endswith("_C") and len(s.split("_", 1)[0]) > 0:
        return s.split("_", 1)[0]
    return s


def _causal_gate_reason(
    *,
    enable_mediation: bool,
    enable_causal_decomposition: bool,
    estimability_row: Dict[str, Any],
) -> Optional[str]:
    tier = str(estimability_row.get("tier", "unlinked"))
    can_do_linked = bool(estimability_row.get("can_do_mediation", False))

    if not bool(enable_mediation):
        return "Mediation disabled by config (enable_mediation=False)."
    if not bool(enable_causal_decomposition):
        return "Causal decomposition disabled by config (enable_causal_decomposition=False)."
    if tier != "fully_linked" or not can_do_linked:
        return str(
            estimability_row.get(
                "reason",
                "Causal decomposition requires fully_linked tier and sufficient overlap.",
            )
        )
    return None


# -----------------------------------------------------------------------------
# Runtime audit (helps avoid silent path/version confusion)
# -----------------------------------------------------------------------------
def runtime_sanity_banner(cfg: PipelineConfig) -> None:
    try:
        import src.omix_io as omix_io  # type: ignore
        omix_path = getattr(omix_io, "__file__", "unknown")
        max_samples = getattr(omix_io, "MAX_ALLOWED_SAMPLES", "NA")
        max_expected = getattr(omix_io, "MAX_EXPECTED_SAMPLE_COLS", "NA")
    except Exception:
        omix_path, max_samples, max_expected = "unknown", "NA", "NA"

    logger.info("PYTHON: %s", sys.executable)
    logger.info("sys.path[0:3]: %s", sys.path[:3])
    logger.info("omix_io loaded from: %s", omix_path)
    logger.info(
        "Guardrails: MAX_ALLOWED_SAMPLES=%s, MAX_EXPECTED_SAMPLE_COLS=%s",
        max_samples,
        max_expected,
    )
    logger.info("cfg.max_allowed_samples=%s", getattr(cfg, "max_allowed_samples", None))


# -----------------------------------------------------------------------------
# Local hard guardrails (duplicate by design)
# -----------------------------------------------------------------------------
def hard_guardrail_matrix_and_meta(matrix: pd.DataFrame, meta: pd.DataFrame, context: str = "") -> None:
    if meta.shape[0] > 5000:
        raise ValueError(
            f"[{context}] Metadata has {meta.shape[0]} rows. "
            "Too large for a sample sheet; likely wrong metadata file."
        )
    if matrix.shape[1] > 20000:
        raise ValueError(
            f"[{context}] Matrix has {matrix.shape[1]} columns. "
            "Far beyond expected processed OMIX sample counts; wrong file or delimiter."
        )


def _get_allowed_samples(meta: pd.DataFrame, sample_col: str, cap: int = 5000) -> set[str]:
    vals = meta[sample_col].astype(str).values
    allowed = set(vals)
    if len(allowed) > cap:
        raise ValueError(
            f"Too many sample IDs in metadata column '{sample_col}' "
            f"({len(allowed)} > {cap}). Wrong column or wrong metadata."
        )
    return allowed


def ensure_numeric_age(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical meta['age'] is numeric.

    Prefers 'agenumb' (OMIX007580 pattern), otherwise picks the most numeric-like column.
    Preserves original categorical age label in 'age_label'.
    """
    meta = meta.loc[:, ~meta.columns.duplicated()].copy()

    # Keep original age label if any
    if "age" in meta.columns:
        meta["age_label"] = meta["age"]

    # 1) Strong preference for OMIX numeric age field
    if "agenumb" in meta.columns:
        meta["age"] = pd.to_numeric(meta["agenumb"], errors="coerce")
        return meta

    # 2) Fallback: find a numeric-like age column
    candidates = [
        "age_num",
        "age_years",
        "Age",
        "Age (years)",
        "Age(Y)",
        "Age (Y)",
        "chrono_age",
        "chronological_age",
        "donor_age",
        "subject_age",
    ]
    existing = [c for c in candidates if c in meta.columns]

    def numeric_rate(s: pd.Series) -> float:
        return pd.to_numeric(s, errors="coerce").notna().mean()

    best = None
    best_rate = 0.0

    for c in existing:
        r = numeric_rate(meta[c])
        if r > best_rate:
            best, best_rate = c, r

    if best and best_rate >= 0.5:
        meta["age"] = pd.to_numeric(meta[best], errors="coerce")

    return meta


def _normalize_pred_output(pred: Any) -> pd.DataFrame:
    """
    Make prediction output robust to different return shapes.
    Must end up with columns: sample_id, predicted_age
    """
    if isinstance(pred, pd.Series):
        df = pred.rename("predicted_age").reset_index()
        df = df.rename(columns={"index": "sample_id"})
        return df

    if isinstance(pred, pd.DataFrame):
        df = pred.copy()

        if "predicted_age" not in df.columns:
            for alt in ("pred_age", "predicted", "age_pred", "biological_age"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "predicted_age"})
                    break

        if "sample_id" not in df.columns:
            # Sometimes the sample ID is the index
            if df.index.name:
                df = df.reset_index().rename(columns={df.index.name: "sample_id"})
            else:
                df = df.reset_index().rename(columns={"index": "sample_id"})

        return df

    raise ValueError("Clock prediction output must be a pandas Series or DataFrame.")


def load_plasma_proteomics_csv(
    path: Path,
    *,
    return_annotations: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads OMIX007581-01-like plasma proteomics wide CSV.
    Returns matrix with rows=unique protein accessions and cols=samples.
    Gene symbols are annotations because they are not unique in this release.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if "Protein accession" in df.columns:
        idx = df["Protein accession"].astype("string").str.strip()
        if idx.isna().any() or idx.eq("").any() or idx.duplicated().any():
            raise ValueError("Protein accession must be complete and unique in the plasma matrix.")
    else:
        idx = pd.Series(
            [f"plasma_feature_{i:05d}" for i in range(len(df))],
            index=df.index,
            dtype="string",
        )

    drop_set = {"Protein accession", "Gene name"}
    sample_cols = [c for c in df.columns if c not in drop_set]

    # Keep only sample-like columns (e.g., FY_1, MWT_2, FGES_4)
    sample_cols = [c for c in sample_cols if re.match(r"^[A-Za-z]+_\d+$", c)]

    mat = df.loc[:, sample_cols].copy()
    mat.index = pd.Index(idx.astype(str), name="feature_id")

    # Numeric conversion (float32-ish)
    for c in mat.columns:
        mat[c] = pd.to_numeric(mat[c], errors="coerce", downcast="float")

    annotations = pd.DataFrame(
        {
            "feature_id": mat.index.astype(str),
            "protein_accession": (
                df["Protein accession"].astype("string").str.strip().to_numpy()
                if "Protein accession" in df.columns
                else mat.index.astype(str)
            ),
            "gene_name": (
                df["Gene name"].astype("string").str.strip().fillna("").to_numpy()
                if "Gene name" in df.columns
                else np.repeat("", len(df))
            ),
        }
    )
    if return_annotations:
        return mat, annotations
    return mat


def build_plasma_metadata_from_columns(sample_cols: list[str]) -> pd.DataFrame:
    """
    Builds minimal metadata from sample IDs like FY_1, MWT_3, FGES_2.
    """
    rows = []
    for s in sample_cols:
        m = re.match(r"^([A-Za-z]+)_(\d+)$", s)
        if not m:
            continue

        code, rep = m.group(1), int(m.group(2))
        sex = code[0].upper() if code else None
        group_code = code[1:].upper() if len(code) > 1 else code.upper()
        animal_id, mapping_rule, mapping_conf, mapping_reason = _map_plasma_sample_to_bulk_animal_id(s)

        rows.append(
            {
                "sample_id": s,
                "raw_code": code,
                "sex": "F" if sex == "F" else ("M" if sex == "M" else pd.NA),
                "group": group_code,  # Y, V, WT, GES, etc.
                "group_code": group_code,
                "replicate": rep,
                "omics": "plasma_proteomics",
                "animal_id": animal_id if animal_id is not None else pd.NA,
                "animal_id_source": "deterministic_rule" if animal_id is not None else "unresolved",
                "animal_id_confidence": mapping_conf,
                "mapping_rule": mapping_rule,
                "mapping_reason": mapping_reason,
            }
        )

    return pd.DataFrame(rows)


def _map_plasma_sample_to_bulk_animal_id(sample_id: str) -> Tuple[Optional[str], str, str, str]:
    """
    Candidate mapping from OMIX007581-style plasma aliases to bulk animal IDs.
    A naming match does not establish cross-modal biological identity.

    Group-name semantics are documented in `docs/group_label_crosswalk.md`.
    In brief: public OMIX/BioProject naming strongly supports `V -> O_V`,
    `WT -> O_WT`, and `GES -> O_GES`, but those remain cross-source mappings
    rather than verbatim article labels.

    Supported candidate mappings (excluded from high-confidence inference):
      FV_2   -> F-V-2
      MWT_3  -> M-WT-3
      FGES_1 -> F-GES-1
      MGES_4 -> M-GES-4

    FY_*/MY_* and other patterns remain unresolved by design.
    """
    s = str(sample_id).strip()
    m = re.match(r"^([A-Za-z]+)_(\d+)$", s)
    if not m:
        return None, "none", "low", "sample_id pattern does not match ^[A-Za-z]+_\\d+$"

    code = m.group(1).upper()
    rep = m.group(2)
    if len(code) < 2:
        return None, "none", "low", "code token too short for deterministic mapping"

    sex = code[0]
    group_code = code[1:]
    if sex not in {"F", "M"}:
        return None, "none", "low", f"unsupported sex token '{sex}'"

    if group_code in {"V", "WT", "GES"}:
        animal_id = f"{sex}-{group_code}-{rep}"
        return (
            animal_id,
            "plasma_code_to_bulk_orig_ident",
            "inferred",
            "V/WT/GES alias match; shared individual numbering awaits author confirmation",
        )

    return None, "none", "low", f"group code '{group_code}' not deterministically linkable"


def _clean_string_set(vals: pd.Series) -> set[str]:
    s = vals.astype(str).str.strip()
    s = s[~s.isin(["", "nan", "None", "<NA>"])]
    return set(s.tolist())


def clean_plasma_matrix(
    mat: pd.DataFrame,
    min_feature_non_nan_frac: float = 0.5,
    min_sample_non_nan_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Apply missingness filters without imputing the source plasma matrix.

    Downstream analyses own their imputation policy. In particular, the
    reference-trained age-state axis derives imputation values from young and
    vehicle samples only, while protein contrasts use observed values.
    Rows = proteins/genes, cols = samples.
    """

    # Drop rows/cols that are entirely missing
    mat = mat.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if mat.empty:
        raise ValueError("Plasma matrix is empty after dropping all-NaN rows/cols.")

    # Filter features (proteins) with too many missing values
    feat_non_nan = mat.notna().mean(axis=1)
    mat = mat.loc[feat_non_nan >= min_feature_non_nan_frac]

    # Filter samples with too many missing values
    samp_non_nan = mat.notna().mean(axis=0)
    mat = mat.loc[:, samp_non_nan >= min_sample_non_nan_frac]

    if mat.empty:
        raise ValueError("Plasma matrix became empty after missingness filtering.")

    return mat


def build_methylation_name_mapping(
    matrix_cols: List[str],
    meta: pd.DataFrame,
    candidate_cols: List[str] = None,
) -> Dict[str, str]:
    """
    Try to map methylation matrix column names to metadata sample IDs.

    Strategy per column:
      1) exact match vs meta[candidate_col]
      2) metadata value contained in column name
      3) column name contained in metadata value

    Only accept mappings that have exactly one unique match.
    """
    if candidate_cols is None:
        candidate_cols = [
            "OriginalSampleName",
            "OriginalSampleName.1",
            "sample",
            "sample_id",
        ]

    # Restrict to columns that actually exist
    candidate_cols = [c for c in candidate_cols if c in meta.columns]
    if not candidate_cols:
        logger.warning("No candidate columns found in methylation metadata for mapping.")
        return {}

    # Ensure everything is string
    meta = meta.copy()
    for c in candidate_cols:
        meta[c] = meta[c].astype(str).fillna("")

    mapping: Dict[str, str] = {}
    unmapped: List[str] = []

    for col in matrix_cols:
        col_str = str(col)
        matches = []

        for c in candidate_cols:
            vals = meta[c].values

            # 1) exact matches
            exact_idx = vals == col_str
            if exact_idx.sum() == 1:
                v = vals[exact_idx][0]
                matches.append((c, v, "exact"))
                continue

            # 2) metadata value contained in column name
            contained_idx = [i for i, v in enumerate(vals) if v and v in col_str]
            if len(contained_idx) == 1:
                v = vals[contained_idx[0]]
                matches.append((c, v, "meta_in_col"))
                continue

            # 3) column name contained in metadata value (less likely)
            contains_idx = [i for i, v in enumerate(vals) if v and col_str in v]
            if len(contains_idx) == 1:
                v = vals[contains_idx[0]]
                matches.append((c, v, "col_in_meta"))
                continue

        if not matches:
            unmapped.append(col_str)
            continue

        # Collect all candidate target labels
        target_labels = {m[1] for m in matches}
        if len(target_labels) == 1:
            target = target_labels.pop()
            mapping[col_str] = target
        else:
            # ambiguous: multiple different metadata labels matched this col
            logger.warning(
                "Ambiguous mapping for methylation column %s -> %s; keeping original.",
                col_str,
                target_labels,
            )
            unmapped.append(col_str)

    logger.info(
        "Methylation mapping: %d mapped, %d unmapped out of %d columns.",
        len(mapping),
        len(unmapped),
        len(matrix_cols),
    )

    if unmapped:
        logger.debug("Unmapped methylation columns (first 20): %s", unmapped[:20])

    return mapping


# -----------------------------------------------------------------------------
# Safe load + standardize
# -----------------------------------------------------------------------------
def load_align_standardize(
    omix: OmixPaths,
    cfg: PipelineConfig,
    assume_counts: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memory-safe load:
      1) Load metadata
      2) Validate metadata size
      3) Pick sample-id column (strict)
      4) Build allowed_samples with cap
      5) Load matrix with allowed_samples
      6) Hard guardrail on shapes
      7) Align matrix + metadata
      8) Standardize metadata columns
      9) Optional early feature reduction
      10) Log1p transform
    """
    meta = load_omix_metadata(omix.metadata)
    assert_metadata_reasonable(meta)

    sample_col = find_first_present_column(meta, cfg.sample_id_col_candidates)
    if sample_col is None:
        raise ValueError(
            f"No reliable sample ID column found in {omix.metadata}. "
            "Tighten sample_id_col_candidates in config."
        )

    allowed_cap = getattr(cfg, "max_allowed_samples", None) or 5000
    allowed = _get_allowed_samples(meta, sample_col, cap=allowed_cap)

    input_audit = None
    if cfg.primate_bulk_input_mode == "omix007580_anonymous_counts":
        if sha256_file(omix.metadata) != OMIX007580_RELEASE_CONTRACT.metadata_sha256:
            raise ValueError(
                "OMIX007580 metadata checksum changed; the curation contract is version-specific"
            )
        matrix, input_audit = load_omix007580_anonymous_counts(
            omix.matrix,
            expected_sample_ids=allowed,
            dtype="float32",
        )
    elif cfg.primate_bulk_input_mode == "feature_indexed":
        matrix = load_omix_matrix(
            omix.matrix,
            allowed_samples=allowed,
            dtype="float32",
        )
    else:
        raise ValueError(f"Unsupported primate bulk input mode: {cfg.primate_bulk_input_mode}")

    hard_guardrail_matrix_and_meta(matrix, meta, context=str(omix.matrix))

    matrix, meta = align_matrix_and_metadata(matrix, meta, cfg.sample_id_col_candidates)

    # Strong post-alignment check: if alignment worked, width shouldn't exceed allowed IDs
    if matrix.shape[1] > len(allowed):
        raise ValueError(
            f"Post-alignment matrix still too wide ({matrix.shape[1]}) vs allowed ({len(allowed)}). "
            "Sample-ID matching did not reduce columns as expected."
        )

    # Keep a raw copy to preserve non-canonical columns (e.g., agenumb)
    meta_raw = meta.copy()

    meta_std = standardize_metadata_columns(
        meta,
        group_candidates=cfg.group_col_candidates,
        tissue_candidates=cfg.tissue_col_candidates,
        age_candidates=cfg.age_col_candidates,
        sex_candidates=cfg.sex_col_candidates,
        animal_id_candidates=cfg.animal_id_col_candidates,
    )

    # Ensure we keep any columns dropped by standardization (e.g., agenumb)
    if "sample_id" in meta_std.columns and "sample_id" in meta_raw.columns:
        shared = set(meta_std.columns)
        extra = [c for c in meta_raw.columns if c not in shared]
        if extra:
            meta = meta_std.merge(
                meta_raw[["sample_id", *extra]],
                on="sample_id",
                how="left",
            )
        else:
            meta = meta_std
    else:
        # Fallback: keep standardized version and reattach missing columns by position
        meta = meta_std.copy()
        for c in meta_raw.columns:
            if c not in meta.columns:
                meta[c] = meta_raw[c].values

    # Remove duplicate column names defensively
    meta = meta.loc[:, ~meta.columns.duplicated()].copy()

    # ---- Early feature reduction (safe if your filter is defensive) ----
    if getattr(cfg, "n_top_features_expr", None) and cfg.n_top_features_expr > 0:
        logger.info("Matrix shape before variance filter: %s", matrix.shape)
        matrix = filter_top_variance(matrix, cfg.n_top_features_expr)

    if cfg.primate_bulk_representation == "log2_cpm":
        if not assume_counts:
            raise ValueError("log2_cpm representation requires count-like input")
        expr_log = log2_cpm_counts(matrix)
    elif cfg.primate_bulk_representation == "log1p_counts":
        expr_log = log1p_counts(matrix, assume_counts=assume_counts)
        expr_log.attrs.update(matrix.attrs)
        expr_log.attrs["representation"] = "log1p_counts"
    else:
        raise ValueError(
            f"Unsupported primate bulk representation: {cfg.primate_bulk_representation}"
        )
    if input_audit is not None:
        expr_log.attrs["input_audit"] = input_audit.to_record()
    return expr_log, meta


def load_methylation_block(cfg: PipelineConfig) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Guarded loading + light QC of Mammal40 methylation (OMIX007582).

    Returns
    -------
    prim_meth_expr : DataFrame or None
        CpG x sample beta matrix (possibly with technical IDs only).
    prim_meth_meta : DataFrame or None
        Metadata aligned to the matrix when possible; otherwise minimal metadata.

    Notes
    -----
    - Uses OMIX007582_beta_matrix.csv (technical IDs) by default.
    - Attempts to map matrix columns to metadata sample IDs, but does not fail hard
      if mapping is incomplete or ambiguous.
    - The public files bundled in this repo currently expose technical matrix IDs
      without a defensible biological sample key; see
      `docs/OMIX007582_sample_map_audit.md`.
    """
    try:
        meth_meta = load_omix_metadata(cfg.primate_methylation.metadata)
        assert_metadata_reasonable(meth_meta)

        matrix_path = cfg.primate_methylation.matrix
        if not matrix_path.exists():
            raise FileNotFoundError(f"Methylation matrix not found: {matrix_path}")

        if matrix_path.stat().st_size == 0:
            raise ValueError(f"Methylation matrix file appears empty: {matrix_path}")

        meth_matrix = pd.read_csv(matrix_path, index_col=0)
        if meth_matrix.shape[0] == 0 or meth_matrix.shape[1] == 0:
            raise ValueError(
                f"Loaded methylation matrix has shape {meth_matrix.shape}; "
                "check OMIX007582_beta_matrix.csv"
            )

        meth_matrix.index = meth_matrix.index.astype(str)
        meth_matrix.columns = meth_matrix.columns.astype(str)

        logger.info(
            "Raw methylation matrix loaded: %d CpGs x %d samples",
            meth_matrix.shape[0],
            meth_matrix.shape[1],
        )

        # Try to map matrix columns to metadata sample IDs (best effort)
        mapping = build_methylation_name_mapping(
            list(meth_matrix.columns),
            meth_meta,
            candidate_cols=["OriginalSampleName", "OriginalSampleName.1", "sample"],
        )

        if mapping:
            meth_matrix = meth_matrix.rename(columns=mapping)
            logger.info(
                "Renamed %d methylation columns to metadata sample IDs (best-effort mapping).",
                len(mapping),
            )
        else:
            logger.warning(
                "No methylation column mapping could be derived; using raw column names."
            )

        try:
            prim_meth_expr, prim_meth_meta = align_matrix_and_metadata(
                meth_matrix,
                meth_meta,
                cfg.sample_id_col_candidates
                + ["OriginalSampleName", "OriginalSampleName.1", "sample"],
            )
            logger.info(
                "Methylation matrix aligned: %d CpGs x %d samples",
                prim_meth_expr.shape[0],
                prim_meth_expr.shape[1],
            )
        except Exception as e_align:
            logger.warning(
                "Could not align methylation matrix to metadata; "
                "using minimal metadata with sample_id only: %s",
                e_align,
            )
            prim_meth_expr = meth_matrix
            prim_meth_meta = pd.DataFrame({"sample_id": prim_meth_expr.columns})

        prim_meth_meta = standardize_metadata_columns(
            prim_meth_meta,
            cfg.group_col_candidates,
            cfg.tissue_col_candidates,
            cfg.age_col_candidates,
            cfg.sex_col_candidates,
            cfg.animal_id_col_candidates,
        )
        if "sample_id" not in prim_meth_meta.columns:
            prim_meth_meta["sample_id"] = prim_meth_expr.columns.astype(str)

        # QC + light feature reduction
        min_non_nan = int(0.8 * prim_meth_expr.shape[1])
        prim_meth_expr = prim_meth_expr.dropna(axis=0, thresh=min_non_nan)

        if prim_meth_expr.shape[0] > 5000:
            prim_meth_expr = filter_top_variance(prim_meth_expr, 5000)

        logger.info(
            "Methylation matrix after filtering: %d CpGs x %d samples",
            prim_meth_expr.shape[0],
            prim_meth_expr.shape[1],
        )

        return prim_meth_expr, prim_meth_meta

    except Exception as e:
        logger.warning("Methylation block skipped: %s", e)
        return None, None


def run_sensitivity_analyses(
    prim_meta: pd.DataFrame,
    prim_plasma_expr: pd.DataFrame,
    prim_plasma_meta: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    try:
        global_contrasts = summarize_global_animal_treatment_contrasts(
            prim_meta,
            tissue_col=cfg.tissue_col_candidates[0],
            group_col=cfg.group_col_candidates[0],
            animal_col="animal_id",
            sex_col="sex",
            value_col="delta_age",
            contrasts=cfg.primate_treatment_contrasts,
            primary_contrast=cfg.primate_primary_contrast,
            min_per_group=max(2, cfg.min_samples_per_group_for_rejuv),
            n_bootstrap=max(500, cfg.n_bootstrap // 2),
            n_permutations=max(500, cfg.treatment_n_permutations // 2),
            random_state=cfg.random_state,
        )
        for row in global_contrasts.to_dict(orient="records"):
            effect = float(row.get("mean_effect", np.nan))
            rows.append(
                {
                    "analysis_type": "animal_level_treatment_contrast",
                    "scenario": row.get("contrast", "NA"),
                    "effect": effect,
                    "direction": "negative" if effect < 0 else ("positive" if effect > 0 else "zero"),
                    "available": bool(row.get("available", True)),
                    "estimable": bool(row.get("estimable", True)),
                    "reason": row.get("reason", ""),
                    "n_used": int(row.get("n_used", 0)),
                    "method": row.get("method", "animal_aggregated_global_bootstrap_permutation"),
                    "ci_low": float(row.get("ci_low", np.nan)),
                    "ci_high": float(row.get("ci_high", np.nan)),
                    "p_value": float(row.get("p_value", np.nan)),
                    "evidence_level": int(row.get("evidence_level", 1)),
                }
            )
    except Exception as exc:
        rows.append(
            {
                "analysis_type": "animal_level_treatment_contrast",
                "scenario": cfg.primate_primary_contrast,
                "effect": np.nan,
                "direction": "NA",
                "available": False,
                "estimable": False,
                "reason": f"Animal-level treatment contrast sensitivity unavailable: {exc}",
                "n_used": 0,
                "method": "animal_aggregated_global_bootstrap_permutation",
                "ci_low": np.nan,
                "ci_high": np.nan,
                "evidence_level": 0,
            }
        )

    thresholds = sorted(set(int(x) for x in (cfg.sensitivity_top_feature_thresholds or [])))
    plasma_groups = set(prim_plasma_meta["group"].astype(str))
    plasma_treated = _canonical_group_label(cfg.primate_treated_label)
    plasma_vehicle = _canonical_group_label(getattr(cfg, "primate_vehicle_label", "O_V"))
    plasma_ctrl = [plasma_vehicle] if plasma_vehicle in plasma_groups else []
    for top_n in thresholds:
        try:
            if not plasma_ctrl:
                raise ValueError("prespecified vehicle plasma group is unavailable")
            score_table, _, axis_summary = build_oriented_plasma_aging_axis(
                plasma_expr=prim_plasma_expr,
                plasma_meta=prim_plasma_meta,
                young_groups=getattr(cfg, "plasma_axis_young_labels", ["Y"]),
                old_control_groups=[plasma_vehicle],
                treated_groups=[plasma_treated],
                n_top_proteins=max(5, int(top_n)),
                min_group_samples=int(getattr(cfg, "plasma_axis_min_group_samples", 2)),
                min_non_nan_frac=float(getattr(cfg, "plasma_axis_min_non_nan_frac", 0.8)),
                n_bootstrap=0,
                n_permutations=0,
                random_state=int(getattr(cfg, "random_state", 42)),
            )
            if axis_summary.empty or not bool(axis_summary.iloc[0].get("estimable", False)):
                raise ValueError("reference-trained plasma age-state axis was not estimable")
            pm = score_table.set_index("sample_id").copy()
            trt = pm.loc[
                pm["group_clean"].astype(str).eq(plasma_treated),
                "plasma_age_axis_score",
            ].dropna()
            ctr = pm.loc[
                pm["group_clean"].astype(str).isin(plasma_ctrl),
                "plasma_age_axis_score",
            ].dropna()
            trt_raw = pm.loc[
                pm["group_clean"].astype(str).eq(plasma_treated),
                "raw_pc1_score",
            ].dropna()
            ctr_raw = pm.loc[
                pm["group_clean"].astype(str).isin(plasma_ctrl),
                "raw_pc1_score",
            ].dropna()
            if len(trt) < 2 or len(ctr) < 2:
                raise ValueError("insufficient treated/control plasma samples")

            effect = float(trt.median() - ctr.median())
            rows.append(
                {
                    "analysis_type": "top_features",
                    "scenario": f"plasma_top_{int(top_n)}",
                    "effect": effect,
                    "effect_raw_pc1_diff": float(trt_raw.median() - ctr_raw.median())
                    if (not trt_raw.empty and not ctr_raw.empty)
                    else np.nan,
                    "direction": "negative" if effect < 0 else ("positive" if effect > 0 else "zero"),
                    "available": True,
                    "estimable": True,
                    "reason": "",
                    "n_used": int(len(trt) + len(ctr)),
                    "method": "reference_trained_plasma_age_axis_top_feature_sensitivity",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 1,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "analysis_type": "top_features",
                    "scenario": f"plasma_top_{int(top_n)}",
                    "effect": np.nan,
                    "effect_raw_pc1_diff": np.nan,
                    "direction": "NA",
                    "available": False,
                    "estimable": False,
                    "reason": f"Top-feature sensitivity not estimable: {str(e)}",
                    "n_used": 0,
                    "method": "plasma_state_median_diff_zscore",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                }
            )

    if not rows:
        return pd.DataFrame(
            [
                {
                    "analysis_type": "none",
                    "scenario": "none",
                    "effect": np.nan,
                    "effect_raw_pc1_diff": np.nan,
                    "direction": "NA",
                    "available": False,
                    "estimable": False,
                    "reason": "No sensitivity analyses were generated.",
                    "n_used": 0,
                    "method": "none",
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "evidence_level": 0,
                    "direction_stable": False,
                }
            ]
        )

    out = pd.DataFrame(rows)
    for analysis_type, sub in out.groupby("analysis_type"):
        idx = sub.index
        dirs = [d for d in sub["direction"].tolist() if d in {"positive", "negative"}]
        stable = bool(len(dirs) > 0 and len(set(dirs)) == 1)
        out.loc[idx, "direction_stable"] = stable

    return out


# -----------------------------------------------------------------------------
# Pipeline core
# -----------------------------------------------------------------------------
def run(cfg: PipelineConfig) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    runtime_sanity_banner(cfg)
    logger.info("Pipeline config: %s", asdict(cfg))

    # ---- Primate bulk (tissues) ----
    prim_expr, prim_meta = load_align_standardize(cfg.primate_bulk, cfg, assume_counts=True)
    bulk_input_audit = prim_expr.attrs.get("input_audit")
    if bulk_input_audit is not None:
        bulk_input_audit_path = cfg.results_dir / "omix007580_input_audit.csv"
        pd.DataFrame([bulk_input_audit]).to_csv(bulk_input_audit_path, index=False)
        logger.info("Saved OMIX007580 input audit to: %s", bulk_input_audit_path)
    # Tissue-level bulk effects are computed later, after rejuvenation_score exists.
    prim_tissue_effects = None
    prim_meth_expr, prim_meth_meta = None, None
    mouse_tissue_effects = pd.DataFrame(columns=["mean_effect", "n_treated", "n_control"])
    # --- Check animal_id on prim_meta ---
    if "animal_id" not in prim_meta.columns or prim_meta["animal_id"].isna().all():
        animal_candidates = getattr(cfg, "animal_id_col_candidates", [])
        col = find_first_present_column(prim_meta, animal_candidates)
        if col is not None:
            prim_meta = prim_meta.copy()
            s = prim_meta[col].astype("string").str.strip()
            prim_meta["animal_id"] = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        else:
            logger.warning("No animal_id-like column found in prim_meta.")

    # ---- Optional: primate methylation (Mammal40) ----
    if getattr(cfg, "enable_methylation_block", False):
        prim_meth_expr, prim_meth_meta = load_methylation_block(cfg)
        if prim_meth_expr is None:
            logger.info("Methylation block enabled, but data not usable; skipping.")
        else:
            logger.info("Methylation block loaded (CpGs x samples): %s", prim_meth_expr.shape)
    else:
        logger.info("Methylation block disabled by config (enable_methylation_block=False).")

    # ---- Age handling and transcriptomic clock ----
    prim_meta = ensure_numeric_age(prim_meta)

    if "age" not in prim_meta.columns:
        raise ValueError(
            "No usable numeric age column could be derived. "
            "Check OMIX007580 metadata and age-related fields."
        )

    if prim_meta["age"].notna().sum() < 8:
        raise ValueError("Too few numeric age values to train a transcriptomic clock.")

    logger.info(
        "Age validity rate: %.3f",
        pd.to_numeric(prim_meta["age"], errors="coerce").notna().mean(),
    )

    clock_exclusions_path = cfg.results_dir / "clock_exclusions_primates.csv"
    prim_expr, prim_meta, clock_exclusions = apply_exact_sample_exclusions(
        prim_expr,
        prim_meta,
        getattr(cfg, "clock_excluded_sample_ids", []),
    )
    clock_exclusions.to_csv(clock_exclusions_path, index=False)
    logger.info("Saved transcriptomic-clock exclusion ledger to: %s", clock_exclusions_path)

    clock_protocol = getattr(cfg, "clock_protocol", "legacy_all_samples_ridge")
    if clock_protocol == "reference_nested_ridge":
        fold_path = getattr(cfg, "clock_fold_assignments", None)
        fold_sha256 = getattr(cfg, "clock_fold_assignments_sha256", None)
        if fold_path is None or fold_sha256 is None:
            raise ValueError("Reference clock requires a frozen fold-assignment contract")
        frozen_clock_folds = pd.read_csv(fold_path)
        if frozen_fold_assignment_digest(frozen_clock_folds) != fold_sha256:
            raise ValueError("Frozen primate clock fold-assignment content changed")
        reference_run = train_reference_transcriptomic_clock(
            prim_expr,
            prim_meta,
            reference_groups=getattr(cfg, "clock_reference_groups", []),
            ridge_alphas=getattr(cfg, "clock_ridge_alphas", []),
            outer_folds=getattr(cfg, "clock_cv_folds", 5),
            inner_folds=getattr(cfg, "clock_inner_folds", 3),
            inner_seed=getattr(cfg, "clock_inner_seed", 20260917),
            diagnostic_seeds=getattr(cfg, "clock_diagnostic_seeds", [17, 29, 43]),
            frozen_fold_assignments=frozen_clock_folds,
        )
        prim_clock = reference_run.clock
        prim_cv_pred_df = reference_run.primary_predictions
        clock_metrics = reference_run.metrics
        reference_run.all_predictions.to_csv(
            cfg.results_dir / "clock_cv_predictions_primates.csv",
            index=False,
        )
        reference_run.fold_audit.to_csv(
            cfg.results_dir / "clock_fit_audit_primates.csv",
            index=False,
        )
        reference_run.hyperparameter_audit.to_csv(
            cfg.results_dir / "clock_hyperparameter_audit_primates.csv",
            index=False,
        )
        clock_method = "controls_only_nested_grouped_ridge"
    elif clock_protocol == "legacy_all_samples_ridge":
        prim_clock, prim_cv_pred_df, clock_metrics = train_transcriptomic_clock(
            prim_expr,
            prim_meta,
            age_col="age",
            model=getattr(cfg, "clock_model", None),
            n_splits=getattr(cfg, "clock_cv_folds", 5),
            random_state=getattr(cfg, "random_state", getattr(cfg, "random_seed", 42)),
            cv_group_col="animal_id",
        )
        clock_method = "legacy_all_samples_transcriptomic_clock_cv"
    else:
        raise ValueError(f"Unsupported transcriptomic-clock protocol: {clock_protocol}")

    logger.info("Transcriptomic clock metrics (train CV): %s", clock_metrics)

    # (optional) log / save metrics
    clock_metrics_path = cfg.results_dir / "clock_metrics_primates.csv"
    clock_df = pd.DataFrame([clock_metrics])
    clock_df = _ensure_standard_schema(
        clock_df,
        available=True,
        estimable=True,
        reason="",
        n_used=int(clock_metrics.get("n_samples", 0)),
        method=clock_method,
        evidence_level=1,
    )
    clock_df.to_csv(clock_metrics_path, index=False)

    # In-sample predictions from the final model (can be useful later, but are
    # NOT used for performance metrics to avoid optimistic bias).
    prim_cv_pred_df = prim_cv_pred_df.rename(columns={"predicted_age": "predicted_age_cv"})
    prim_meta = prim_meta.merge(
        prim_cv_pred_df[["sample_id", "predicted_age_cv"]],
        on="sample_id",
        how="left",
    )

    # 2) Cross-validated predictions + honest performance metrics
    prim_pred_full = predict_biological_age(
        prim_clock,
        prim_expr,
        prim_meta,
    )
    prim_meta = prim_meta.merge(
        prim_pred_full[["sample_id", "predicted_age"]],
        on="sample_id",
        how="left",
    )

    try:
        plot_age_scatter(
            prim_meta,
            chrono_col="age",
            pred_col="predicted_age_cv",
            out_path=cfg.figures_dir / "primates_age_scatter.png",
        )
    except Exception as e:
        logger.warning("plot_age_scatter failed: %s", e)

    # Use cross-validated predictions for rejuvenation score if available,
    # otherwise fall back to in-sample predictions.
    pred_col_for_rejuvenation = "predicted_age_cv" if "predicted_age_cv" in prim_meta.columns else "predicted_age"

    prim_meta = build_proxy_rejuvenation_score(
        prim_meta,
        pred_age_col=pred_col_for_rejuvenation,
        chrono_age_col="age",
        out_col="rejuvenation_score",
    )

    # Rejuvenation score distribution by group
    try:
        if "group" in prim_meta.columns:
            plot_rejuvenation_by_group(
                prim_meta,
                group_col="group",
                rejuvenation_col="rejuvenation_score",
                out_path=cfg.figures_dir / "primates_rejuvenation_by_group.png",
                tissue_col="tissue" if "tissue" in prim_meta.columns else None,
            )
    except Exception as e:
        logger.warning("plot_rejuvenation_by_group failed: %s", e)

    prim_meta = compute_delta_age(
        prim_meta,
        pred_age_col=pred_col_for_rejuvenation,
        chrono_age_col="age",
        out_col="delta_age",
    )

    # Prespecified animal-level treatment contrasts. The global summary is
    # secondary because it averages repeated tissues within each animal.
    tissue_col = cfg.tissue_col_candidates[0]
    treatment_contrasts = list(cfg.primate_treatment_contrasts or [])
    global_rejuv = summarize_global_animal_treatment_contrasts(
        prim_meta,
        tissue_col=tissue_col,
        group_col=cfg.group_col_candidates[0],
        animal_col="animal_id",
        sex_col="sex",
        value_col="delta_age",
        contrasts=treatment_contrasts,
        primary_contrast=cfg.primate_primary_contrast,
        min_per_group=max(2, cfg.min_samples_per_group_for_rejuv),
        n_bootstrap=cfg.n_bootstrap,
        n_permutations=cfg.treatment_n_permutations,
        random_state=cfg.random_state,
    )
    if global_rejuv.empty:
        global_rejuv = _result_stub(
            reason="No global animal-level treatment contrasts passed minimum support.",
            method="animal_aggregated_global_bootstrap_permutation",
            extra={
                "contrast": cfg.primate_primary_contrast,
                "scope": "global_secondary",
                "mean_effect": np.nan,
                "p_value": np.nan,
            },
        )
    global_rejuv = _ensure_standard_schema(
        global_rejuv,
        method="animal_aggregated_global_bootstrap_permutation",
    )
    global_rejuv_path = cfg.results_dir / "global_rejuvenation_summary.csv"
    global_rejuv.to_csv(global_rejuv_path, index=False)
    logger.info("Saved secondary animal-level global summary to: %s", global_rejuv_path)

    # Tissue-wise effects use one biological animal per row and retain all
    # three prespecified contrasts. Public figures select the primary contrast.
    rejuv_by_tissue = summarize_animal_treatment_contrasts_by_tissue(
        prim_meta,
        tissue_col=tissue_col,
        group_col=cfg.group_col_candidates[0],
        animal_col="animal_id",
        sex_col="sex",
        value_col="delta_age",
        contrasts=treatment_contrasts,
        primary_contrast=cfg.primate_primary_contrast,
        min_per_group=max(2, cfg.min_samples_per_group_for_rejuv),
        n_bootstrap=cfg.n_bootstrap,
        n_permutations=cfg.treatment_n_permutations,
        random_state=cfg.random_state,
    )

    rejuv_by_tissue_path = cfg.results_dir / "rejuvenation_by_tissue.csv"
    if rejuv_by_tissue.empty:
        rejuv_by_tissue = pd.DataFrame(
            [
                {
                    "tissue": "NA",
                    "contrast": cfg.primate_primary_contrast,
                    "is_primary": True,
                    "n_ctrl": 0,
                    "n_trt": 0,
                    "mean_effect": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "ci_crosses_zero": False,
                    "p_value_adjusted": np.nan,
                    "available": False,
                    "estimable": False,
                    "reason": "No tissue-level rejuvenation estimates passed filters.",
                    "n_used": 0,
                    "method": "animal_mean_difference_bootstrap_permutation_by_tissue",
                    "evidence_level": 0,
                }
            ]
        )
    rejuv_by_tissue = _ensure_standard_schema(
        rejuv_by_tissue,
        method="animal_mean_difference_bootstrap_permutation_by_tissue",
    )
    rejuv_by_tissue.to_csv(rejuv_by_tissue_path, index=False)
    logger.info("Saved tissue-level rejuvenation summary to: %s", rejuv_by_tissue_path)

    primary_rejuv_by_tissue = rejuv_by_tissue.loc[
        rejuv_by_tissue.get("is_primary", pd.Series(False, index=rejuv_by_tissue.index)).astype(bool)
        & rejuv_by_tissue["estimable"].astype(bool)
    ].copy()

    clustered_sensitivity = summarize_clustered_treatment_sensitivity(
        prim_meta,
        tissue_col=tissue_col,
        group_col=cfg.group_col_candidates[0],
        animal_col="animal_id",
        sex_col="sex",
        age_col="age",
        value_col="delta_age",
        contrasts=treatment_contrasts,
        primary_contrast=cfg.primate_primary_contrast,
        min_animals_per_group=max(2, cfg.min_samples_per_group_for_rejuv),
    )
    if clustered_sensitivity.empty:
        clustered_sensitivity = _result_stub(
            reason="No tissue-adjusted animal-clustered sensitivity contrast was estimable.",
            method="ols_tissue_age_sex_animal_clustered",
            extra={
                "contrast": cfg.primate_primary_contrast,
                "mean_effect": np.nan,
                "cluster_unit": "animal_id",
            },
        )
    clustered_sensitivity = _ensure_standard_schema(
        clustered_sensitivity,
        method="ols_tissue_age_sex_animal_clustered",
    )
    clustered_sensitivity_path = cfg.results_dir / "global_rejuvenation_clustered_sensitivity.csv"
    clustered_sensitivity.to_csv(clustered_sensitivity_path, index=False)
    logger.info("Saved animal-clustered sensitivity to: %s", clustered_sensitivity_path)

    # ---- Tissue-level expression effects (uses rejuvenation_score) ----
    tissue_expr_effects = summarize_tissue_expression_effects(
        prim_expr,
        prim_meta,
        tissue_col=cfg.tissue_col_candidates[0],
        group_col=cfg.group_col_candidates[0],
        control_labels=[cfg.primate_vehicle_label],
        treated_labels=[cfg.primate_treated_label],
        outcome_col="delta_age",
        min_per_group=cfg.min_samples_per_group_for_rejuv,
        covariate_cols=getattr(cfg, "tissue_effect_covariates", ["age", "sex", "batch"]),
    )

    tissue_expr_path = cfg.results_dir / "tissue_expression_effects.csv"
    if tissue_expr_effects.empty:
        logger.warning("No tissue expression effects were computed (empty DataFrame). Writing stub.")
        tissue_expr_effects = pd.DataFrame(
            [
                {
                    "tissue": "NA",
                    "n_ctrl": 0,
                    "n_trt": 0,
                    "n_samples": 0,
                    "n_used": 0,
                    "n_genes_modeled": 0,
                    "method": "ols_sample_level_outcome_by_tissue",
                    "covariates_used": "none",
                    "top_genes": "",
                    "mean_effect": None,
                    "median_effect": None,
                    "effect_se": None,
                    "ci_low": None,
                    "ci_high": None,
                    "p_value": None,
                    "fdr_q_value": None,
                    "available": False,
                    "estimable": False,
                    "reason": "No tissues passed min_per_group/covariate filters.",
                    "evidence_level": 0,
                }
            ]
        )
    else:
        tissue_expr_effects["evidence_level"] = 1
    tissue_expr_effects = annotate_effect_uncertainty(tissue_expr_effects, effect_col="mean_effect")
    tissue_expr_effects = _ensure_standard_schema(
        tissue_expr_effects,
        method="ols_sample_level_outcome_by_tissue",
    )
    tissue_expr_effects.to_csv(tissue_expr_path, index=False)
    logger.info("Saved tissue expression effects to: %s", tissue_expr_path)

    # ---- Primary tissue effects for downstream exosome comparison ----
    if not primary_rejuv_by_tissue.empty:
        prim_tissue_effects = _table_to_effect_index(
            primary_rejuv_by_tissue,
            tissue_col="tissue",
            effect_col="mean_effect",
            treated_col="n_trt",
            control_col="n_ctrl",
        )
    else:
        prim_tissue_effects = pd.DataFrame(columns=["mean_effect", "n_treated", "n_control"])

    # ---- Mouse exosome mechanism-support block (OMIX009283) ----
    mouse_block_reason = (
        "Mouse exosome block disabled by config for this data profile."
        if not getattr(cfg, "enable_mouse_exosome_block", False)
        else "Mouse exosome input data unavailable."
    )
    mouse_exosome_effects = _result_stub(
        reason=mouse_block_reason,
        method="mouse_tissue_clock_contrast",
        extra={"tissue": "NA", "contrast": "NA", "treated_arm": "NA", "control_arm": "NA"},
    )
    mouse_signature_summary = _result_stub(
        reason=mouse_block_reason,
        method="mouse_exosome_signature_summary",
        extra={"contrast": "NA"},
    )
    mouse_sample_outcomes = pd.DataFrame()
    if getattr(cfg, "enable_mouse_exosome_block", False) and getattr(cfg, "mouse_exosome_bulk", None) is not None:
        try:
            mouse_raw = load_omix_matrix(cfg.mouse_exosome_bulk.matrix, dtype="float32")
            mouse_meta = build_mouse_exosome_metadata(list(mouse_raw.columns))
            mouse_raw, mouse_meta = align_matrix_and_metadata(
                mouse_raw,
                mouse_meta,
                ["sample_id"],
            )
            mouse_expr = log1p_counts(mouse_raw, assume_counts=True)
            mouse_exosome_effects, mouse_sample_outcomes = compute_mouse_exosome_tissue_effects(
                mouse_expr,
                mouse_meta,
                reference_arms=getattr(cfg, "mouse_reference_arms", ["Baseline"]),
                contrasts=_contrast_pairs(getattr(cfg, "mouse_contrasts", ["GES_vs_Veh", "WT_vs_Veh", "GES_vs_WT"])),
                min_reference_samples=int(getattr(cfg, "mouse_min_reference_samples", 20)),
                min_reference_ages=int(getattr(cfg, "mouse_min_reference_ages", 4)),
                min_samples_per_group=int(getattr(cfg, "mouse_min_samples_per_group", 3)),
                n_bootstrap=int(getattr(cfg, "mouse_tissue_bootstrap", 1000)),
                n_permutations=int(getattr(cfg, "mouse_tissue_permutations", 1000)),
                random_state=int(getattr(cfg, "random_state", getattr(cfg, "random_seed", 42))),
                return_sample_outcomes=True,
            )
            mouse_exosome_effects = _ensure_standard_schema(
                mouse_exosome_effects,
                method="mouse_tissue_clock_contrast",
            )
            mouse_signature_summary = _ensure_standard_schema(
                summarize_mouse_exosome_signatures(mouse_exosome_effects),
                method="mouse_exosome_signature_summary",
            )

            primary_mouse = mouse_exosome_effects.loc[
                (mouse_exosome_effects["contrast"] == "GES_vs_Veh")
                & mouse_exosome_effects["estimable"].astype(bool)
            ].copy()
            mouse_tissue_effects = _table_to_effect_index(
                primary_mouse,
                tissue_col="tissue",
                effect_col="mean_effect",
                treated_col="n_treated",
                control_col="n_control",
            )
            if not mouse_tissue_effects.empty:
                mapped_index = [
                    getattr(cfg, "mouse_to_primate_tissue_map", {}).get(str(t), str(t))
                    for t in mouse_tissue_effects.index
                ]
                mouse_tissue_effects = mouse_tissue_effects.copy()
                mouse_tissue_effects.index = mapped_index
                mouse_tissue_effects = mouse_tissue_effects[~mouse_tissue_effects.index.duplicated(keep="first")]
        except Exception as e:
            logger.warning("Mouse exosome block failed: %s", e)
            mouse_exosome_effects = _result_stub(
                reason=f"Mouse exosome block failed: {e}",
                method="mouse_tissue_clock_contrast",
                extra={"tissue": "NA", "contrast": "NA", "treated_arm": "NA", "control_arm": "NA"},
            )
            mouse_signature_summary = _result_stub(
                reason=f"Mouse exosome block failed: {e}",
                method="mouse_exosome_signature_summary",
                extra={"contrast": "NA"},
            )
            mouse_tissue_effects = pd.DataFrame(columns=["mean_effect", "n_treated", "n_control"])
    else:
        logger.info("Mouse exosome block disabled by config (enable_mouse_exosome_block=False).")

    mouse_effects_path = cfg.results_dir / "mouse_exosome_effects.csv"
    mouse_exosome_effects.to_csv(mouse_effects_path, index=False)
    logger.info("Saved mouse exosome tissue effects to: %s", mouse_effects_path)

    mouse_signature_path = cfg.results_dir / "mouse_exosome_signature_summary.csv"
    mouse_signature_summary.to_csv(mouse_signature_path, index=False)
    logger.info("Saved mouse exosome signature summary to: %s", mouse_signature_path)

    # ---- Methylation validation block ----
    methylation_rejuv = _result_stub(
        reason="Methylation validation not available.",
        method="methylation_stage_proxy_clock",
        extra={"tissue": "NA"},
    )
    multimodal_concordance = _result_stub(
        reason="Multimodal concordance not available.",
        method="multimodal_concordance",
        extra={"n_common_tissues": 0},
    )
    if prim_meth_expr is not None and prim_meth_meta is not None:
        try:
            meth_meta = prim_meth_meta.copy()
            meth_meta["sample_id"] = meth_meta["sample_id"].astype(str)
            meth_meta["age_proxy_years"] = assign_group_stage_age(
                meth_meta["group"].astype(str),
                getattr(cfg, "methylation_group_age_map", {}),
            )
            meth_meta["tissue"] = meth_meta["tissue"].map(
                lambda x: getattr(cfg, "methylation_to_primate_tissue_map", {}).get(str(x), str(x))
            )
            meth_meta = meth_meta.loc[meth_meta["age_proxy_years"].notna()].copy()

            meth_clock, meth_cv_pred_df, meth_metrics = train_transcriptomic_clock(
                prim_meth_expr,
                meth_meta[["sample_id", "age_proxy_years"]].copy(),
                age_col="age_proxy_years",
                model=getattr(cfg, "clock_model", None),
                n_splits=min(5, max(2, int(meth_meta["group"].nunique()))),
                random_state=int(getattr(cfg, "random_state", getattr(cfg, "random_seed", 42))),
                cv_group_col=None,
            )
            meth_cv_pred_df = meth_cv_pred_df.rename(columns={"predicted_age": "predicted_age_cv"})
            meth_meta = meth_meta.merge(
                meth_cv_pred_df[["sample_id", "predicted_age_cv"]],
                on="sample_id",
                how="left",
            )
            meth_pred_full = predict_biological_age(meth_clock, prim_meth_expr, meth_meta)
            meth_meta = meth_meta.merge(
                meth_pred_full[["sample_id", "predicted_age"]],
                on="sample_id",
                how="left",
            )
            meth_meta = compute_delta_age(
                meth_meta,
                pred_age_col="predicted_age_cv" if "predicted_age_cv" in meth_meta.columns else "predicted_age",
                chrono_age_col="age_proxy_years",
                out_col="delta_age",
            )
            methylation_rejuv = summarize_rejuvenation_by_tissue(
                meth_meta,
                tissue_col="tissue",
                group_col="group",
                value_col="delta_age",
                control_labels=[cfg.primate_vehicle_label],
                treated_labels=[cfg.primate_treated_label],
                min_per_group=max(2, cfg.min_samples_per_group_for_rejuv),
                n_bootstrap=max(500, cfg.n_bootstrap // 2),
                random_state=cfg.random_state,
            )
            if methylation_rejuv.empty:
                methylation_rejuv = _result_stub(
                    reason="No methylation tissues passed validation filters.",
                    method="methylation_stage_proxy_clock",
                    extra={"tissue": "NA"},
                )
            else:
                methylation_rejuv["n_used"] = (
                    pd.to_numeric(methylation_rejuv.get("n_ctrl", 0), errors="coerce").fillna(0).astype(int)
                    + pd.to_numeric(methylation_rejuv.get("n_trt", 0), errors="coerce").fillna(0).astype(int)
                )
                methylation_rejuv["method"] = "methylation_stage_proxy_clock"
                methylation_rejuv["available"] = True
                methylation_rejuv["estimable"] = True
                methylation_rejuv["reason"] = (
                    "DNAmAge proxy trained on stage-mapped methylation samples."
                )
                methylation_rejuv["evidence_level"] = 1
            methylation_rejuv = annotate_effect_uncertainty(methylation_rejuv, effect_col="effect_median")
            methylation_rejuv = _ensure_standard_schema(
                methylation_rejuv,
                method="methylation_stage_proxy_clock",
            )

            multimodal_concordance = _ensure_standard_schema(
                summarize_multimodal_concordance(
                    primary_rejuv_by_tissue,
                    methylation_rejuv.loc[methylation_rejuv["estimable"].astype(bool)],
                    min_common_tissues=int(getattr(cfg, "methylation_min_common_tissues", 3)),
                    n_bootstrap=int(getattr(cfg, "n_bootstrap", 2000)),
                    n_permutations=int(getattr(cfg, "exosome_fraction_permutations", 1000)),
                    random_state=int(getattr(cfg, "random_state", getattr(cfg, "random_seed", 42))),
                ),
                method="multimodal_concordance",
            )
            logger.info("Methylation validation metrics: %s", meth_metrics)
        except Exception as e:
            logger.warning("Methylation validation failed: %s", e)
            methylation_rejuv = _result_stub(
                reason=f"Methylation validation failed: {e}",
                method="methylation_stage_proxy_clock",
                extra={"tissue": "NA"},
            )
            multimodal_concordance = _result_stub(
                reason=f"Methylation validation failed: {e}",
                method="multimodal_concordance",
                extra={"n_common_tissues": 0},
            )

    methylation_rejuv_path = cfg.results_dir / "methylation_rejuvenation_by_tissue.csv"
    methylation_rejuv.to_csv(methylation_rejuv_path, index=False)
    logger.info("Saved methylation rejuvenation summary to: %s", methylation_rejuv_path)

    multimodal_concordance_path = cfg.results_dir / "multimodal_concordance_summary.csv"
    multimodal_concordance.to_csv(multimodal_concordance_path, index=False)
    logger.info("Saved multimodal concordance summary to: %s", multimodal_concordance_path)

    # ---- Targeted subset-validation audits ----
    ovary_subset_validation = _result_stub(
        reason="Subset validation block disabled or unavailable.",
        method="subset_sample_info_audit",
        extra={"subset": "ovary", "bulk_tissue": "Ovary"},
    )
    hippocampus_subset_validation = _result_stub(
        reason="Subset validation block disabled or unavailable.",
        method="subset_sample_info_audit",
        extra={"subset": "hippocampus", "bulk_tissue": "Hippocampus"},
    )
    if getattr(cfg, "enable_subset_validation_block", False):
        try:
            data_root = cfg.primate_bulk.matrix.parent
            ovary_info = read_subset_sample_info(data_root / "OMIX007583-01.zip")
            ovary_subset_validation = _ensure_standard_schema(
                build_subset_validation_table(
                    ovary_info,
                    subset_name="ovary",
                    bulk_tissue="Ovary",
                    group_map={
                        "Y": "Y_C",
                        "M": "M_C",
                        "O": "O_C",
                        "O_V": "O_V",
                        "O_WT": "O_WT",
                        "O_GES": "O_GES",
                    },
                    bulk_effects=primary_rejuv_by_tissue,
                ),
                method="subset_sample_info_audit",
            )
        except Exception as e:
            logger.warning("Ovary subset validation failed: %s", e)
            ovary_subset_validation = _result_stub(
                reason=f"Ovary subset validation failed: {e}",
                method="subset_sample_info_audit",
                extra={"subset": "ovary", "bulk_tissue": "Ovary"},
            )

        try:
            data_root = cfg.primate_bulk.matrix.parent
            hippocampus_info = read_subset_sample_info(data_root / "OMIX007586-02.zip")
            hippocampus_subset_validation = _ensure_standard_schema(
                build_subset_validation_table(
                    hippocampus_info,
                    subset_name="hippocampus",
                    bulk_tissue="Hippocampus",
                    group_map={
                        "A1_Ctrl": "Y_C",
                        "A2_Ctrl": "M_C",
                        "A3_Ctrl": "O_C",
                        "A4_Ctrl": "O_V",
                        "A4_WTC": "O_WT",
                        "A4_SRC": "O_GES",
                    },
                    bulk_effects=primary_rejuv_by_tissue,
                ),
                method="subset_sample_info_audit",
            )
        except Exception as e:
            logger.warning("Hippocampus subset validation failed: %s", e)
            hippocampus_subset_validation = _result_stub(
                reason=f"Hippocampus subset validation failed: {e}",
                method="subset_sample_info_audit",
                extra={"subset": "hippocampus", "bulk_tissue": "Hippocampus"},
            )

    ovary_subset_path = cfg.results_dir / "ovary_subset_validation.csv"
    ovary_subset_validation.to_csv(ovary_subset_path, index=False)
    logger.info("Saved ovary subset validation to: %s", ovary_subset_path)

    hippocampus_subset_path = cfg.results_dir / "hippocampus_subset_validation.csv"
    hippocampus_subset_validation.to_csv(hippocampus_subset_path, index=False)
    logger.info("Saved hippocampus subset validation to: %s", hippocampus_subset_path)

    # ---- Primate plasma (OMIX007581 wide proteomics CSV with embedded sample IDs) ----
    plasma_path = cfg.primate_plasma.matrix

    if plasma_path.is_dir():
        candidates = sorted(plasma_path.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in {plasma_path}")
        plasma_file = candidates[0]
    else:
        plasma_file = plasma_path

    prim_plasma_expr, plasma_feature_annotations = load_plasma_proteomics_csv(
        plasma_file,
        return_annotations=True,
    )
    prim_plasma_meta = build_plasma_metadata_from_columns(list(prim_plasma_expr.columns))
    prim_plasma_expr = clean_plasma_matrix(
        prim_plasma_expr,
        min_feature_non_nan_frac=0.5,
        min_sample_non_nan_frac=0.5,
    )
    prim_plasma_meta["sample_id"] = prim_plasma_meta["sample_id"].astype(str)
    bulk_valid_ids = _clean_string_set(
        prim_meta["animal_id"] if "animal_id" in prim_meta.columns else pd.Series(dtype="string")
    )

    # Enrich plasma metadata with animal_id only if optional metadata is provided.
    plasma_metadata_path = getattr(cfg.primate_plasma, "metadata", None)
    if plasma_metadata_path is None:
        logger.info(
            "OMIX007581 metadata not provided; proceeding with column-derived sample metadata only."
        )
    elif not Path(plasma_metadata_path).exists():
        logger.info(
            "OMIX007581 metadata path not found (%s); proceeding with column-derived sample metadata only.",
            plasma_metadata_path,
        )
    else:
        try:
            plasma_meta_raw = load_omix_metadata(plasma_metadata_path)

            sample_col = find_first_present_column(plasma_meta_raw, cfg.sample_id_col_candidates)
            animal_col = find_first_present_column(plasma_meta_raw, cfg.animal_id_col_candidates)

            if sample_col and animal_col:
                tmp = plasma_meta_raw[[sample_col, animal_col]].copy()
                tmp.columns = ["sample_id", "animal_id_metadata"]
                tmp["sample_id"] = tmp["sample_id"].astype(str).str.strip()
                tmp["animal_id_metadata"] = (
                    tmp["animal_id_metadata"]
                    .astype(str)
                    .str.strip()
                    .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
                )
                tmp = tmp.dropna(subset=["sample_id", "animal_id_metadata"])
                if tmp["sample_id"].duplicated().any():
                    raise ValueError(
                        "Optional OMIX007581 metadata contains duplicate sample_id values; "
                        "candidate linkage was ignored."
                    )

                prim_plasma_meta = prim_plasma_meta.merge(tmp, on="sample_id", how="left")
                meta_mask = prim_plasma_meta["animal_id_metadata"].notna()
                if meta_mask.any():
                    prim_plasma_meta.loc[meta_mask, "animal_id"] = prim_plasma_meta.loc[meta_mask, "animal_id_metadata"]
                    prim_plasma_meta.loc[meta_mask, "animal_id_source"] = "unverified_metadata"
                    prim_plasma_meta.loc[meta_mask, "animal_id_confidence"] = "unverified_metadata"
                    prim_plasma_meta.loc[meta_mask, "mapping_rule"] = "unverified_metadata_sample_id_match"
                    prim_plasma_meta.loc[meta_mask, "mapping_reason"] = (
                        "Candidate from optional OMIX007581 metadata; not validated biological identity"
                    )
                prim_plasma_meta = prim_plasma_meta.drop(columns=["animal_id_metadata"], errors="ignore")

                if prim_plasma_meta["animal_id"].isna().all():
                    logger.warning(
                        "Plasma metadata merge on sample_id succeeded but all animal_id are NaN; "
                        "check OMIX007581 metadata and sample ID mapping."
                    )
            else:
                logger.warning(
                    "Could not find sample_id/animal_id columns in OMIX007581 metadata; "
                    "skipping animal_id enrichment for plasma."
                )

        except Exception as e:
            logger.warning("Could not enrich prim_plasma_meta with OMIX007581 metadata: %s", e)

    # Fill unresolved animal_id rows using conservative deterministic rule.
    if "animal_id" not in prim_plasma_meta.columns:
        prim_plasma_meta["animal_id"] = pd.NA
    for col, default in {
        "animal_id_source": "unresolved",
        "animal_id_confidence": "low",
        "mapping_rule": "none",
        "mapping_reason": "unresolved",
    }.items():
        if col not in prim_plasma_meta.columns:
            prim_plasma_meta[col] = default
        else:
            prim_plasma_meta[col] = prim_plasma_meta[col].fillna(default)

    unresolved_mask = prim_plasma_meta["animal_id"].isna()
    if unresolved_mask.any():
        for idx, sid in prim_plasma_meta.loc[unresolved_mask, "sample_id"].items():
            mapped_id, mapping_rule, mapping_conf, mapping_reason = _map_plasma_sample_to_bulk_animal_id(sid)
            if mapped_id is not None:
                prim_plasma_meta.at[idx, "animal_id"] = mapped_id
                prim_plasma_meta.at[idx, "animal_id_source"] = "deterministic_rule"
            else:
                prim_plasma_meta.at[idx, "animal_id_source"] = "unresolved"
            prim_plasma_meta.at[idx, "animal_id_confidence"] = mapping_conf
            prim_plasma_meta.at[idx, "mapping_rule"] = mapping_rule
            prim_plasma_meta.at[idx, "mapping_reason"] = mapping_reason

    for column in ("linkage_evidence_source", "linkage_manifest_sha256"):
        if column not in prim_plasma_meta.columns:
            prim_plasma_meta[column] = pd.NA

    linkage_manifest_path = getattr(cfg, "plasma_linkage_manifest", None)
    linkage_manifest_sha256 = getattr(cfg, "plasma_linkage_manifest_sha256", None)
    linkage_manifest_reason = "No validated plasma linkage manifest was configured."
    linkage_manifest_accepted = False
    n_validated_manifest_links = 0
    observed_manifest_sha256 = ""
    if linkage_manifest_path is not None:
        try:
            validated_links = load_validated_plasma_linkage_manifest(
                linkage_manifest_path,
                expected_sha256=linkage_manifest_sha256,
                plasma_meta=prim_plasma_meta,
                primate_meta=prim_meta,
            )
            validated_by_sample = validated_links.set_index("sample_id")
            validated_mask = prim_plasma_meta["sample_id"].isin(validated_by_sample.index)
            for target, source in {
                "animal_id": "animal_id",
                "animal_id_source": "animal_id_source",
                "animal_id_confidence": "animal_id_confidence",
                "mapping_rule": "mapping_rule",
                "mapping_reason": "mapping_reason",
                "linkage_evidence_source": "evidence_source",
                "linkage_manifest_sha256": "linkage_manifest_sha256",
            }.items():
                prim_plasma_meta.loc[validated_mask, target] = prim_plasma_meta.loc[
                    validated_mask, "sample_id"
                ].map(validated_by_sample[source])
            linkage_manifest_accepted = True
            n_validated_manifest_links = int(len(validated_links))
            observed_manifest_sha256 = str(
                validated_links["linkage_manifest_sha256"].iloc[0]
            )
            linkage_manifest_reason = ""
        except Exception as exc:
            linkage_manifest_reason = f"Validated plasma linkage manifest rejected: {exc}"
            logger.warning("%s", linkage_manifest_reason)

    linkage_manifest_audit = _ensure_standard_schema(
        pd.DataFrame(
            [{
                "manifest_configured": linkage_manifest_path is not None,
                "manifest_accepted": linkage_manifest_accepted,
                "manifest_name": (
                    Path(linkage_manifest_path).name
                    if linkage_manifest_path is not None
                    else ""
                ),
                "expected_sha256": str(linkage_manifest_sha256 or ""),
                "observed_sha256": observed_manifest_sha256,
                "n_validated_links": n_validated_manifest_links,
            }]
        ),
        available=linkage_manifest_path is not None,
        estimable=linkage_manifest_accepted,
        reason=linkage_manifest_reason,
        reason_code=(
            None if linkage_manifest_accepted else PLASMA_LINKAGE_CONFIDENCE_MISSING
        ),
        n_used=n_validated_manifest_links,
        method="validated_plasma_linkage_manifest_contract",
        evidence_level=2 if linkage_manifest_accepted else 0,
    )
    linkage_manifest_audit_path = cfg.results_dir / "plasma_linkage_manifest_audit.csv"
    linkage_manifest_audit.to_csv(linkage_manifest_audit_path, index=False)
    logger.info("Saved plasma linkage manifest audit to: %s", linkage_manifest_audit_path)

    # Validate candidate animal_id against known bulk animal IDs.
    if bulk_valid_ids:
        is_mapped = prim_plasma_meta["animal_id"].notna()
        in_bulk = (
            prim_plasma_meta["animal_id"]
            .astype(str)
            .str.strip()
            .isin(bulk_valid_ids)
        )
        invalid_mask = is_mapped & (~in_bulk)
        if invalid_mask.any():
            prim_plasma_meta.loc[invalid_mask, "animal_id"] = pd.NA
            prim_plasma_meta.loc[invalid_mask, "animal_id_source"] = "invalid"
            prim_plasma_meta.loc[invalid_mask, "animal_id_confidence"] = "low"
            prim_plasma_meta.loc[invalid_mask, "mapping_reason"] = (
                prim_plasma_meta.loc[invalid_mask, "mapping_reason"].astype(str)
                + "; mapped animal_id absent from bulk metadata"
            )

    # Persist explicit mapping table for transparency and downstream audits.
    high_conf_values = tuple(
        getattr(cfg, "linkage_high_conf_values", ("high", "metadata_exact"))
    )
    high_conf_set = {str(x).strip().lower() for x in high_conf_values}
    map_df = prim_plasma_meta.copy()
    map_df["valid_in_bulk"] = (
        map_df["animal_id"].notna()
        & map_df["animal_id"].astype(str).str.strip().isin(bulk_valid_ids)
    )
    map_df["high_conf_link"] = (
        map_df["animal_id_confidence"].astype(str).str.strip().str.lower().isin(high_conf_set)
        & map_df["valid_in_bulk"]
    )
    map_cols = [
        "sample_id",
        "raw_code",
        "group",
        "group_code",
        "sex",
        "replicate",
        "animal_id",
        "animal_id_source",
        "animal_id_confidence",
        "mapping_rule",
        "mapping_reason",
        "linkage_evidence_source",
        "linkage_manifest_sha256",
        "valid_in_bulk",
        "high_conf_link",
    ]
    for c in map_cols:
        if c not in map_df.columns:
            map_df[c] = pd.NA
    map_df = map_df[map_cols]
    map_df = _ensure_standard_schema(
        map_df,
        available=True,
        estimable=bool(map_df["high_conf_link"].any()),
        reason=(
            ""
            if bool(map_df["high_conf_link"].any())
            else "No high-confidence plasma-to-animal links could be established."
        ),
        n_used=int(len(map_df)),
        method="plasma_linkage_evidence_mapping",
        evidence_level=2 if bool(map_df["high_conf_link"].any()) else 0,
    )
    plasma_map_path = cfg.results_dir / "plasma_to_animal_map.csv"
    map_df.to_csv(plasma_map_path, index=False)
    logger.info("Saved plasma-to-animal mapping table to: %s", plasma_map_path)

    valid_mask = (
        prim_plasma_meta["animal_id"].notna()
        & prim_plasma_meta["animal_id"].astype(str).str.strip().isin(bulk_valid_ids)
    )
    high_conf_mask = prim_plasma_meta["animal_id_confidence"].astype(str).str.strip().str.lower().isin(high_conf_set)
    mapped_valid_ids = prim_plasma_meta.loc[valid_mask, "animal_id"].astype(str).str.strip()
    mapped_high_conf_valid_ids = (
        prim_plasma_meta.loc[valid_mask & high_conf_mask, "animal_id"].astype(str).str.strip()
    )
    collision_count = int(mapped_high_conf_valid_ids.duplicated().sum())
    n_plasma_total = int(len(prim_plasma_meta))
    n_mapped_valid = int(mapped_valid_ids.nunique())
    n_confirmed_valid = int(mapped_high_conf_valid_ids.nunique())
    linkage_qc = pd.DataFrame(
        [
            {
                "n_plasma_total": n_plasma_total,
                "n_mapped_non_null": int(prim_plasma_meta["animal_id"].notna().sum()),
                "n_mapped_high_conf": int((prim_plasma_meta["animal_id"].notna() & high_conf_mask).sum()),
                "n_mapped_valid_in_bulk": n_mapped_valid,
                "n_unresolved": int(prim_plasma_meta["animal_id"].isna().sum()),
                "mapping_collision_count": collision_count,
                "mapping_coverage": (float(n_mapped_valid) / float(n_plasma_total)) if n_plasma_total > 0 else 0.0,
            }
        ]
    )
    linkage_qc = _ensure_standard_schema(
        linkage_qc,
        available=True,
        estimable=bool(n_confirmed_valid > 0 and collision_count == 0),
        reason=(
            "No high-confidence plasma-to-animal links could be established; alias matches remain candidates."
            if n_confirmed_valid <= 0
            else (
                f"Detected {collision_count} duplicate high-confidence plasma-to-animal "
                "mapping collision(s)."
                if collision_count > 0
                else ""
            )
        ),
        n_used=n_plasma_total,
        method="plasma_linkage_qc",
        evidence_level=2 if n_confirmed_valid > 0 and collision_count == 0 else 0,
    )
    linkage_qc_path = cfg.results_dir / "linkage_qc_report.csv"
    linkage_qc.to_csv(linkage_qc_path, index=False)
    logger.info("Saved linkage QC report to: %s", linkage_qc_path)

    
    n_plasma_samples = int(len(prim_plasma_meta))
    if n_plasma_samples < 50:
        logger.warning(
            "Plasma biomarker ranking uses only %d samples; large |rho| may be unstable.",
            n_plasma_samples,
        )

    biomarker_bootstrap = int(getattr(cfg, "plasma_biomarker_bootstrap", 120))
    biomarker_top_k = int(getattr(cfg, "plasma_biomarker_stability_top_k", 250))
    # Keep runtime bounded on small public plasma cohorts while retaining
    # bootstrap-based stability checks.
    if n_plasma_samples <= 40:
        biomarker_bootstrap = min(biomarker_bootstrap, 120)
        biomarker_top_k = min(biomarker_top_k, 250)
    if n_plasma_samples <= 25:
        biomarker_bootstrap = min(biomarker_bootstrap, 80)
        biomarker_top_k = min(biomarker_top_k, 150)
    logger.info(
        "Plasma biomarker stability settings: n_bootstrap=%d, top_k=%d",
        biomarker_bootstrap,
        biomarker_top_k,
    )

    plasma_biomarkers = compute_plasma_protein_contrasts(
        plasma_expr=prim_plasma_expr,
        plasma_meta=prim_plasma_meta,
        feature_annotations=plasma_feature_annotations,
        contrasts=getattr(cfg, "plasma_treatment_contrasts", None)
        or [
            ("GES_vs_V", "GES", "V"),
            ("WT_vs_V", "WT", "V"),
            ("GES_vs_WT", "GES", "WT"),
        ],
        primary_contrast=getattr(cfg, "plasma_primary_contrast", "GES_vs_V"),
        min_per_group=getattr(cfg, "plasma_contrast_min_per_group", 3),
        n_bootstrap=biomarker_bootstrap,
        random_state=getattr(cfg, "random_state", getattr(cfg, "random_seed", 42)),
        min_sign_agreement=getattr(cfg, "plasma_biomarker_sign_agreement_min", 0.8),
        stability_top_k=biomarker_top_k,
    )
    plasma_biomarkers = _ensure_standard_schema(
        plasma_biomarkers,
        available=not plasma_biomarkers.empty,
        estimable=not plasma_biomarkers.empty,
        reason=(
            "Small plasma groups (n=8 each) and incompletely documented source normalization; "
            "interpret treatment-associated protein contrasts as exploratory."
            if not plasma_biomarkers.empty and n_plasma_samples < 50
            else ("" if not plasma_biomarkers.empty else "No plasma protein contrasts could be estimated.")
        ),
        n_used=n_plasma_samples,
        method="pairwise_ols_hc3_log2_abundance_group_sex",
        evidence_level=1 if not plasma_biomarkers.empty else 0,
    )

    out_biomarkers_csv = cfg.results_dir / "plasma_biomarkers.csv"
    plasma_biomarkers.to_csv(out_biomarkers_csv, index=False)
    logger.info("Saved plasma biomarker table to: %s", out_biomarkers_csv)
    logger.info("Plasma matrix shape: %s", prim_plasma_expr.shape)
    logger.info(
        "Plasma groups: %s",
        prim_plasma_meta.get("group", pd.Series(dtype=str)).value_counts().to_dict(),
    )

    # The legacy all-sample PC1 state is retired. The reference-trained axis
    # below supplies the only plasma state score used by downstream guarded
    # analyses.
    prim_plasma_state = pd.Series(dtype=float, name="plasma_state_score")

    # ---- Oriented plasma age-state axis ----
    plasma_axis_method = "reference_trained_oriented_plasma_pc1_age_axis"
    if getattr(cfg, "enable_plasma_age_axis", True):
        try:
            plasma_group_values = {
                _canonical_group_label(group)
                for group in prim_plasma_meta.get("group", pd.Series(dtype=str)).dropna().astype(str)
            }
            plasma_young_groups = tuple(
                sorted(
                    {
                        _canonical_group_label(group)
                        for group in getattr(cfg, "plasma_axis_young_labels", ["Y", "Y_C"])
                        if _canonical_group_label(group) in plasma_group_values
                    }
                )
            ) or ("Y",)
            plasma_treated_group = _canonical_group_label(cfg.primate_treated_label)
            plasma_old_control_groups = tuple(
                sorted(
                    {
                        _canonical_group_label(group)
                        for group in getattr(cfg, "plasma_axis_old_control_labels", ["O_V", "V"])
                        if _canonical_group_label(group) in plasma_group_values
                        and _canonical_group_label(group) not in set(plasma_young_groups)
                        and _canonical_group_label(group) != plasma_treated_group
                    }
                )
            )
            if not plasma_old_control_groups:
                # Vehicle is the only prespecified old reference for this axis.
                # Do not substitute WTC or another observed group when V is absent.
                plasma_old_control_groups = ("V",)

            plasma_axis_scores, plasma_axis_loadings, plasma_axis_summary = build_oriented_plasma_aging_axis(
                plasma_expr=prim_plasma_expr,
                plasma_meta=prim_plasma_meta,
                feature_annotations=plasma_feature_annotations,
                young_groups=plasma_young_groups,
                old_control_groups=plasma_old_control_groups,
                treated_groups=(plasma_treated_group,),
                n_top_proteins=int(getattr(cfg, "n_top_plasma_features", 50)),
                min_group_samples=int(getattr(cfg, "plasma_axis_min_group_samples", 2)),
                min_non_nan_frac=float(getattr(cfg, "plasma_axis_min_non_nan_frac", 0.8)),
                n_bootstrap=max(500, int(getattr(cfg, "n_bootstrap", 2000)) // 2),
                n_permutations=int(getattr(cfg, "exosome_fraction_permutations", 1000)),
                random_state=int(getattr(cfg, "random_state", getattr(cfg, "random_seed", 42))),
            )
        except Exception as e:
            logger.warning("Oriented plasma age-state axis failed: %s", e)
            plasma_axis_scores = _result_stub(
                reason=f"Oriented plasma age-state axis failed: {e}",
                method=plasma_axis_method,
            )
            plasma_axis_loadings = plasma_axis_scores.copy()
            plasma_axis_summary = plasma_axis_scores.copy()
    else:
        plasma_axis_scores = _result_stub(
            reason="Plasma age-state axis block disabled by config.",
            method=plasma_axis_method,
        )
        plasma_axis_loadings = plasma_axis_scores.copy()
        plasma_axis_summary = plasma_axis_scores.copy()

    plasma_axis_estimable = bool(
        not plasma_axis_summary.empty
        and "estimable" in plasma_axis_summary.columns
        and plasma_axis_summary["estimable"].astype(bool).any()
    )
    plasma_axis_reason = (
        str(plasma_axis_summary["reason"].dropna().astype(str).iloc[0])
        if not plasma_axis_summary.empty
        and "reason" in plasma_axis_summary.columns
        and not plasma_axis_summary["reason"].dropna().empty
        else ""
    )
    plasma_axis_scores = _ensure_standard_schema(
        plasma_axis_scores,
        available=not plasma_axis_scores.empty,
        estimable=plasma_axis_estimable,
        reason=plasma_axis_reason if not plasma_axis_estimable else None,
        method=plasma_axis_method,
        evidence_level=1 if plasma_axis_estimable else 0,
    )
    plasma_axis_loadings = _ensure_standard_schema(
        plasma_axis_loadings,
        available=not plasma_axis_loadings.empty,
        estimable=plasma_axis_estimable,
        reason=plasma_axis_reason if not plasma_axis_estimable else None,
        method=plasma_axis_method,
        evidence_level=1 if plasma_axis_estimable else 0,
    )
    plasma_axis_summary = _ensure_standard_schema(
        plasma_axis_summary,
        method=plasma_axis_method,
        evidence_level=1 if plasma_axis_estimable else 0,
    )

    plasma_axis_scores_path = cfg.results_dir / "plasma_age_axis_scores.csv"
    plasma_axis_loadings_path = cfg.results_dir / "plasma_age_axis_loadings.csv"
    plasma_axis_summary_path = cfg.results_dir / "plasma_age_axis_summary.csv"
    plasma_axis_scores.to_csv(plasma_axis_scores_path, index=False)
    plasma_axis_loadings.to_csv(plasma_axis_loadings_path, index=False)
    plasma_axis_summary.to_csv(plasma_axis_summary_path, index=False)
    logger.info("Saved oriented plasma age-state axis scores to: %s", plasma_axis_scores_path)
    logger.info("Saved oriented plasma age-state loadings to: %s", plasma_axis_loadings_path)
    logger.info("Saved oriented plasma age-state summary to: %s", plasma_axis_summary_path)

    if {"sample_id", "plasma_age_axis_score"}.issubset(plasma_axis_scores.columns):
        merge_cols = ["sample_id", "plasma_age_axis_score"]
        if "raw_pc1_score" in plasma_axis_scores.columns:
            merge_cols.append("raw_pc1_score")
        prim_plasma_meta = prim_plasma_meta.drop(
            columns=[
                col
                for col in [*merge_cols, "plasma_state_score"]
                if col != "sample_id" and col in prim_plasma_meta.columns
            ],
            errors="ignore",
        ).merge(
            plasma_axis_scores[merge_cols],
            on="sample_id",
            how="left",
        )
        prim_plasma_meta["plasma_state_score"] = prim_plasma_meta["plasma_age_axis_score"]
        prim_plasma_state = prim_plasma_meta.set_index("sample_id")["plasma_state_score"].copy()

    plasma_axis_delta_age = correlate_plasma_axis_with_delta_age(
        prim_meta=prim_meta,
        plasma_meta=prim_plasma_meta,
        high_conf_values=high_conf_values,
        min_animals=int(getattr(cfg, "plasma_axis_min_linked_animals", 8)),
        n_bootstrap=max(500, int(getattr(cfg, "n_bootstrap", 2000)) // 2),
        n_permutations=int(getattr(cfg, "exosome_fraction_permutations", 1000)),
        random_state=int(getattr(cfg, "random_state", getattr(cfg, "random_seed", 42))),
    )
    plasma_axis_delta_age = _ensure_standard_schema(
        plasma_axis_delta_age,
        method="linked_plasma_age_axis_delta_age_spearman",
    )
    plasma_axis_delta_age_path = cfg.results_dir / "plasma_age_axis_delta_age_correlation.csv"
    plasma_axis_delta_age.to_csv(plasma_axis_delta_age_path, index=False)
    logger.info("Saved plasma axis to delta-age correlation to: %s", plasma_axis_delta_age_path)

    # ---- Linkage audit + estimability gate (bulk <-> plasma) ----
    linkage_audit = audit_primate_plasma_linkage(
        prim_meta=prim_meta,
        prim_plasma_meta=prim_plasma_meta,
        animal_col="animal_id",
        group_col="group",
        sex_col="sex",
        animal_confidence_col="animal_id_confidence",
        high_conf_values=high_conf_values,
        treated_label=cfg.primate_treated_label,
        control_labels=tuple(getattr(cfg, "linked_plasma_control_labels", ("O_V", "O_WT"))),
    )
    linkage_audit = _ensure_standard_schema(
        pd.DataFrame([linkage_audit]),
        available=True,
        estimable=True,
        reason="",
        n_used=int(len(prim_meta)),
        method="animal_id_linkage_audit",
        evidence_level=2 if int(linkage_audit.get("n_overlap_animal_ids_high_conf", 0) or 0) > 0 else 0,
    )
    linkage_audit_path = cfg.results_dir / "linkage_audit.csv"
    linkage_audit.to_csv(linkage_audit_path, index=False)
    logger.info("Saved linkage audit summary to: %s", linkage_audit_path)

    estimability = build_estimability_report(
        linkage_audit.iloc[0].to_dict(),
        min_samples_for_mediation=cfg.min_samples_for_mediation,
        min_overlap_animals=int(getattr(cfg, "linkage_min_overlap_animals", 20)),
        min_treated_overlap=int(getattr(cfg, "linkage_min_treated_overlap", 6)),
        min_control_overlap=int(getattr(cfg, "linkage_min_control_overlap", 12)),
    )
    estimability = _ensure_standard_schema(
        pd.DataFrame([estimability]),
        available=True,
        estimable=bool(estimability.get("can_do_mediation", False)),
        reason="",
        n_used=int(estimability.get("n_overlap_animal_ids", 0)),
        method="estimability_gate",
        evidence_level=2 if estimability.get("tier") != "unlinked" else 0,
    )
    estimability_path = cfg.results_dir / "estimability_report.csv"
    estimability.to_csv(estimability_path, index=False)
    estimability_row = estimability.iloc[0].to_dict()
    logger.info(
        "Estimability gate: tier=%s, can_do_mediation=%s, reason=%s",
        estimability_row.get("tier"),
        estimability_row.get("can_do_mediation"),
        estimability_row.get("reason"),
    )
    logger.info("Saved estimability report to: %s", estimability_path)

    # ---- Optional mediation: X = treatment, M = plasma_state_score, Y = rejuvenation_score ----
    med = None
    med_df = None
    mediation_csv = cfg.results_dir / "mediation_summary.csv"
    tier = str(estimability_row.get("tier", "unlinked"))
    mediation_reason = _causal_gate_reason(
        enable_mediation=cfg.enable_mediation,
        enable_causal_decomposition=cfg.enable_causal_decomposition,
        estimability_row=estimability_row,
    )

    if mediation_reason is not None:
        logger.warning("Mediation skipped by strict gate: %s", mediation_reason)
        med_df = _build_mediation_stub(
            reason=mediation_reason,
            tier=tier,
            n_overlap_animal_ids=int(estimability_row.get("n_overlap_animal_ids", 0)),
        )
    else:
        has_animal_id_prim = "animal_id" in prim_meta.columns
        has_animal_id_plasma = "animal_id" in prim_plasma_meta.columns

        if not (has_animal_id_prim and has_animal_id_plasma):
            med_df = _build_mediation_stub(
                reason="animal_id column missing in prim_meta and/or prim_plasma_meta.",
                tier="unlinked",
                n_overlap_animal_ids=0,
            )
        else:
            if "group_binary" not in prim_meta.columns:
                prim_meta = prim_meta.copy()
                prim_meta["group_binary"] = (prim_meta["group"] == cfg.primate_treated_label).astype(int)

            prim_merge = _build_animal_level_mediation_table(
                prim_meta,
                prim_plasma_meta,
                high_conf_values=high_conf_values,
            )
            n_overlap_animals = int(prim_merge["animal_id"].nunique()) if not prim_merge.empty else 0
            logger.info(
                "Mediation: merged table has %d animal-level rows from %d unique animals.",
                len(prim_merge),
                n_overlap_animals,
            )

            if n_overlap_animals < cfg.min_samples_for_mediation:
                med_df = _build_mediation_stub(
                    reason=f"Too few overlapping animals for mediation (n={n_overlap_animals}).",
                    tier=tier,
                    n_overlap_animal_ids=n_overlap_animals,
                    n_used=int(len(prim_merge)),
                )
            else:
                med = call_with_supported_kwargs(
                    simple_mediation_bootstrap,
                    prim_merge,
                    x_col="group_binary",
                    treatment="group_binary",
                    m_col="plasma_state_score",
                    mediator="plasma_state_score",
                    y_col="rejuvenation_score",
                    outcome="rejuvenation_score",
                    n_boot=cfg.mediation_bootstrap,
                    seed=cfg.random_seed,
                    random_state=cfg.random_seed,
                )

                if isinstance(med, dict):
                    med_df = pd.DataFrame([med])
                elif isinstance(med, pd.Series):
                    med_df = med.to_frame().T
                else:
                    med_df = med.copy()

                med_df["tier"] = tier
                med_df["n_overlap_animal_ids"] = n_overlap_animals
                med_df["available"] = True
                med_df["estimable"] = True
                med_df["reason"] = ""
                med_df["n_used"] = int(len(prim_merge))
                med_df["method"] = "linear_mediation_bootstrap"
                med_df = _attach_primary_mediation_interval(med_df)
                med_df["evidence_level"] = _evidence_level(
                    estimable=True, tier=tier, has_linked_mediation=True
                )

                try:
                    plot_mediation_effects_bar(
                        med_df,
                        out_path=cfg.figures_dir / "mediation_effects_bar.png",
                    )
                except Exception as e:
                    logger.warning("plot_mediation_effects_bar failed: %s", e)

    med_df = _ensure_standard_schema(
        med_df,
        method="linear_mediation_bootstrap",
        evidence_level=int(med_df["evidence_level"].iloc[0]) if not med_df.empty else 0,
    )
    med_df.to_csv(mediation_csv, index=False)
    logger.info("Saved mediation summary to: %s", mediation_csv)

    # ---- Cross-pattern comparison and exosome-alignment summaries ----
    pattern_comparison: Dict[str, Any] = {
        "n_common_tissues": 0,
        "spearman_rho": np.nan,
        "pearson_r": np.nan,
    }
    exo_fraction: Dict[str, Any] = {}
    exosome_alignment_by_tissue = _result_stub(
        reason="Mouse exosome alignment not available.",
        method="cross_species_effect_alignment",
        extra={"contrast": "NA", "mouse_tissue": "NA", "primate_tissue": "NA"},
    )
    exosome_alignment_summary = _result_stub(
        reason="Mouse exosome alignment not available.",
        method="cross_species_effect_alignment_summary",
        extra={"contrast": "NA", "n_common_tissues": 0},
    )

    try:
        tissue_mapping_path = getattr(cfg, "mouse_tissue_mapping_path", None)
        if tissue_mapping_path is None:
            raise ValueError("No cross-species tissue mapping contract was configured.")
        tissue_mapping_contract = load_cross_species_tissue_mapping(tissue_mapping_path)
        tissue_mapping_contract.to_csv(
            cfg.results_dir / "cross_species_tissue_mapping.csv", index=False
        )
        exosome_alignment_by_tissue, exosome_alignment_summary = compute_exosome_alignment_tables(
            prim_tissue_effects,
            mouse_exosome_effects,
            tissue_map=getattr(cfg, "mouse_to_primate_tissue_map", None),
            tissue_mapping=tissue_mapping_contract,
            contrasts=getattr(cfg, "mouse_alignment_contrasts", ["GES_vs_Veh", "WT_vs_Veh"]),
            min_common_tissues=int(getattr(cfg, "exosome_min_common_tissues", 3)),
            n_bootstrap=int(getattr(cfg, "n_bootstrap", 2000)),
            n_permutations=int(getattr(cfg, "exosome_fraction_permutations", 1000)),
            random_state=int(getattr(cfg, "random_state", getattr(cfg, "random_seed", 42))),
        )
        exosome_alignment_by_tissue = _ensure_standard_schema(
            exosome_alignment_by_tissue,
            method="cross_species_effect_alignment",
        )
        exosome_alignment_summary = _ensure_standard_schema(
            exosome_alignment_summary,
            method="cross_species_effect_alignment_summary",
        )
    except Exception as e:
        logger.warning("Exosome alignment failed: %s", e)
        exosome_alignment_by_tissue = _result_stub(
            reason=f"Exosome alignment failed: {e}",
            method="cross_species_effect_alignment",
            extra={"contrast": "NA", "mouse_tissue": "NA", "primate_tissue": "NA"},
        )
        exosome_alignment_summary = _result_stub(
            reason=f"Exosome alignment failed: {e}",
            method="cross_species_effect_alignment_summary",
            extra={"contrast": "NA", "n_common_tissues": 0},
        )

    exosome_alignment_by_tissue_path = cfg.results_dir / "exosome_alignment_by_tissue.csv"
    exosome_alignment_by_tissue.to_csv(exosome_alignment_by_tissue_path, index=False)
    logger.info("Saved exosome alignment-by-tissue summary to: %s", exosome_alignment_by_tissue_path)

    exosome_alignment_summary_path = cfg.results_dir / "exosome_alignment_summary.csv"
    exosome_alignment_summary.to_csv(exosome_alignment_summary_path, index=False)
    logger.info("Saved exosome alignment summary to: %s", exosome_alignment_summary_path)

    response_alignment_by_tissue = _result_stub(
        reason="Cross-species response-alignment inputs are unavailable.",
        method="cross_species_hedges_g_response_pair",
        extra={
            "mouse_tissue": "NA",
            "primate_tissue": "NA",
            "macaque_standardized_effect": np.nan,
            "mouse_standardized_effect": np.nan,
        },
    )
    response_alignment_summary = _result_stub(
        reason="Cross-species response-alignment inputs are unavailable.",
        method="cross_species_hedges_g_response_alignment",
        extra={
            "contrast_pair": "NA",
            "n_common_tissues": 0,
            "cosine_similarity": np.nan,
            "relative_response_norm": np.nan,
            "aligned_response_coefficient": np.nan,
        },
    )
    try:
        response_alignment_by_tissue, response_alignment_summary = (
            compute_cross_species_response_alignment(
                prim_meta,
                mouse_sample_outcomes,
                getattr(cfg, "mouse_tissue_mapping_path"),
                primate_treated=cfg.primate_treated_label,
                primate_control=cfg.primate_vehicle_label,
                mouse_treated=getattr(cfg, "mouse_treated_label", "GES"),
                mouse_control="Veh",
                min_per_group=max(2, int(cfg.min_samples_per_group_for_rejuv)),
                min_common_tissues=int(getattr(cfg, "exosome_min_common_tissues", 3)),
                n_bootstrap=int(getattr(cfg, "response_alignment_bootstrap", 1000)),
                n_permutations=int(getattr(cfg, "response_alignment_permutations", 1000)),
                random_state=int(getattr(cfg, "random_state", getattr(cfg, "random_seed", 42))),
            )
        )
        response_alignment_by_tissue = _ensure_standard_schema(
            response_alignment_by_tissue,
            method="cross_species_hedges_g_response_pair",
        )
        response_alignment_summary = _ensure_standard_schema(
            response_alignment_summary,
            method="cross_species_hedges_g_response_alignment",
        )
    except Exception as e:
        logger.warning("Cross-species response alignment failed: %s", e)
        response_alignment_by_tissue = _result_stub(
            reason=f"Cross-species response alignment failed: {e}",
            method="cross_species_hedges_g_response_pair",
            extra={
                "mouse_tissue": "NA",
                "primate_tissue": "NA",
                "macaque_standardized_effect": np.nan,
                "mouse_standardized_effect": np.nan,
            },
        )
        response_alignment_summary = _result_stub(
            reason=f"Cross-species response alignment failed: {e}",
            method="cross_species_hedges_g_response_alignment",
            extra={
                "contrast_pair": "NA",
                "n_common_tissues": 0,
                "cosine_similarity": np.nan,
                "relative_response_norm": np.nan,
                "aligned_response_coefficient": np.nan,
            },
        )

    response_alignment_by_tissue_path = (
        cfg.results_dir / "cross_species_response_alignment_by_tissue.csv"
    )
    response_alignment_by_tissue.to_csv(response_alignment_by_tissue_path, index=False)
    logger.info(
        "Saved cross-species response alignment by tissue to: %s",
        response_alignment_by_tissue_path,
    )

    response_alignment_summary_path = (
        cfg.results_dir / "cross_species_response_alignment_summary.csv"
    )
    response_alignment_summary.to_csv(response_alignment_summary_path, index=False)
    logger.info(
        "Saved cross-species response alignment summary to: %s",
        response_alignment_summary_path,
    )

    try:
        pattern_comparison = call_with_supported_kwargs(
            compare_effect_patterns,
            prim_tissue_effects,
            mouse_tissue_effects,
            tissue_weighting=getattr(cfg, "tissue_weighting", None),
            weighting=getattr(cfg, "tissue_weighting", None),
        )
    except Exception as e:
        logger.warning("Pattern comparison failed: %s", str(e))

    exo_fraction = _causal_exosome_fraction_status()

    # ------------------------- Translational insights -------------------------
    prim_outcomes = prim_meta.copy()
    for col in ("sample_id", "group", "tissue", "rejuvenation_score", "predicted_age_cv", "predicted_age"):
        if col not in prim_outcomes.columns:
            prim_outcomes[col] = pd.NA

    prim_ctrl = getattr(cfg, "primate_control_labels", None) or [cfg.control_label]
    mouse_tr = getattr(cfg, "mouse_treated_label", "GES")
    mouse_ctrl = getattr(cfg, "mouse_control_labels", None) or ["Veh", "WT", "Ctrl", "Baseline"]
    insights = None
    if getattr(cfg, "enable_translation_module", False):
        try:
            sig = inspect.signature(generate_translational_insights)
            params = sig.parameters

            def pick(*names):
                for n in names:
                    if n in params:
                        return n
                return None

            kw: Dict[str, Any] = {}

            # --- Map core concepts to whichever names your module actually uses ---
            n_cells = pick("effect_cells", "prim_tissue_effects", "tissue_effects", "bulk_effects", "effect_a")
            if n_cells:
                kw[n_cells] = prim_tissue_effects

            n_exo = pick("effect_exosomes", "prim_plasma_state", "plasma_state", "plasma", "effect_b")
            if n_exo:
                kw[n_exo] = prim_plasma_state

            n_comp = pick("pattern_comparison", "comparison", "pattern")
            if n_comp:
                kw[n_comp] = pattern_comparison

            n_frac = pick("exosome_fraction", "exo_fraction")
            if n_frac:
                kw[n_frac] = exo_fraction

            n_med = pick("mediation", "med")
            if n_med:
                kw[n_med] = med

            n_out = pick("prim_outcomes", "prim_meta", "outcomes_prim")
            if n_out:
                kw[n_out] = prim_outcomes

            n_outcol = pick("outcome_col_prim")
            if n_outcol:
                kw[n_outcol] = "rejuvenation_score"

            n_pt = pick("prim_treated_label")
            if n_pt:
                kw[n_pt] = cfg.primate_treated_label

            n_pc = pick("prim_control_labels")
            if n_pc:
                kw[n_pc] = prim_ctrl

            n_mt = pick("mouse_treated_label")
            if n_mt:
                kw[n_mt] = mouse_tr

            n_mc = pick("mouse_control_labels")
            if n_mc:
                kw[n_mc] = mouse_ctrl

            if "prim_expr_log" in params:
                kw["prim_expr_log"] = prim_expr

            # --- Build REQUIRED positional args in the exact signature order ---
            if "prim_meta" in params:
                kw["prim_meta"] = prim_meta

            required_names = [
                name
                for name, p in params.items()
                if p.default is inspect._empty
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]

            args: List[Any] = []
            for name in required_names:
                if name not in kw:
                    raise TypeError(f"Missing required arg for translation module: {name}")
                args.append(kw.pop(name))  # remove so we cannot pass twice

            # Call with guaranteed no-duplicates
            insights = generate_translational_insights(*args, **kw)

        except Exception as e:
            logger.warning("Translational insights skipped due to compatibility issue: %s", str(e))
            insights = None
    else:
        logger.info("Translational module disabled by config (enable_translation_module=False).")
    # Optional save (keep simple and always safe)
    try:
        if insights is not None:
            cfg.results_dir.mkdir(parents=True, exist_ok=True)
            (cfg.results_dir / "translational_insights.txt").write_text(
                insights if isinstance(insights, str) else str(insights),
                encoding="utf-8",
            )
    except Exception as e:
        logger.warning("Could not save translational insights: %s", str(e))

    # ---- Tissue concordance plot (primate vs mouse) ----
    try:
        if "mouse_tissue_effects" in locals() and mouse_tissue_effects is not None:  # type: ignore[name-defined]
            plot_tissue_concordance(
                prim_tissue_effects,
                mouse_tissue_effects,  # type: ignore[name-defined]
                cfg.figures_dir / "tissue_concordance.png",
            )
        else:
            logger.warning("Skipping tissue concordance: mouse_tissue_effects not available.")
    except Exception as e:
        logger.warning("plot_tissue_concordance skipped: %s", str(e))

    # ---- Plasma biomarker ranking ----
    try:
        plot_df = plasma_biomarkers
        if "stable_association" in plasma_biomarkers.columns:
            stable = plasma_biomarkers[
                pd.to_numeric(plasma_biomarkers["stable_association"], errors="coerce").fillna(0).astype(int) == 1
            ]
            if not stable.empty:
                plot_df = stable
            else:
                logger.warning(
                    "No bootstrap-stable plasma associations; plotting full ranking with caution."
                )
        plot_plasma_biomarker_ranking(
            plasma_biomarkers=plot_df,
            outpath=cfg.figures_dir / "plasma_biomarker_ranking.png",
            top_n=cfg.top_plasma_biomarkers,
            requires_spearman=getattr(cfg, "plasma_ranking_requires_spearman", False),
            fallback_metric=getattr(cfg, "plasma_ranking_fallback_metric", "variance"),
        )
    except Exception as e:
        logger.warning("plot_plasma_biomarker_ranking skipped: %s", str(e))

    # ---- Final logging & causal non-identifiability summary ----
    logger.info(
        "Causal exosome-attributable fraction is not identified; "
        "cross-species response alignment is reported separately."
    )

    # Normalise exo_fraction and ALWAYS write a summary CSV
    if exo_fraction is None:
        exo_fraction = {}
    elif not isinstance(exo_fraction, dict):
        exo_fraction = {"raw_result": exo_fraction}

    exo_summary = {
        "n_common_tissues": exo_fraction.get("n_common_tissues", 0),
        "cells_median_abs": exo_fraction.get("cells_median_abs", np.nan),
        "exo_median_abs": exo_fraction.get("exo_median_abs", np.nan),
        "ratio": exo_fraction.get("ratio", np.nan),
        "empirical_p_value": exo_fraction.get("empirical_p_value", np.nan),
        "tier": estimability_row.get("tier", "unlinked"),
        "available": bool(exo_fraction.get("available", False)),
        "estimable": bool(exo_fraction.get("estimable", False)),
        "reason": str(exo_fraction.get("reason", "")),
        "n_used": int(exo_fraction.get("n_used", 0) or 0),
        "method": str(exo_fraction.get("method", "causal_exosome_fraction_not_identified")),
        "ci_low": exo_fraction.get("ci_low", np.nan),
        "ci_high": exo_fraction.get("ci_high", np.nan),
    }
    exo_summary["evidence_level"] = 0
    exo_df = pd.DataFrame([exo_summary])
    exo_df = _ensure_standard_schema(
        exo_df,
        method=exo_summary["method"],
        evidence_level=int(exo_summary["evidence_level"]),
    )
    exo_path = cfg.results_dir / "exosome_fraction_summary.csv"
    exo_df.to_csv(exo_path, index=False)
    logger.info("Saved exosome fraction summary to: %s", exo_path)

    # ---- Sensitivity analyses ----
    sensitivity_df = run_sensitivity_analyses(
        prim_meta=prim_meta,
        prim_plasma_expr=prim_plasma_expr,
        prim_plasma_meta=prim_plasma_meta,
        cfg=cfg,
    )
    sensitivity_df = _ensure_standard_schema(
        sensitivity_df,
        method="sensitivity_analysis",
    )
    sensitivity_path = cfg.results_dir / "sensitivity_summary.csv"
    sensitivity_df.to_csv(sensitivity_path, index=False)
    logger.info("Saved sensitivity summary to: %s", sensitivity_path)

    # Keep the existing translational insights log
    if insights:
        logger.info("Translational summary written to results directory.")

# -----------------------------------------------------------------------------
# Minimal CLI + defaults
# -----------------------------------------------------------------------------
def _build_default_config(
    base_dir: Path,
    *,
    profile: str = "auto",
    data_root: Optional[Path] = None,
) -> PipelineConfig:
    """
    Centralized defaults.
    Supports both full-data and demo-data profiles without editing source paths.
    """
    explicit_data_root = Path(data_root).resolve() if data_root is not None else None
    resolved_profile = _resolve_profile_name(base_dir, profile, explicit_data_root=explicit_data_root)
    resolved_data_root = _resolve_profile_data_root(
        base_dir,
        resolved_profile,
        explicit_data_root=explicit_data_root,
    )
    spec = PROFILE_SPECS[resolved_profile]

    primate_bulk = _make_omix_paths(
        resolved_data_root,
        matrix_name=spec["bulk_matrix"],
        metadata_name=spec["bulk_metadata"],
        required=True,
    )
    primate_plasma = _make_omix_paths(
        resolved_data_root,
        matrix_name=spec["plasma_matrix"],
        metadata_name=spec["plasma_metadata"],
        required=True,
    )
    primate_methylation = _make_omix_paths(
        resolved_data_root,
        matrix_name=spec["methylation_matrix"],
        metadata_name=spec["methylation_metadata"],
        required=False,
    )
    mouse_exosome_bulk = _make_omix_paths(
        resolved_data_root,
        matrix_name=spec["mouse_matrix"],
        metadata_name=spec["mouse_metadata"],
        required=False,
    )

    cfg = PipelineConfig(
        data_profile=resolved_profile,
        data_root=resolved_data_root,
        primate_bulk=primate_bulk,
        primate_bulk_input_mode=spec["bulk_input_mode"],
        primate_bulk_representation=spec["bulk_representation"],
        clock_protocol=spec["clock_protocol"],
        clock_excluded_sample_ids=spec["clock_excluded_sample_ids"],
        clock_fold_assignments=(
            base_dir / spec["clock_fold_file"]
            if spec["clock_fold_file"] is not None
            else None
        ),
        clock_fold_assignments_sha256=spec["clock_fold_sha256"],
        primate_plasma=primate_plasma,
        primate_methylation=primate_methylation,
        mouse_exosome_bulk=mouse_exosome_bulk,
        mouse_tissue_mapping_path=base_dir / "config" / "cross_species_tissue_map.csv",
        results_dir=base_dir / "results",
        figures_dir=base_dir / "figures",
    )
    if primate_methylation is None:
        cfg.enable_methylation_block = False
    if mouse_exosome_bulk is None:
        cfg.enable_mouse_exosome_block = False
    subset_ok = bool(spec["subset_files"]) and all(
        (resolved_data_root / rel).exists() for rel in spec["subset_files"]
    )
    if not subset_ok:
        cfg.enable_subset_validation_block = False
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["auto", "full", "demo"],
        default="auto",
        help="Choose the bundled data profile. 'auto' prefers full data when available, else demo data.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the profile data directory directly, without editing source code paths.",
    )
    parser.add_argument("--safe", action="store_true", help="Extra conservative mode for laptops.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    cfg = _build_default_config(base_dir, profile=args.profile, data_root=args.data_root)
    logger.info("Resolved data profile: %s", cfg.data_profile)
    logger.info("Resolved data root: %s", cfg.data_root)

    # Conservative overrides for your hardware
    if args.safe:
        cfg.n_top_features_expr = 3000
        cfg.top_genes_per_tissue = min(cfg.top_genes_per_tissue, 75)
        cfg.enable_mediation = False
        cfg.mouse_tissue_bootstrap = min(cfg.mouse_tissue_bootstrap, 250)
        cfg.mouse_tissue_permutations = min(cfg.mouse_tissue_permutations, 250)
        cfg.exosome_fraction_bootstrap = min(cfg.exosome_fraction_bootstrap, 500)
        cfg.exosome_fraction_permutations = min(cfg.exosome_fraction_permutations, 250)
        cfg.response_alignment_bootstrap = min(cfg.response_alignment_bootstrap, 250)
        cfg.response_alignment_permutations = min(cfg.response_alignment_permutations, 250)
        cfg.n_bootstrap = min(cfg.n_bootstrap, 500)
        cfg.treatment_n_permutations = min(cfg.treatment_n_permutations, 500)
        cfg.plasma_biomarker_bootstrap = min(cfg.plasma_biomarker_bootstrap, 80)
        cfg.plasma_biomarker_stability_top_k = min(cfg.plasma_biomarker_stability_top_k, 150)

    run_manifest_path = begin_run_manifest(
        cfg,
        repo_root=base_dir,
        argv=[sys.executable, "-m", "src.run_pipeline", *sys.argv[1:]],
    )
    logger.info("Run provenance manifest started at: %s", run_manifest_path)
    try:
        run(cfg)
    except BaseException as exc:
        finalize_run_manifest(
            run_manifest_path,
            cfg,
            status="failed",
            error=exc,
        )
        raise
    else:
        finalize_run_manifest(run_manifest_path, cfg, status="completed")
        logger.info("Run provenance manifest completed at: %s", run_manifest_path)


if __name__ == "__main__":
    main()

