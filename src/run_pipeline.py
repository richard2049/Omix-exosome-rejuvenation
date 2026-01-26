from __future__ import annotations

"""
run_pipeline.py

End-to-end exploratory pipeline for the SRSC project:
- loads primate bulk RNA-seq (OMIX007580),
- builds and *cross-validates* a lightweight transcriptomic clock,
- derives a proxy rejuvenation score,
- estimates tissue-level exosome-like effects,
- processes plasma proteomics (OMIX007581) to build a plasma state score,
- optionally loads Mammal40 methylation (OMIX007582) in an exploratory fashion.

Status: exploratory / work in progress (v0.1 -> v0.2-dev).
"""

import argparse
import sys
import inspect
import re
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, List

import pandas as pd
from sklearn.model_selection import KFold

from .config import PipelineConfig, OmixPaths
from .logging_utils import get_logger

from .omix_io import (
    load_omix_matrix,
    load_omix_metadata,
    align_matrix_and_metadata,
    find_first_present_column,
    assert_metadata_reasonable,
)

from .preprocessing import (
    standardize_metadata_columns,
    log1p_counts,
    filter_top_variance,
    build_proxy_rejuvenation_score,
)

from .clocks import (
    train_transcriptomic_clock,
    predict_biological_age,
    summarize_clock_performance,
)

from .exosome_effect import (
    compute_group_effect_by_tissue,
    compare_effect_patterns,
    estimate_exosome_fraction,
    build_plasma_state_score,
    simple_mediation_bootstrap,
)

from .translation_module import generate_translational_insights
from .viz import (
    plot_tissue_concordance,
    plot_plasma_biomarker_ranking,
    plot_age_scatter,
    plot_rejuvenation_by_group,
)

from .rejuvenation import (
    compute_delta_age,
    summarize_rejuvenation_by_tissue,
    summarise_global_rejuvenation,
    summarize_tissue_expression_effects
)

logger = get_logger(__name__)


def call_with_supported_kwargs(func, /, *args, **kwargs):
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return func(*args, **filtered)


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


def load_plasma_proteomics_csv(path: Path) -> pd.DataFrame:
    """
    Loads OMIX007581-01-like plasma proteomics wide CSV.
    Returns matrix with rows=features, cols=samples.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Use Gene name if available, else Protein accession, else first col
    if "Gene name" in df.columns:
        idx = df["Gene name"].astype(str)
    elif "Protein accession" in df.columns:
        idx = df["Protein accession"].astype(str)
    else:
        idx = df.iloc[:, 0].astype(str)

    drop_set = {"Protein accession", "Gene name"}
    sample_cols = [c for c in df.columns if c not in drop_set]

    # Keep only sample-like columns (e.g., FY_1, MWT_2, FGES_4)
    sample_cols = [c for c in sample_cols if re.match(r"^[A-Za-z]+_\d+$", c)]

    mat = df.loc[:, sample_cols].copy()
    mat.index = idx

    # Numeric conversion (float32-ish)
    for c in mat.columns:
        mat[c] = pd.to_numeric(mat[c], errors="coerce", downcast="float")

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

        rows.append(
            {
                "sample_id": s,
                "sex": "F" if sex == "F" else ("M" if sex == "M" else pd.NA),
                "group": group_code,  # Y, V, WT, GES, etc.
                "replicate": rep,
                "omics": "plasma_proteomics",
            }
        )

    return pd.DataFrame(rows)


def clean_plasma_matrix(
    mat: pd.DataFrame,
    min_feature_non_nan_frac: float = 0.5,
    min_sample_non_nan_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Clean + impute plasma proteomics matrix for PCA-based scoring.
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

    # Median imputation per feature (robust for proteomics)
    med = mat.median(axis=1)
    mat = mat.T.fillna(med).T

    # Final safety
    if mat.isna().any().any():
        # Edge case: a feature with all NaN after filtering; remove it
        mat = mat.dropna(axis=0, how="any")

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

    matrix = load_omix_matrix(
        omix.matrix,
        allowed_samples=allowed,
        dtype="float32",
    )

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

    expr_log = log1p_counts(matrix, assume_counts=assume_counts)
    return expr_log, meta


def load_methylation_block(cfg: PipelineConfig) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Optional loading + light QC of Mammal40 methylation (OMIX007582).

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
    - Intended for exploratory analyses in v0.1; not yet tightly integrated into
      the exosome fraction estimation.
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


# -----------------------------------------------------------------------------
# Cross-validated clock evaluation
# -----------------------------------------------------------------------------
# def cross_validate_clock(
#     expr_log: pd.DataFrame,
#     meta: pd.DataFrame,
#     cfg: PipelineConfig,
#     age_col: str = "age",
# ) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """
    Compute out-of-fold (cross-validated) age predictions and summarize performance.

    Returns
    -------
    meta_out : DataFrame
        Metadata with an extra 'predicted_age_cv' column (may contain NaNs if
        something failed for some samples).
    metrics : dict or None
        Summary metrics from summarize_clock_performance, or None if CV failed.

    Notes
    -----
    - Uses the same train_transcriptomic_clock / predict_biological_age functions
      that are used for the final model, but inside K-fold splits.
    - This gives a scientifically more honest estimate of generalization than
      in-sample R^2 / MAE on the training data.
    """
    meta = meta.copy()

    if age_col not in meta.columns:
        raise ValueError(f"Age column '{age_col}' not found in metadata.")

    if "sample_id" not in meta.columns:
        raise ValueError("Metadata must contain 'sample_id' before CV.")

    n_samples = meta.shape[0]
    if n_samples < 4:
        logger.warning("Too few samples (%d) for cross-validation; skipping.", n_samples)
        return meta, None

    # Decide number of folds
    default_folds = getattr(cfg, "clock_cv_folds", 5) or 5
    n_splits = min(default_folds, n_samples)
    if n_splits < 2:
        logger.warning("Cannot run CV with <2 folds; skipping.")
        return meta, None

    random_state = getattr(cfg, "random_state", getattr(cfg, "random_seed", 42))

    logger.info("Running %d-fold cross-validation for the transcriptomic clock.", n_splits)

    sample_ids = meta["sample_id"].tolist()
    cv_pred = pd.Series(index=sample_ids, dtype="float64")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(kf.split(sample_ids), start=1):
        train_ids = [sample_ids[i] for i in train_idx]
        test_ids = [sample_ids[i] for i in test_idx]

        expr_train = expr_log.loc[:, train_ids]
        expr_test = expr_log.loc[:, test_ids]

        meta_train = meta.iloc[train_idx].reset_index(drop=True)
        meta_test = meta.iloc[test_idx].reset_index(drop=True)

        clock_result = train_transcriptomic_clock(  # <-- CHANGED
            expr_train,
            meta_train,
            age_col=age_col,
            model=getattr(cfg, "clock_model", None),
            n_top_features=getattr(cfg, "n_top_features_clock", 3000),
        )

        if isinstance(clock_result, tuple):        # <-- CHANGED
            clock_fold, clock_aux = clock_result
        else:
            clock_fold, clock_aux = clock_result, None

        pred_fold = predict_biological_age(
            clock_fold,
            expr_test,
            meta_test,
        )
        pred_df = _normalize_pred_output(pred_fold)

        pred_df = pred_df.set_index("sample_id")
        for sid, val in pred_df["predicted_age"].items():
            if sid in cv_pred.index:
                cv_pred.loc[sid] = float(val)

        logger.info(
            "CV fold %d/%d: got %d predictions.",
            fold,
            n_splits,
            pred_df.shape[0],
        )

    meta["predicted_age_cv"] = meta["sample_id"].map(cv_pred)

    try:
        metrics = summarize_clock_performance(
            meta,
            chrono_col=age_col,
            pred_col="predicted_age_cv",
        )
    except Exception as e:
        logger.warning("summarize_clock_performance failed on CV predictions: %s", e)
        metrics = None

    return meta, metrics


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

    # ---- Optional: primate methylation (Mammal40) ----
    prim_meth_expr, prim_meth_meta = load_methylation_block(cfg)
    if prim_meth_expr is None:
        logger.info("Methylation data not available or not usable; skipping methylation-derived analyses for v0.1.")
    else:
        logger.info("Methylation block loaded (CpGs x samples): %s", prim_meth_expr.shape)

    tissue_expr_effects = summarize_tissue_expression_effects(
        prim_expr,
        prim_meta,
        tissue_col=cfg.tissue_col_candidates[0],
        group_col=cfg.group_col_candidates[0],
        control_labels=cfg.primate_control_labels,
        treated_labels=[cfg.primate_treated_label],
        min_per_group=max(3, cfg.min_samples_for_mediation // 2),
)

    tissue_expr_path = cfg.results_dir / "tissue_expression_effects.csv"
    tissue_expr_effects.to_csv(tissue_expr_path, index=False)
    logger.info("Saved tissue expression effects to: %s", tissue_expr_path)

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

    # 1) Train a final transcriptomic clock on all samples
    prim_clock, prim_cv_pred_df, clock_metrics = train_transcriptomic_clock(
            prim_expr,
            prim_meta,
            age_col="age",
            model=getattr(cfg, "clock_model", None),
            n_top_features=getattr(cfg, "n_top_features_clock", 3000),
            n_splits=getattr(cfg, "clock_cv_folds", 5),
            random_state=getattr(cfg, "random_state", getattr(cfg, "random_seed", 42)),
    )

    logger.info("Transcriptomic clock metrics (train CV): %s", clock_metrics)

    # (optional) log / save metrics
    clock_metrics_path = cfg.results_dir / "clock_metrics_primates.csv"
    pd.DataFrame([clock_metrics]).to_csv(clock_metrics_path, index=False)

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

    # --- 3) Rejuvenation score using CV predictions ---
    prim_meta = build_proxy_rejuvenation_score(
        prim_meta,
        pred_age_col="predicted_age_cv",
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

    # Use cross-validated predictions for rejuvenation score if available,
    # otherwise fall back to in-sample predictions.
    pred_col_for_rejuvenation = "predicted_age_cv" if "predicted_age_cv" in prim_meta.columns else "predicted_age"

    prim_meta = build_proxy_rejuvenation_score(
        prim_meta,
        pred_age_col=pred_col_for_rejuvenation,
        chrono_age_col="age",
        out_col="rejuvenation_score",
    )

    # Plot rejuvenation score distribution by group (exploratory)
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
    pred_age_col="predicted_age",
    chrono_age_col="age",
    out_col="delta_age",
    )

    # Global summary
    global_rejuv = summarise_global_rejuvenation(
        prim_meta,
        group_col=cfg.group_col_candidates[0],  # p.ej. 'group'
        value_col="delta_age",
        control_labels=cfg.primate_control_labels,
        treated_labels=[cfg.primate_treated_label],
        min_per_group= getattr(cfg, "min_samples_per_group_for_rejuv", 4),
        # min_per_group= 4,
        n_bootstrap=cfg.n_bootstrap,
        random_state=cfg.random_state,
    )
    
    logger.info("Global rejuvenation summary: %s", global_rejuv)


    # Summary by tissue
    tissue_col = cfg.tissue_col_candidates[0]  # p.ej. 'tissue'
    rejuv_by_tissue = summarize_rejuvenation_by_tissue(
        prim_meta,
        tissue_col=tissue_col,
        group_col=cfg.group_col_candidates[0],
        value_col="delta_age",
        control_labels=cfg.primate_control_labels,
        treated_labels=[cfg.primate_treated_label],
        min_per_group= getattr(cfg, "min_samples_per_group_for_rejuv", 4),
        # min_per_group = cfg.min_samples_for_mediation,
        n_bootstrap=cfg.n_bootstrap,
        random_state=cfg.random_state,
    )

    rejuv_by_tissue_path = cfg.results_dir / "rejuvenation_by_tissue.csv"
    rejuv_by_tissue.to_csv(rejuv_by_tissue_path, index=False)
    logger.info("Saved tissue-level rejuvenation summary to: %s", rejuv_by_tissue_path)

    # ---- Tissue-level effect sizes ----
    prim_tissue_effects = call_with_supported_kwargs(
        compute_group_effect_by_tissue,
        # Provide multiple possible parameter names for the expression matrix
        expr=prim_expr,
        matrix=prim_expr,
        data=prim_expr,
        X=prim_expr,
        # Provide multiple possible parameter names for metadata
        meta=prim_meta,
        metadata=prim_meta,
        meta_df=prim_meta,
        # Outcome column name your exosome module likely expects
        outcome_col="rejuvenation_score",
        y_col="rejuvenation_score",
        # Group/treatment fields
        group_col="group",
        treated_label=cfg.primate_treated_label,
        control_label=getattr(cfg, "control_label", None),
        control_labels=getattr(cfg, "primate_control_labels", None) or [cfg.control_label],
        tissue_col="tissue",
        # Top genes parameter name variants
        top_k=cfg.top_genes_per_tissue,
        top_n=cfg.top_genes_per_tissue,
        n_top=cfg.top_genes_per_tissue,
    )

    # ---- Primate plasma (OMIX007581 wide proteomics CSV with embedded sample IDs) ----
    plasma_path = cfg.primate_plasma.matrix

    if plasma_path.is_dir():
        candidates = sorted(plasma_path.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in {plasma_path}")
        plasma_file = candidates[0]
    else:
        plasma_file = plasma_path

    prim_plasma_expr = load_plasma_proteomics_csv(plasma_file)
    prim_plasma_meta = build_plasma_metadata_from_columns(list(prim_plasma_expr.columns))
    prim_plasma_expr = clean_plasma_matrix(
        prim_plasma_expr,
        min_feature_non_nan_frac=0.5,
        min_sample_non_nan_frac=0.5,
    )

    logger.info("Plasma matrix shape: %s", prim_plasma_expr.shape)
    logger.info(
        "Plasma groups: %s",
        prim_plasma_meta.get("group", pd.Series(dtype=str)).value_counts().to_dict(),
    )

    prim_plasma_state = call_with_supported_kwargs(
        build_plasma_state_score,
        plasma_matrix=prim_plasma_expr,  # main argument
        plasma_meta=prim_plasma_meta,  # in case the signature uses it
        meta=prim_plasma_meta,  # alias
        metadata=prim_plasma_meta,  # alias
        group_col="group",
        treated_label=cfg.primate_treated_label,
        control_label=getattr(cfg, "control_label", None),
        control_labels=getattr(cfg, "primate_control_labels", None),
        out_col="plasma_state_score",
    )

    # ---- Cross-pattern comparison (bulk vs plasma) ----
    pattern_comparison = call_with_supported_kwargs(
        compare_effect_patterns,
        prim_tissue_effects,  # effect_a
        prim_plasma_state,  # effect_b
        tissue_weighting=getattr(cfg, "tissue_weighting", None),
        weighting=getattr(cfg, "tissue_weighting", None),
    )
    if isinstance(pattern_comparison, dict) and pattern_comparison.get("n_common_tissues", 0) == 0:
        logger.warning("No common tissues for exosome fraction estimation; result will be non-informative.")

    # Estimate exosome-attributable fraction using bulk/tissue vs plasma proxy
    exo_fraction = call_with_supported_kwargs(
        estimate_exosome_fraction,
        prim_tissue_effects,  # -> effect_cells (positional)
        prim_plasma_state,  # -> effect_exosomes (positional)
        method=getattr(cfg, "exosome_fraction_method", None),
        strategy=getattr(cfg, "exosome_fraction_method", None),
    )

    # ---- Optional mediation ----
    med = None
    if cfg.enable_mediation:
        if "animal_id" in prim_meta.columns and "animal_id" in prim_plasma_meta.columns:
            if "group_binary" not in prim_meta.columns:
                # Minimal robust encoding
                prim_meta = prim_meta.copy()
                prim_meta["group_binary"] = (prim_meta["group"] == cfg.primate_treated_label).astype(int)

            prim_merge = prim_meta.merge(
                prim_plasma_meta[["animal_id", "plasma_state_score"]],
                on="animal_id",
                how="inner",
            )

            if len(prim_merge) >= cfg.min_samples_for_mediation:
                med = simple_mediation_bootstrap(
                    prim_merge,
                    x_col="group_binary",
                    m_col="plasma_state_score",
                    y_col="rejuvenation_score",
                    n_boot=cfg.mediation_bootstrap,
                    seed=cfg.random_seed,
                )
            else:
                logger.warning("Too few samples for stable mediation; skipping.")

    # ------------------------- Translational insights -------------------------
    prim_outcomes = prim_meta.copy()
    for col in ("sample_id", "group", "tissue", "rejuvenation_score", "predicted_age_cv", "predicted_age"):
        if col not in prim_outcomes.columns:
            prim_outcomes[col] = pd.NA

    prim_ctrl = getattr(cfg, "primate_control_labels", None) or [cfg.control_label]
    mouse_tr = getattr(cfg, "mouse_treated_label", "Exosome")
    mouse_ctrl = getattr(cfg, "mouse_control_labels", None) or ["Saline", "Control", "WTC"]
    insights = None
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
                str(cfg.figures_dir / "tissue_concordance.png"),
            )
        else:
            logger.warning("Skipping tissue concordance: mouse_tissue_effects not available.")
    except Exception as e:
        logger.warning("plot_tissue_concordance skipped: %s", str(e))

    # ---- Plasma biomarker ranking ----
    try:
        plot_plasma_biomarker_ranking(
            prim_plasma_expr,
            str(cfg.figures_dir / "plasma_biomarker_ranking.png"),
            top_n=cfg.top_plasma_biomarkers,
        )
    except Exception as e:
        logger.warning("plot_plasma_biomarker_ranking skipped: %s", e)

    logger.info("Done. Exosome-attributable fraction estimate: %s", exo_fraction)
    if insights:
        logger.info("Translational summary written to results directory.")


# -----------------------------------------------------------------------------
# Minimal CLI + defaults
# -----------------------------------------------------------------------------
def _build_default_config(base_dir: Path) -> PipelineConfig:
    """
    Centralized defaults.
    Adjust filenames to your real downloaded names.
    """
    data = base_dir / "data/PROCESSED"

    return PipelineConfig(
        primate_bulk=OmixPaths(
            matrix=data / "OMIX007580_01_example.txt",
            metadata=data / "OMIX007580-02_example.csv",
        ),
        primate_plasma=OmixPaths(
            matrix=data / "OMIX007581-01_example.csv",
            metadata=data / "OMIX007581_metadata.xlsx",
        ),
        primate_methylation=OmixPaths(
            matrix=data / "OMIX007582_beta_matrix_example.csv",  # technical IDs by default
            metadata=data / "OMIX007582-02_example.csv",
        ),
        mouse_exosome_bulk=OmixPaths(
            matrix=data / "OMIX009283-01.txt",
            metadata=data / "OMIX009283_metadata.csv",
        ),
        results_dir=base_dir / "results",
        figures_dir=base_dir / "figures",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe", action="store_true", help="Extra conservative mode for laptops.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    cfg = _build_default_config(base_dir)

    # Conservative overrides for your hardware
    if args.safe:
        cfg.n_top_features_expr = 3000
        cfg.n_top_features_clock = 1500
        cfg.top_genes_per_tissue = min(cfg.top_genes_per_tissue, 75)
        cfg.enable_mediation = False

    run(cfg)


if __name__ == "__main__":
    main()