"""
preprocessing.py

Preprocessing utilities for bulk and plasma omics matrices, including
metadata standardization, variance-based feature filtering, log transforms
and construction of proxy rejuvenation scores at the sample level.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .omix_io import find_first_present_column

logger = get_logger(__name__)


def standardize_metadata_columns(
    meta: pd.DataFrame,
    group_candidates: List[str],
    tissue_candidates: List[str],
    age_candidates: List[str],
    sex_candidates: List[str],
    animal_id_candidates: List[str]
) -> pd.DataFrame:
    """
    Standardize key metadata columns into:
        group, tissue, age, sex, animal_id

    Missing columns are created as NA.

    Returns
    -------
    pd.DataFrame
    """
    meta = meta.copy()

    mapping = {
        "group": find_first_present_column(meta, group_candidates),
        "tissue": find_first_present_column(meta, tissue_candidates),
        "age": find_first_present_column(meta, age_candidates),
        "sex": find_first_present_column(meta, sex_candidates),
        "animal_id": find_first_present_column(meta, animal_id_candidates),
    }

    for new_name, old_name in mapping.items():
        if old_name and old_name != new_name:
            meta = meta.rename(columns={old_name: new_name})
        elif not old_name:
            meta[new_name] = pd.NA

    for col in ("group", "tissue", "sex", "animal_id"):
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip()

    return meta


def log1p_counts(matrix: pd.DataFrame, assume_counts: bool = True) -> pd.DataFrame:
    """
    Apply log1p transform for count-like matrices.

    Parameters
    ----------
    matrix : pd.DataFrame
    assume_counts : bool
        If True, apply log1p without heuristics.

    Returns
    -------
    pd.DataFrame
    """
    if assume_counts:
        return np.log1p(matrix)

    vals = matrix.values
    if np.nanmax(vals) > 100:
        return np.log1p(matrix)

    return matrix


def filter_top_variance(matrix: pd.DataFrame, n_top: int) -> pd.DataFrame:
    """
    Keep the top n_top most variable rows (features).
    Uses position-based indexing to avoid expensive label reindexing.
    Hard-stops on suspiciously wide matrices.
    """
    n_top = int(n_top or 0)

    # Absolute safety: if this triggers, something upstream is still wrong
    if matrix.shape[1] > 20000:
        raise ValueError(
            f"Matrix too wide for variance filtering ({matrix.shape[1]} columns). "
            "Upstream matrix selection/alignment likely failed."
        )

    n_rows = matrix.shape[0]
    if n_top <= 0 or n_rows == 0:
        return matrix

    # If you ask for more features than exist, just skip
    if n_top >= n_rows:
        logger.info("Skipping variance filter: n_top=%d >= n_rows=%d", n_top, n_rows)
        return matrix

    # Fast numeric variance over columns (samples)
    arr = matrix.to_numpy(copy=False)

    # ddof=0 for stability, and avoid NaN issues
    var = np.nanvar(arr, axis=1)

    # Partial selection for speed
    idx = np.argpartition(var, -n_top)[-n_top:]
    idx = idx[np.argsort(var[idx])[::-1]]

    return matrix.iloc[idx]


def build_proxy_rejuvenation_score(
     meta: pd.DataFrame,
    pred_age_col: str = "predicted_age",
    chrono_age_col: str = "age",
    out_col: str = "rejuvenation_score",
    **kwargs,
) -> pd.DataFrame:
    meta = meta.copy()

    if pred_age_col not in meta.columns or chrono_age_col not in meta.columns:
        raise ValueError(
            f"Missing columns for rejuvenation score: "
            f"{pred_age_col} or {chrono_age_col}"
        )

    pred = pd.to_numeric(meta[pred_age_col], errors="coerce")
    chrono = pd.to_numeric(meta[chrono_age_col], errors="coerce")

    meta[out_col] = chrono - pred  # positive = "younger than expected"
    return meta