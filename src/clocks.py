"""
clocks.py

Implements lightweight transcriptomic clocks for primate bulk data and
helper functions to predict biological age and interface with downstream
rejuvenation scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SimpleClock:
    coef: np.ndarray
    intercept: float
    feature_index: pd.Index


def _parse_age_to_years(obj) -> pd.Series:
    """
    Accepts a Series or (in edge cases) a single-column DataFrame.
    Returns numeric ages with NaN for non-parseable entries.
    """
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(dtype="float64")
        obj = obj.iloc[:, 0]

    s = obj.astype(str).str.strip()
    num = s.str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(num, errors="coerce")

def train_transcriptomic_clock(
    matrix: pd.DataFrame,
    meta: pd.DataFrame,
    age_col: str = "agenumb",
    model: Optional[str] = None,
    n_top_features: int = 3000,
    **kwargs,
) -> SimpleClock:
    """
    Train a very lightweight linear age model.
    matrix: rows=features (genes), cols=samples
    meta: must contain 'sample_id' and age_col
    """
    if "sample_id" not in meta.columns:
        raise ValueError("Metadata must contain 'sample_id'.")

    meta_idx = meta.set_index("sample_id", drop=False)

    # Ensure we train on intersection of samples
    common = [c for c in matrix.columns if c in meta_idx.index]
    if len(common) < 5:
        raise ValueError("Too few common samples between matrix and metadata for clock training.")

    age_series = _parse_age_to_years(meta_idx.loc[common, age_col])
    valid_mask = age_series.notna().values

    if valid_mask.sum() < 5:
        raise ValueError(
            f"Too few valid numeric ages after parsing '{age_col}'. "
            "Your age column may be wrong."
        )

    common_valid = [s for s, ok in zip(common, valid_mask) if ok]

    # X: samples x features
    X = matrix.loc[:, common_valid].T.to_numpy(dtype=np.float32, copy=False)
    y = age_series.loc[common_valid].to_numpy(dtype=np.float32, copy=False)

    # Center for numerical stability
    X_mean = X.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    Xc = X - X_mean
    yc = y - y_mean

    # Ridge-like closed form with tiny lambda for stability
    # (keeps dependencies minimal)
    lam = 1e-3
    XtX = Xc.T @ Xc
    XtX.flat[:: XtX.shape[0] + 1] += lam
    Xty = Xc.T @ yc

    coef = np.linalg.solve(XtX, Xty).astype(np.float32)
    intercept = y_mean - float((X_mean @ coef.reshape(-1, 1)).ravel()[0])

    return SimpleClock(coef=coef, intercept=intercept, feature_index=matrix.index)


def predict_biological_age(clock, matrix, meta=None, **kwargs
) -> pd.DataFrame:
    """
    Predict age for samples in matrix.
    Returns a DataFrame with sample_id and predicted_age.
    """
    # Align features
    common_features = matrix.index.intersection(clock.feature_index)
    if len(common_features) < 50:
        logger.warning("Low feature overlap for prediction: %d", len(common_features))

    mat = matrix.loc[common_features]
    coef = pd.Series(clock.coef, index=clock.feature_index).loc[common_features].to_numpy()

    X = mat.T.to_numpy(dtype=np.float32, copy=False)
    pred = X @ coef + clock.intercept

    out = pd.DataFrame({"sample_id": mat.columns.astype(str), "predicted_age": pred})

    if meta is not None and "sample_id" in meta.columns:
        out = out.merge(meta[["sample_id"]], on="sample_id", how="right")

    return out