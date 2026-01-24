"""
clocks.py

Implements lightweight transcriptomic clocks for primate bulk data and
helper functions to predict biological age and interface with downstream
rejuvenation scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import logging

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from .logging_utils import get_logger

logger = logging.getLogger(__name__)


@dataclass
class SimpleClock:
    coef: np.ndarray
    intercept: float
    feature_index: pd.Index


@dataclass
class TrainedClock:
    """
    Container for a trained transcriptomic clock.

    Attributes
    ----------
    estimator : Any
        Fitted sklearn-like estimator implementing .predict().
    features : list of str
        Gene/feature names used during training (column names in expr_log).
    """
    estimator: Any
    features: list[str]


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
    expr_log: pd.DataFrame,
    meta: pd.DataFrame,
    age_col: str = "agenumb",
    model: Optional[str] = "ridge",
    n_top_features: Optional[int] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[TrainedClock, pd.DataFrame, Dict[str, float]]:
    """
    Train a simple transcriptomic aging clock and evaluate it with
    honest out-of-sample predictions using K-fold cross-validation.

    Parameters
    ----------
    expr_log : DataFrame
        Log-transformed expression matrix (genes x samples).
    meta : DataFrame
        Metadata with at least ['sample_id', age_col].
    age_col : str
        Column in meta containing chronological age (numeric).
    model : {"ridge", None}, optional
        Type of regressor to use (currently only Ridge).
    n_top_features : int or None
        (Currently ignored inside this function; feature reduction is
        expected to be handled upstream in the pipeline.)
    n_splits : int
        Number of CV folds.
    random_state : int
        Random seed for shuffling in KFold and Ridge.

    Returns
    -------
    clock : TrainedClock
        Container with fitted estimator and feature list used at training.
    pred_df : DataFrame
        One row per training sample with:
        - 'sample_id'
        - 'predicted_age' (cross-validated prediction)
        - age_col (true age)
    metrics : dict
        MAE, RMSE, Pearson/Spearman correlation, calibration slope, n_samples.
    """
    if "sample_id" not in meta.columns:
        raise ValueError("Metadata must contain a 'sample_id' column for the clock.")

    # Keep only samples with numeric age
    meta = meta.copy()
    meta[age_col] = pd.to_numeric(meta[age_col], errors="coerce")
    mask = meta[age_col].notna()

    if mask.sum() < max(20, n_splits * 3):
        raise ValueError(
            f"Not enough samples with valid {age_col} to train a clock "
            f"({mask.sum()} found; need at least ~{max(20, n_splits * 3)})."
        )

    meta_train = meta.loc[mask].copy()

    # Align expression and metadata by sample_id
    sample_ids = [s for s in meta_train["sample_id"].astype(str) if s in expr_log.columns]
    if len(sample_ids) < max(20, n_splits * 3):
        raise ValueError(
            f"After alignment only {len(sample_ids)} samples remain. "
            "Check that 'sample_id' matches expression column names."
        )

    meta_train = meta_train.set_index("sample_id").loc[sample_ids]
    X = expr_log.loc[:, sample_ids].T  # samples x features
    y = meta_train[age_col].astype(float).values

    feature_names_used = list(X.columns)

    # Choose model
    if model is None or str(model).lower() == "ridge":
        base_estimator = Ridge(alpha=1.0, random_state=random_state)
    else:
        logger.warning("Unknown clock_model '%s'; falling back to Ridge.", model)
        base_estimator = Ridge(alpha=1.0, random_state=random_state)

    # Cross-validation OUT-OF-SAMPLE
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred_cv = cross_val_predict(base_estimator, X.values, y, cv=cv)

    # Metrics
    mae = mean_absolute_error(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    r_pearson = pearsonr(y, y_pred_cv)[0]
    r_spearman = spearmanr(y, y_pred_cv).correlation

    # Calibration slope (y_pred vs y_true)
    slope = np.polyfit(y, y_pred_cv, 1)[0]

    metrics = {
        "n_samples": float(len(y)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "pearson_r": float(r_pearson),
        "spearman_r": float(r_spearman),
        "calibration_slope": float(slope),
    }

    logger.info(
        "Clock CV performance (n=%d): MAE=%.3f, RMSE=%.3f, r=%.3f, slope=%.3f",
        len(y),
        mae,
        rmse,
        r_pearson,
        slope,
    )

    # Fit the model with all the data (on the same feature set X)
    fitted_estimator = base_estimator.fit(X.values, y)

    # Wrap everything into a TrainedClock
    clock = TrainedClock(
        estimator=fitted_estimator,
        features=feature_names_used,
    )

    # DataFrame with cross-validated predictions
    pred_df = pd.DataFrame(
        {
            "sample_id": meta_train.index.values,
            "predicted_age": y_pred_cv,
            age_col: y,
        }
    )

    return clock, pred_df, metrics


def predict_biological_age(
    model: Any,
    expr_log: pd.DataFrame,
    meta: pd.DataFrame,
    sample_id_col: str = "sample_id",
) -> pd.DataFrame:
    """
    Use a trained transcriptomic clock to predict biological age.

    Assumes that `expr_log` has the same feature set (genes) as the matrix
    used during training. This is true in the current SRSC pipeline, where
    the same `expr_log` is passed to both training and prediction.

    Parameters
    ----------
    model
        Fitted clock object. Can be:
        - TrainedClock instance
        - A raw sklearn estimator
        - Or the (model, pred_df, metrics) tuple returned by
          `train_transcriptomic_clock`, in which case the first element
          is used.
    expr_log : DataFrame
        Log-transformed expression matrix (genes x samples).
    meta : DataFrame
        Metadata table containing a sample ID column.
    sample_id_col : str, default "sample_id"
        Column in `meta` that matches the columns of `expr_log`.

    Returns
    -------
    pred_df : DataFrame
        DataFrame with columns ['sample_id', 'predicted_age'].
    """
    # Unwrap if caller passed the full (model, pred_df, metrics) tuple
    if isinstance(model, tuple):
        model = model[0]

    # If it's our wrapper, extract the underlying estimator
    if isinstance(model, TrainedClock):
        estimator = model.estimator
    else:
        estimator = model

    if sample_id_col not in meta.columns:
        raise ValueError(f"Metadata is missing column '{sample_id_col}'.")

    # 1D list of sample IDs
    sample_ids = meta[sample_id_col].astype(str).tolist()

    # Drop IDs not present in the expression matrix
    missing_ids = [sid for sid in sample_ids if sid not in expr_log.columns]
    if missing_ids:
        logger.warning(
            "[predict_biological_age] %d sample IDs not found in expression matrix; "
            "they will be skipped. Examples: %s",
            len(missing_ids),
            missing_ids[:5],
        )
        sample_ids = [sid for sid in sample_ids if sid in expr_log.columns]
        meta = meta[meta[sample_id_col].isin(sample_ids)].copy()

    if not sample_ids:
        raise ValueError("No overlapping sample IDs between metadata and expression matrix.")

    # genes x samples → samples x genes
    X_full = expr_log.loc[:, sample_ids].T  # shape: (n_samples, n_genes)

    # IMPORTANT: we assume here that the feature set (genes) is the same as
    # the one used during training. This holds in the current pipeline because
    # training and prediction both receive the same `expr_log`.
    X_values = X_full.to_numpy()
    y_pred = estimator.predict(X_values)

    pred_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "predicted_age": y_pred,
        }
    )
    return pred_df


def summarize_clock_performance(
    meta: pd.DataFrame,
    chrono_col: str = "agenumb",
    pred_col: str = "predicted_age",
) -> Dict[str, float]:
    """
    Compute basic performance metrics for an aging clock.

    Returns a dict with MAE, RMSE, Pearson r, Spearman rho, and
    a simple calibration slope (OLS on y_pred ~ y_true).
    """
    df = meta[[chrono_col, pred_col]].copy()
    df = df.dropna()

    if df.shape[0] < 10:
        raise ValueError(
            f"Not enough paired ages to evaluate clock: n={df.shape[0]}"
        )

    y_true = df[chrono_col].astype(float).values
    y_pred = df[pred_col].astype(float).values

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    # Correlations
    r_pearson, _ = pearsonr(y_true, y_pred)
    r_spearman, _ = spearmanr(y_true, y_pred)

    # Calibration slope (OLS univariado)
    x = y_true
    y = y_pred
    slope = float(
        np.dot(x - x.mean(), y - y.mean())
        / np.dot(x - x.mean(), x - x.mean())
    )

    return {
        "n_samples": float(df.shape[0]),
        "mae": mae,
        "rmse": rmse,
        "pearson_r": float(r_pearson),
        "spearman_r": float(r_spearman),
        "calibration_slope": slope,
    }