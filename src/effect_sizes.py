"""Small, dependency-light effect-size utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def hedges_g(treated: Sequence[float], control: Sequence[float]) -> float:
    """Return bias-corrected standardized mean difference (treated - control)."""
    treated_arr = np.asarray(list(treated), dtype=float)
    control_arr = np.asarray(list(control), dtype=float)
    treated_arr = treated_arr[np.isfinite(treated_arr)]
    control_arr = control_arr[np.isfinite(control_arr)]
    n_treated = int(len(treated_arr))
    n_control = int(len(control_arr))
    degrees_freedom = n_treated + n_control - 2
    if n_treated < 2 or n_control < 2 or degrees_freedom <= 1:
        return np.nan
    pooled_variance = (
        (n_treated - 1) * float(np.var(treated_arr, ddof=1))
        + (n_control - 1) * float(np.var(control_arr, ddof=1))
    ) / float(degrees_freedom)
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return np.nan
    cohen_d = float((treated_arr.mean() - control_arr.mean()) / np.sqrt(pooled_variance))
    correction = 1.0 - 3.0 / (4.0 * float(degrees_freedom) - 1.0)
    return float(correction * cohen_d)
