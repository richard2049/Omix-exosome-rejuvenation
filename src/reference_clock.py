"""Controls-only, animal-grouped transcriptomic reference clock.

This module implements the prospectively frozen Ridge branch selected for the
public macaque clock: log2-CPM input, controls-only fitting, nested grouped
hyperparameter selection, and cross-fitted predictions for every eligible
animal.  Treatment samples can be predicted but can never enter training.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .clocks import TrainedClock


@dataclass
class ReferenceClockRun:
    clock: TrainedClock
    primary_predictions: pd.DataFrame
    all_predictions: pd.DataFrame
    metrics: dict[str, object]
    fold_audit: pd.DataFrame
    hyperparameter_audit: pd.DataFrame


FROZEN_FOLD_COLUMNS = ("animal_id", "legacy", "seed_17", "seed_29", "seed_43")


def frozen_fold_assignment_digest(assignments: pd.DataFrame) -> str:
    """Hash scientific fold content independently of CSV line endings."""
    missing = sorted(set(FROZEN_FOLD_COLUMNS) - set(assignments.columns))
    if missing:
        raise ValueError(f"Frozen fold assignments are missing columns: {missing}")
    canonical = assignments.loc[:, FROZEN_FOLD_COLUMNS].copy()
    canonical["animal_id"] = canonical["animal_id"].astype(str)
    for column in FROZEN_FOLD_COLUMNS[1:]:
        numeric = pd.to_numeric(canonical[column], errors="coerce")
        if numeric.isna().any() or (numeric != np.floor(numeric)).any():
            raise ValueError(f"Frozen fold column {column} contains invalid values")
        canonical[column] = numeric.astype(int)
    canonical = canonical.sort_values("animal_id").reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_exact_sample_exclusions(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exclude only explicitly named samples and return a provenance ledger."""
    if "sample_id" not in metadata.columns:
        raise ValueError("Clock exclusions require metadata.sample_id")
    meta = metadata.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    if meta["sample_id"].duplicated().any():
        raise ValueError("Clock exclusions require unique metadata sample identifiers")
    if expression.columns.duplicated().any():
        raise ValueError("Clock exclusions require unique expression sample identifiers")

    exclusions = [str(sample_id) for sample_id in sample_ids]
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("Clock exclusion list contains duplicate sample identifiers")
    if not exclusions:
        return expression, meta, pd.DataFrame(
            columns=["sample_id", "animal_id", "group", "reason"]
        )

    missing_meta = sorted(set(exclusions) - set(meta["sample_id"]))
    missing_matrix = sorted(set(exclusions) - set(map(str, expression.columns)))
    if missing_meta or missing_matrix:
        raise ValueError(
            "Approved clock exclusions do not match the current inputs; "
            f"missing_metadata={missing_meta}, missing_matrix={missing_matrix}"
        )

    ledger_columns = [
        column for column in ("sample_id", "animal_id", "group", "tissue", "age")
        if column in meta.columns
    ]
    ledger = meta.loc[meta["sample_id"].isin(exclusions), ledger_columns].copy()
    if len(ledger) != len(exclusions):
        raise ValueError("Each approved clock exclusion must identify exactly one sample")
    ledger["reason"] = "unresolved_declared_animal_alias"
    keep = ~meta["sample_id"].isin(exclusions)
    retained_meta = meta.loc[keep].copy()
    retained_expression = expression.loc[:, retained_meta["sample_id"].tolist()].copy()
    retained_expression.attrs.update(expression.attrs)
    return retained_expression, retained_meta, ledger.reset_index(drop=True)


def assign_animal_folds(
    metadata: pd.DataFrame,
    *,
    n_splits: int,
    seed: Optional[int],
) -> np.ndarray:
    """Assign every row from one animal to the same outer fold."""
    if "animal_id" not in metadata.columns or metadata["animal_id"].isna().any():
        raise ValueError("Animal-grouped cross-fitting requires complete animal_id values")
    n_animals = int(metadata["animal_id"].astype(str).nunique())
    if n_animals < n_splits:
        raise ValueError(f"Need at least {n_splits} animals; found {n_animals}")
    splitter = GroupKFold(
        n_splits=n_splits,
        shuffle=seed is not None,
        random_state=seed,
    )
    labels = np.full(len(metadata), -1, dtype=int)
    groups = metadata["animal_id"].astype(str).to_numpy()
    for fold, (train, test) in enumerate(splitter.split(metadata, groups=groups)):
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Animal overlap detected across outer folds")
        labels[test] = fold
    if (labels < 0).any():
        raise ValueError("Outer fold assignment left samples unassigned")
    return labels


def assign_frozen_animal_folds(
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    split: str,
    n_splits: int,
) -> np.ndarray:
    """Apply an exact animal-to-fold map without depending on sample counts."""
    required = {"animal_id", split}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Frozen fold assignments are missing columns: {missing}")
    table = assignments[["animal_id", split]].copy()
    table["animal_id"] = table["animal_id"].astype(str)
    if table["animal_id"].duplicated().any() or table["animal_id"].isna().any():
        raise ValueError("Frozen fold assignments contain duplicate or missing animals")
    observed_animals = set(metadata["animal_id"].astype(str))
    mapped_animals = set(table["animal_id"])
    if observed_animals != mapped_animals:
        raise ValueError(
            "Frozen fold animal set differs from eligible metadata; "
            f"missing={sorted(observed_animals - mapped_animals)}, "
            f"unexpected={sorted(mapped_animals - observed_animals)}"
        )
    numeric = pd.to_numeric(table[split], errors="coerce")
    if numeric.isna().any() or (numeric != np.floor(numeric)).any():
        raise ValueError(f"Frozen fold column {split} contains invalid values")
    table[split] = numeric.astype(int)
    if set(table[split]) != set(range(n_splits)):
        raise ValueError(f"Frozen fold column {split} does not cover folds 0..{n_splits - 1}")
    mapped = metadata["animal_id"].astype(str).map(table.set_index("animal_id")[split])
    if mapped.isna().any():
        raise ValueError(f"Frozen fold column {split} left animals unmapped")
    return mapped.to_numpy(dtype=int)


def _animal_mae(metadata: pd.DataFrame, observed: np.ndarray, predicted: np.ndarray) -> float:
    errors = pd.Series(
        np.abs(np.asarray(observed) - np.asarray(predicted)),
        index=metadata["animal_id"].astype(str).to_numpy(),
    )
    return float(errors.groupby(level=0).mean().mean())


def _select_ridge_alpha(
    x: np.ndarray,
    metadata: pd.DataFrame,
    alphas: Sequence[float],
    *,
    inner_folds: int,
    inner_seed: int,
) -> tuple[float, list[dict[str, object]]]:
    ordered_alphas = sorted({float(alpha) for alpha in alphas}, reverse=True)
    if not ordered_alphas or any(alpha <= 0 for alpha in ordered_alphas):
        raise ValueError("Ridge alpha grid must contain unique positive values")
    groups = metadata["animal_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < inner_folds:
        raise ValueError("Insufficient reference animals for inner grouped validation")
    ages = metadata["age"].to_numpy(dtype=float)
    splitter = GroupKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=inner_seed,
    )
    predictions = np.full((len(ordered_alphas), len(metadata)), np.nan, dtype=float)
    failures: dict[int, list[str]] = {index: [] for index in range(len(ordered_alphas))}

    for fold, (train, validation) in enumerate(splitter.split(x, groups=groups)):
        if set(groups[train]) & set(groups[validation]):
            raise ValueError("Animal overlap detected across inner folds")
        preprocessing = make_pipeline(VarianceThreshold(), StandardScaler())
        try:
            x_train = preprocessing.fit_transform(x[train])
            x_validation = preprocessing.transform(x[validation])
        except ValueError as exc:
            raise ValueError(f"Inner preprocessing failed: {exc}") from exc
        if np.unique(ages[train]).size < 2:
            raise ValueError("Inner training fold has insufficient age variation")
        for candidate, alpha in enumerate(ordered_alphas):
            try:
                model = Ridge(alpha=alpha, solver="cholesky")
                model.fit(x_train, ages[train])
                predicted = model.predict(x_validation)
                if not np.isfinite(predicted).all():
                    raise ValueError("non-finite validation predictions")
                predictions[candidate, validation] = predicted
            except (ValueError, FloatingPointError) as exc:
                failures[candidate].append(f"inner_fold={fold}:{exc}")

    audit: list[dict[str, object]] = []
    for candidate, alpha in enumerate(ordered_alphas):
        valid = bool(np.isfinite(predictions[candidate]).all())
        audit.append(
            {
                "candidate": candidate,
                "alpha": alpha,
                "valid_all_inner_folds": valid,
                "animal_mae": (
                    _animal_mae(metadata, ages, predictions[candidate])
                    if valid
                    else np.nan
                ),
                "failure": "|".join(failures[candidate]),
            }
        )
    valid_rows = [row for row in audit if row["valid_all_inner_folds"]]
    if not valid_rows:
        raise ValueError("No Ridge candidate was valid in every inner fold")
    selected = min(valid_rows, key=lambda row: (float(row["animal_mae"]), int(row["candidate"])))
    return float(selected["alpha"]), audit


def _fit_outer_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, object]:
    if np.unique(y_train).size < 2:
        raise ValueError("Outer reference training fold has insufficient age variation")
    estimator = _fit_ridge_estimator(x_train, y_train, alpha)
    predicted = estimator.predict(x_test)
    if not np.isfinite(predicted).all():
        raise ValueError("Outer Ridge fit produced non-finite predictions")
    return predicted, estimator


def _fit_ridge_estimator(x: np.ndarray, y: np.ndarray, alpha: float) -> object:
    if np.unique(y).size < 2:
        raise ValueError("Reference training data have insufficient age variation")
    estimator = make_pipeline(
        VarianceThreshold(),
        StandardScaler(),
        Ridge(alpha=alpha, solver="cholesky"),
    )
    estimator.fit(x, y)
    return estimator


def _reference_metrics(
    metadata: pd.DataFrame,
    predictions: pd.DataFrame,
    reference_groups: Sequence[str],
) -> dict[str, float]:
    evaluation = metadata.merge(
        predictions[["sample_id", "predicted_age"]],
        on="sample_id",
        validate="one_to_one",
    )
    evaluation = evaluation.loc[evaluation["group"].isin(reference_groups)].copy()
    if evaluation.empty or not np.isfinite(evaluation["predicted_age"]).all():
        raise ValueError("Reference metrics require complete finite control predictions")
    evaluation["absolute_error"] = np.abs(evaluation["predicted_age"] - evaluation["age"])
    evaluation["squared_error"] = (evaluation["predicted_age"] - evaluation["age"]) ** 2
    animal = evaluation.groupby("animal_id", as_index=False).agg(
        age=("age", "mean"),
        predicted_age=("predicted_age", "mean"),
        absolute_error=("absolute_error", "mean"),
        squared_error=("squared_error", "mean"),
    )
    if len(animal) < 3 or animal["age"].nunique() < 2:
        raise ValueError("Reference metrics require at least three animals and two ages")
    return {
        "n_samples": float(len(evaluation)),
        "n_animals": float(len(animal)),
        "MAE": float(animal["absolute_error"].mean()),
        "RMSE": float(np.sqrt(animal["squared_error"].mean())),
        "pearson_r": float(pearsonr(animal["age"], animal["predicted_age"])[0]),
        "spearman_r": float(spearmanr(animal["age"], animal["predicted_age"]).correlation),
        "calibration_slope": float(np.polyfit(animal["age"], animal["predicted_age"], 1)[0]),
        "sample_weighted_MAE": float(evaluation["absolute_error"].mean()),
        "sample_weighted_RMSE": float(np.sqrt(evaluation["squared_error"].mean())),
    }


def train_reference_transcriptomic_clock(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    reference_groups: Sequence[str],
    ridge_alphas: Sequence[float],
    outer_folds: int = 5,
    inner_folds: int = 3,
    inner_seed: int = 20260917,
    diagnostic_seeds: Sequence[int] = (17, 29, 43),
    frozen_fold_assignments: Optional[pd.DataFrame] = None,
) -> ReferenceClockRun:
    """Fit the approved controls-only Ridge protocol and return all audits."""
    if expression.attrs.get("representation") != "log2_cpm":
        raise ValueError("Reference clock requires the approved log2_cpm representation")
    required = {"sample_id", "animal_id", "group", "age"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Reference clock metadata is missing columns: {missing}")
    meta = metadata.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["animal_id"] = meta["animal_id"].astype("string").str.strip()
    meta["group"] = meta["group"].astype("string").str.strip()
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    if meta[["sample_id", "animal_id", "group", "age"]].isna().any().any():
        raise ValueError("Reference clock metadata contains missing required values")
    if meta["sample_id"].duplicated().any():
        raise ValueError("Reference clock metadata contains duplicate sample identifiers")
    if set(meta["sample_id"]) != set(map(str, expression.columns)):
        raise ValueError("Reference clock expression and metadata sample sets differ")
    if not np.isfinite(meta["age"]).all() or (meta["age"] < 0).any():
        raise ValueError("Reference clock metadata contains invalid ages")
    if (meta.groupby("animal_id")[["age", "group"]].nunique() > 1).any().any():
        raise ValueError("Reference clock animal metadata is internally contradictory")
    reference_groups = tuple(str(group) for group in reference_groups)
    reference = meta["group"].isin(reference_groups)
    if not reference.any():
        raise ValueError("No samples belong to the configured reference groups")
    if meta.loc[reference, "animal_id"].nunique() < max(outer_folds, inner_folds + 1):
        raise ValueError("Insufficient reference animals for nested grouped validation")

    ordered_samples = meta["sample_id"].tolist()
    x = expression.loc[:, ordered_samples].T.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("Reference clock expression contains non-finite values")
    partitions: Mapping[str, Optional[int]] = {
        "legacy": None,
        **{f"seed_{int(seed)}": int(seed) for seed in diagnostic_seeds},
    }
    alpha_min = min(map(float, ridge_alphas))
    alpha_max = max(map(float, ridge_alphas))
    all_prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    hyper_rows: list[dict[str, object]] = []

    for split_name, seed in partitions.items():
        if frozen_fold_assignments is None:
            folds = assign_animal_folds(meta, n_splits=outer_folds, seed=seed)
        else:
            folds = assign_frozen_animal_folds(
                meta,
                frozen_fold_assignments,
                split=split_name,
                n_splits=outer_folds,
            )
        split_predictions = np.full(len(meta), np.nan, dtype=float)
        for fold in range(outer_folds):
            test = folds == fold
            train = (~test) & reference.to_numpy()
            train_meta = meta.loc[train].reset_index(drop=True)
            test_meta = meta.loc[test].reset_index(drop=True)
            train_animals = set(train_meta["animal_id"].astype(str))
            test_animals = set(test_meta["animal_id"].astype(str))
            overlap = train_animals & test_animals
            if overlap:
                raise ValueError(f"Animal leakage in {split_name} fold {fold}: {sorted(overlap)}")
            if not train_meta["group"].isin(reference_groups).all():
                raise ValueError("Treatment samples entered reference-clock training")
            alpha, candidate_audit = _select_ridge_alpha(
                x[train],
                train_meta,
                ridge_alphas,
                inner_folds=inner_folds,
                inner_seed=inner_seed,
            )
            predicted, _ = _fit_outer_ridge(
                x[train],
                train_meta["age"].to_numpy(dtype=float),
                x[test],
                alpha,
            )
            split_predictions[test] = predicted
            for row in candidate_audit:
                hyper_rows.append({"split": split_name, "outer_fold": fold, **row})
            fold_rows.append(
                {
                    "split": split_name,
                    "fold": fold,
                    "n_train_samples": int(train.sum()),
                    "n_train_animals": int(len(train_animals)),
                    "n_test_samples": int(test.sum()),
                    "n_test_animals": int(len(test_animals)),
                    "n_train_treated": int((~train_meta["group"].isin(reference_groups)).sum()),
                    "animal_overlap": int(len(overlap)),
                    "selected_alpha": alpha,
                    "alpha_boundary": bool(alpha in (alpha_min, alpha_max)),
                    "representation": str(expression.attrs.get("representation", "unspecified")),
                }
            )
        if not np.isfinite(split_predictions).all():
            raise ValueError(f"Cross-fitting left incomplete predictions for {split_name}")
        all_prediction_frames.append(
            pd.DataFrame(
                {
                    "sample_id": ordered_samples,
                    "predicted_age": split_predictions,
                    "age": meta["age"].to_numpy(dtype=float),
                    "animal_id": meta["animal_id"].astype(str).to_numpy(),
                    "group": meta["group"].astype(str).to_numpy(),
                    "split": split_name,
                    "fold": folds,
                }
            )
        )

    all_predictions = pd.concat(all_prediction_frames, ignore_index=True)
    primary = all_predictions.loc[all_predictions["split"].eq("legacy")].copy()
    primary = primary.sort_values("sample_id").reset_index(drop=True)
    metrics: dict[str, object] = _reference_metrics(meta, primary, reference_groups)

    reference_meta = meta.loc[reference].reset_index(drop=True)
    final_alpha, final_audit = _select_ridge_alpha(
        x[reference.to_numpy()],
        reference_meta,
        ridge_alphas,
        inner_folds=inner_folds,
        inner_seed=inner_seed,
    )
    final_estimator = _fit_ridge_estimator(
        x[reference.to_numpy()],
        reference_meta["age"].to_numpy(dtype=float),
        final_alpha,
    )
    for row in final_audit:
        hyper_rows.append({"split": "final", "outer_fold": -1, **row})

    metrics.update(
        {
            "n_evaluation_samples": float(len(meta)),
            "n_evaluation_animals": float(meta["animal_id"].nunique()),
            "training_scope": "controls_only_global",
            "reference_groups": "|".join(reference_groups),
            "representation": str(expression.attrs.get("representation", "unspecified")),
            "cv_strategy": "nested_animal_grouped_cross_fitting",
            "cv_n_splits": float(outer_folds),
            "cv_n_groups": float(meta["animal_id"].nunique()),
            "diagnostic_partitions": "|".join(partitions),
            "n_features": float(x.shape[1]),
            "input_dtype": str(x.dtype),
            "ridge_alpha": final_alpha,
            "ridge_alpha_boundary": bool(final_alpha in (alpha_min, alpha_max)),
            "ridge_solver": "cholesky",
            "selection_metric": "mean_within_animal_mean_absolute_control_error",
        }
    )
    clock = TrainedClock(estimator=final_estimator, features=list(expression.index.astype(str)))
    return ReferenceClockRun(
        clock=clock,
        primary_predictions=primary,
        all_predictions=all_predictions,
        metrics=metrics,
        fold_audit=pd.DataFrame(fold_rows),
        hyperparameter_audit=pd.DataFrame(hyper_rows),
    )
