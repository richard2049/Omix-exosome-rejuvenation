from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import log2_cpm_counts
from src.reference_clock import (
    apply_exact_sample_exclusions,
    assign_frozen_animal_folds,
    frozen_fold_assignment_digest,
    train_reference_transcriptomic_clock,
)


REFERENCE_GROUPS = ["Y_C", "M_C", "O_C", "O_V"]
PUBLIC_FOLD_SHA256 = "666731461817ec5c9e9b9cb3007a62ee328b61a805f0caab2779fc2663bcf98e"


def _synthetic_clock_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260924)
    rows: list[dict[str, object]] = []
    columns: dict[str, np.ndarray] = {}
    groups = [
        "Y_C", "Y_C", "M_C", "M_C", "O_C", "O_C", "O_V", "O_V",
        "O_WT", "O_WT", "O_GES", "O_GES",
    ]
    ages = [4, 5, 9, 11, 16, 18, 20, 22, 20, 21, 20, 22]
    for animal_number, (group, age) in enumerate(zip(groups, ages), start=1):
        animal_id = f"A{animal_number:02d}"
        for tissue_number, tissue in enumerate(("Liver", "Heart"), start=1):
            sample_id = f"{animal_id}_{tissue}"
            signal = np.array(
                [
                    100 + age * 8 + tissue_number,
                    600 - age * 5 + tissue_number,
                    40 + animal_number,
                    10 + tissue_number,
                    80 + (age % 3) * 7,
                    25 + animal_number * 2,
                ],
                dtype=float,
            )
            columns[sample_id] = signal + rng.integers(0, 4, size=len(signal))
            rows.append(
                {
                    "sample_id": sample_id,
                    "animal_id": animal_id,
                    "group": group,
                    "age": float(age),
                    "tissue": tissue,
                }
            )
    counts = pd.DataFrame(columns, index=[f"anonymous_{i}" for i in range(6)])
    expression = log2_cpm_counts(counts)
    return expression, pd.DataFrame(rows)


def test_log2_cpm_is_samplewise_and_preserves_provenance():
    counts = pd.DataFrame(
        {"s1": [1.0, 3.0], "s2": [2.0, 2.0]},
        index=["f1", "f2"],
    )
    counts.attrs["input_audit"] = {"source": "fixture"}

    observed = log2_cpm_counts(counts)
    expected = np.log2(1.0 + counts.to_numpy() / np.array([[4.0, 4.0]]) * 1_000_000.0)

    np.testing.assert_allclose(observed.to_numpy(), expected)
    assert observed.attrs["representation"] == "log2_cpm"
    assert observed.attrs["input_audit"] == {"source": "fixture"}


@pytest.mark.parametrize(
    "values, message",
    [
        ([[0.0, 1.0], [0.0, 2.0]], "positive total"),
        ([[-1.0, 1.0], [2.0, 2.0]], "non-finite or negative"),
        ([[np.nan, 1.0], [2.0, 2.0]], "non-finite or negative"),
    ],
)
def test_log2_cpm_rejects_invalid_samples(values, message):
    counts = pd.DataFrame(values, index=["f1", "f2"], columns=["s1", "s2"])
    with pytest.raises(ValueError, match=message):
        log2_cpm_counts(counts)


def test_exact_exclusion_is_traced_and_does_not_mutate_source():
    expression, metadata = _synthetic_clock_data()
    original_columns = expression.columns.tolist()
    excluded = metadata.iloc[-1]

    retained_expression, retained_metadata, ledger = apply_exact_sample_exclusions(
        expression,
        metadata,
        [str(excluded["sample_id"])],
    )

    assert expression.columns.tolist() == original_columns
    assert len(retained_metadata) == len(metadata) - 1
    assert retained_expression.shape[1] == expression.shape[1] - 1
    assert ledger.loc[0, "sample_id"] == excluded["sample_id"]
    assert ledger.loc[0, "reason"] == "unresolved_declared_animal_alias"


def test_exact_exclusion_fails_if_approved_sample_is_absent():
    expression, metadata = _synthetic_clock_data()
    with pytest.raises(ValueError, match="do not match the current inputs"):
        apply_exact_sample_exclusions(expression, metadata, ["absent_sample"])


def test_frozen_fold_map_is_animal_level_and_complete():
    _, metadata = _synthetic_clock_data()
    animals = metadata["animal_id"].drop_duplicates().tolist()
    assignments = pd.DataFrame(
        {
            "animal_id": animals,
            "legacy": [index % 3 for index in range(len(animals))],
        }
    )

    folds = assign_frozen_animal_folds(
        metadata,
        assignments,
        split="legacy",
        n_splits=3,
    )

    observed = pd.DataFrame({"animal_id": metadata["animal_id"], "fold": folds})
    assert observed.groupby("animal_id")["fold"].nunique().eq(1).all()
    assert set(folds) == {0, 1, 2}


def test_frozen_fold_map_rejects_changed_animal_set():
    _, metadata = _synthetic_clock_data()
    assignments = pd.DataFrame(
        {"animal_id": metadata["animal_id"].drop_duplicates().iloc[:-1], "legacy": 0}
    )
    with pytest.raises(ValueError, match="animal set differs"):
        assign_frozen_animal_folds(metadata, assignments, split="legacy", n_splits=3)


def test_frozen_fold_digest_is_order_stable_but_content_sensitive():
    assignments = pd.DataFrame(
        {
            "animal_id": ["B", "A"],
            "legacy": [0, 1],
            "seed_17": [1, 0],
            "seed_29": [0, 1],
            "seed_43": [1, 0],
        }
    )
    reordered = assignments.iloc[::-1].copy()
    changed = assignments.copy()
    changed.loc[0, "legacy"] = 1

    assert frozen_fold_assignment_digest(assignments) == frozen_fold_assignment_digest(reordered)
    assert frozen_fold_assignment_digest(assignments) != frozen_fold_assignment_digest(changed)


def test_public_frozen_fold_map_contract():
    path = Path(__file__).resolve().parents[1] / "config" / "primate_clock_folds.csv"
    assignments = pd.read_csv(path)

    assert assignments.shape == (60, 5)
    assert assignments["animal_id"].is_unique
    assert frozen_fold_assignment_digest(assignments) == PUBLIC_FOLD_SHA256
    for column in ("legacy", "seed_17", "seed_29", "seed_43"):
        assert set(assignments[column]) == {0, 1, 2, 3, 4}


def test_reference_clock_has_complete_predictions_and_no_treated_training():
    expression, metadata = _synthetic_clock_data()

    result = train_reference_transcriptomic_clock(
        expression,
        metadata,
        reference_groups=REFERENCE_GROUPS,
        ridge_alphas=[100.0, 1.0, 0.0001],
        outer_folds=3,
        inner_folds=2,
        diagnostic_seeds=[17],
    )

    assert set(result.primary_predictions["sample_id"]) == set(metadata["sample_id"])
    assert result.primary_predictions["predicted_age"].notna().all()
    assert set(result.all_predictions["split"]) == {"legacy", "seed_17"}
    assert len(result.all_predictions) == 2 * len(metadata)
    assert result.fold_audit["n_train_treated"].eq(0).all()
    assert result.fold_audit["animal_overlap"].eq(0).all()
    assert result.fold_audit["alpha_boundary"].isin([True, False]).all()
    assert result.metrics["n_samples"] == 16.0
    assert result.metrics["n_animals"] == 8.0
    assert result.metrics["n_evaluation_samples"] == 24.0
    assert result.metrics["training_scope"] == "controls_only_global"
    assert result.metrics["representation"] == "log2_cpm"
    assert len(result.hyperparameter_audit) == (2 * 3 + 1) * 3


def test_reference_clock_rejects_unapproved_representation():
    expression, metadata = _synthetic_clock_data()
    expression.attrs["representation"] = "log1p_counts"
    with pytest.raises(ValueError, match="approved log2_cpm"):
        train_reference_transcriptomic_clock(
            expression,
            metadata,
            reference_groups=REFERENCE_GROUPS,
            ridge_alphas=[1.0],
            outer_folds=3,
            inner_folds=2,
            diagnostic_seeds=[],
        )


def test_treatment_values_and_ages_cannot_change_control_predictions():
    expression, metadata = _synthetic_clock_data()
    baseline = train_reference_transcriptomic_clock(
        expression,
        metadata,
        reference_groups=REFERENCE_GROUPS,
        ridge_alphas=[1.0],
        outer_folds=3,
        inner_folds=2,
        diagnostic_seeds=[],
    )

    changed_expression = expression.copy()
    treated = metadata.loc[~metadata["group"].isin(REFERENCE_GROUPS), "sample_id"].tolist()
    changed_expression.loc[:, treated] = changed_expression.loc[:, treated] * 1000.0 + 500.0
    changed_expression.attrs.update(expression.attrs)
    changed_metadata = metadata.copy()
    changed_metadata.loc[~changed_metadata["group"].isin(REFERENCE_GROUPS), "age"] += 50.0
    changed = train_reference_transcriptomic_clock(
        changed_expression,
        changed_metadata,
        reference_groups=REFERENCE_GROUPS,
        ridge_alphas=[1.0],
        outer_folds=3,
        inner_folds=2,
        diagnostic_seeds=[],
    )

    controls = metadata.loc[metadata["group"].isin(REFERENCE_GROUPS), "sample_id"]
    first = baseline.primary_predictions.set_index("sample_id").loc[controls, "predicted_age"]
    second = changed.primary_predictions.set_index("sample_id").loc[controls, "predicted_age"]
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-12)


def test_primary_predictions_match_independent_fixed_alpha_refits():
    expression, metadata = _synthetic_clock_data()
    animals = metadata["animal_id"].drop_duplicates().tolist()
    frozen = pd.DataFrame(
        {
            "animal_id": animals,
            "legacy": [index % 3 for index in range(len(animals))],
        }
    )
    result = train_reference_transcriptomic_clock(
        expression,
        metadata,
        reference_groups=REFERENCE_GROUPS,
        ridge_alphas=[1.0],
        outer_folds=3,
        inner_folds=2,
        diagnostic_seeds=[],
        frozen_fold_assignments=frozen,
    )

    x = expression.loc[:, metadata["sample_id"]].T.to_numpy(dtype=np.float64)
    fold_by_animal = frozen.set_index("animal_id")["legacy"]
    folds = metadata["animal_id"].map(fold_by_animal).to_numpy()
    reference = metadata["group"].isin(REFERENCE_GROUPS).to_numpy()
    expected = np.full(len(metadata), np.nan)
    for fold in range(3):
        train = (folds != fold) & reference
        test = folds == fold
        independent = make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            Ridge(alpha=1.0, solver="cholesky"),
        )
        independent.fit(x[train], metadata.loc[train, "age"])
        expected[test] = independent.predict(x[test])

    observed = (
        result.primary_predictions.set_index("sample_id")
        .loc[metadata["sample_id"], "predicted_age"]
        .to_numpy()
    )
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
