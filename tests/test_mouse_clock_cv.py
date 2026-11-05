"""Guardrails for the mouse tissue clock cross-validation unit.

The experimental unit in OMIX009283 is the mouse. A tissue clock evaluated
with plain KFold over samples that contain replicates of the same mouse would
report over-optimistic clock_cv_* metrics that feed evidence-level-1 effect
tables. These tests pin the contract:

- a metadata animal_id column is used when explicitly requested via cv_group_col;
- the chosen strategy must be declared in the output table;
- data without animal identity must fall back honestly to sample-level KFold
  instead of pretending grouped evaluation.

Grouping is opt-in. These tests do not validate biological identity or change
the experimental unit of treatment-effect bootstrap or permutation inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.attribution import compute_mouse_exosome_tissue_effects


def _mouse_fixture(
    *,
    with_animal_id: bool,
    n_animals_per_arm: int = 6,
    replicates_per_animal: int = 2,
    n_genes: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two tissues would need more rows; one tissue keeps the fixture small.

    The clock needs >=20 reference samples and >=4 unique reference ages, so
    the baseline arm provides replicated animals across a range of ages.
    """
    rng = np.random.default_rng(7)
    rows = []
    sample_ids = []

    # train_transcriptomic_clock requires >=20 valid-age reference samples.
    arms = [("Baseline", 10)] + [
        (arm, n_animals_per_arm) for arm in ("Veh", "GES", "WT")
    ]
    baseline_ages = np.linspace(8.0, 26.0, 10)

    for arm, n_animals in arms:
        for animal_index in range(n_animals):
            age = (
                float(baseline_ages[animal_index])
                if arm == "Baseline"
                else 20.0 + (animal_index % 3)
            )
            animal_key = f"{arm}_{animal_index}"
            for rep in range(replicates_per_animal):
                sample_id = f"{animal_key}_rep{rep}"
                sample_ids.append(sample_id)
                row = {
                    "sample_id": sample_id,
                    "tissue": "Liver",
                    "arm": arm,
                    "age": age,
                    "sex": "F" if animal_index % 2 == 0 else "M",
                    "replicate": rep,
                }
                if with_animal_id:
                    row["animal_id"] = animal_key
                rows.append(row)

    meta = pd.DataFrame(rows)

    expression = pd.DataFrame(
        rng.normal(size=(n_genes, len(sample_ids))),
        index=[f"gene_{i:03d}" for i in range(n_genes)],
        columns=sample_ids,
    )
    # Age-informative signal so the clock trains on something real.
    age_by_sample = meta.set_index("sample_id")["age"]
    expression.loc["gene_000"] = age_by_sample.to_numpy(dtype=float)
    expression.loc["gene_001"] = 0.5 * age_by_sample.to_numpy(dtype=float) + rng.normal(
        scale=0.1, size=len(sample_ids)
    )
    return expression, meta


def _run(expr: pd.DataFrame, meta: pd.DataFrame, cv_group_col: str) -> pd.DataFrame:
    return compute_mouse_exosome_tissue_effects(
        expr,
        meta,
        reference_arms=("Baseline",),
        contrasts=(("GES", "Veh"),),
        min_reference_samples=20,
        min_reference_ages=4,
        min_samples_per_group=3,
        n_bootstrap=50,
        n_permutations=49,
        random_state=5,
        cv_group_col=cv_group_col,
    )


def test_animal_identity_column_yields_grouped_clock_cv():
    expr, meta = _mouse_fixture(with_animal_id=True)
    result = _run(expr, meta, cv_group_col="animal_id")

    row = result.loc[result["contrast"].eq("GES_vs_Veh")].iloc[0]
    assert bool(row["estimable"]) is True
    assert row["clock_cv_strategy"] == "GroupKFold(animal_id)"
    assert row["clock_cv_n_groups"] == 10.0
    assert np.isfinite(row["clock_cv_mae"])
    assert row["clock_cv_unit"] == "animal_id"


def test_missing_animal_identity_falls_back_honestly_to_sample_kfold():
    expr, meta = _mouse_fixture(with_animal_id=False)
    result = _run(expr, meta, cv_group_col="animal_id")

    row = result.loc[result["contrast"].eq("GES_vs_Veh")].iloc[0]
    assert bool(row["estimable"]) is True
    assert row["clock_cv_strategy"] == "KFold"
    assert row["clock_cv_unit"] == "sample"
