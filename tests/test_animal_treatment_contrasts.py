from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rejuvenation import (
    summarize_animal_treatment_contrasts_by_tissue,
    summarize_clustered_treatment_sensitivity,
    summarize_global_animal_treatment_contrasts,
)
from src.report_figures import plot_tissue_rejuvenation_forest


CONTRASTS = [
    ("SRC_vs_vehicle", "O_GES", "O_V"),
    ("WTC_vs_vehicle", "O_WT", "O_V"),
    ("SRC_vs_WTC", "O_GES", "O_WT"),
]


def _balanced_fixture() -> pd.DataFrame:
    rows = []
    group_shift = {"O_V": 0.0, "O_WT": -1.0, "O_GES": -2.0}
    tissue_shift = {"Heart": 0.0, "Liver": 1.0, "Kidney": -0.5}
    for group_index, group in enumerate(["O_V", "O_WT", "O_GES"]):
        for animal_index in range(6):
            animal_id = f"{group}_{animal_index + 1}"
            sex = "F" if animal_index < 3 else "M"
            age = 20.0 + (animal_index % 3)
            animal_offset = (animal_index - 2.5) * 0.1
            for tissue, tissue_effect in tissue_shift.items():
                rows.append(
                    {
                        "sample_id": f"{animal_id}_{tissue}",
                        "animal_id": animal_id,
                        "group": group,
                        "sex": sex,
                        "age": age,
                        "tissue": tissue,
                        "delta_age": group_shift[group] + tissue_effect + animal_offset,
                    }
                )
    # An age-reference cohort must not enter intervention contrasts.
    rows.append(
        {
            "sample_id": "Y_1_Heart",
            "animal_id": "Y_1",
            "group": "Y_C",
            "sex": "F",
            "age": 5.0,
            "tissue": "Heart",
            "delta_age": -1000.0,
        }
    )
    return pd.DataFrame(rows)


def test_tissue_contrasts_use_animals_and_prespecified_vehicle_comparator():
    result = summarize_animal_treatment_contrasts_by_tissue(
        _balanced_fixture(),
        tissue_col="tissue",
        group_col="group",
        animal_col="animal_id",
        sex_col="sex",
        contrasts=CONTRASTS,
        min_per_group=3,
        n_bootstrap=200,
        n_permutations=199,
        random_state=17,
    )

    assert len(result) == 9
    primary = result.loc[result["contrast"].eq("SRC_vs_vehicle")]
    assert primary["is_primary"].all()
    assert set(primary["n_trt"]) == {6}
    assert set(primary["n_ctrl"]) == {6}
    np.testing.assert_allclose(primary["mean_effect"], -2.0, atol=1e-12)
    np.testing.assert_allclose(
        primary["primary_fwer_p_value"],
        np.minimum(1.0, primary["p_value"] * 3),
    )
    assert (result["holm_all_contrasts_p_value"] >= result["p_value"]).all()
    assert set(result["bootstrap_unit"]) == {"animal"}
    assert set(result["permutation_unit"]) == {"animal"}
    assert set(result["ci_type"]) == {"pointwise_percentile_bootstrap"}


def test_tissue_contrast_results_are_invariant_to_input_row_order():
    frame = _balanced_fixture()
    kwargs = dict(
        tissue_col="tissue",
        group_col="group",
        animal_col="animal_id",
        sex_col="sex",
        contrasts=CONTRASTS,
        min_per_group=3,
        n_bootstrap=100,
        n_permutations=99,
        random_state=29,
    )
    left = summarize_animal_treatment_contrasts_by_tissue(frame, **kwargs)
    right = summarize_animal_treatment_contrasts_by_tissue(
        frame.sample(frac=1.0, random_state=123), **kwargs
    )
    columns = ["tissue", "contrast", "mean_effect", "ci_low", "ci_high", "p_value"]
    pd.testing.assert_frame_equal(
        left[columns].sort_values(["tissue", "contrast"]).reset_index(drop=True),
        right[columns].sort_values(["tissue", "contrast"]).reset_index(drop=True),
    )


def test_duplicate_animal_tissue_rows_fail_closed():
    frame = _balanced_fixture()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate animal-tissue"):
        summarize_animal_treatment_contrasts_by_tissue(
            frame,
            tissue_col="tissue",
            group_col="group",
            animal_col="animal_id",
            sex_col="sex",
            contrasts=CONTRASTS,
            n_bootstrap=20,
            n_permutations=19,
        )


def test_global_summary_gives_each_animal_equal_weight_before_resampling():
    frame = pd.DataFrame(
        [
            {"animal_id": "G1", "group": "O_GES", "sex": "F", "tissue": "A", "delta_age": 0.0},
            {"animal_id": "G1", "group": "O_GES", "sex": "F", "tissue": "B", "delta_age": 0.0},
            {"animal_id": "G1", "group": "O_GES", "sex": "F", "tissue": "C", "delta_age": 0.0},
            {"animal_id": "G2", "group": "O_GES", "sex": "M", "tissue": "A", "delta_age": 10.0},
            {"animal_id": "V1", "group": "O_V", "sex": "F", "tissue": "A", "delta_age": 0.0},
            {"animal_id": "V1", "group": "O_V", "sex": "F", "tissue": "B", "delta_age": 0.0},
            {"animal_id": "V2", "group": "O_V", "sex": "M", "tissue": "A", "delta_age": 0.0},
        ]
    )
    result = summarize_global_animal_treatment_contrasts(
        frame,
        tissue_col="tissue",
        group_col="group",
        animal_col="animal_id",
        sex_col="sex",
        contrasts=[CONTRASTS[0]],
        min_per_group=2,
        n_bootstrap=100,
        n_permutations=99,
        random_state=43,
    )
    assert len(result) == 1
    assert result.iloc[0]["mean_effect"] == pytest.approx(5.0)
    assert result.iloc[0]["n_trt"] == 2
    assert result.iloc[0]["n_ctrl"] == 2
    assert result.iloc[0]["n_tissues_per_animal_min"] == 1
    assert result.iloc[0]["n_tissues_per_animal_max"] == 3


def test_clustered_sensitivity_uses_animals_as_covariance_clusters():
    result = summarize_clustered_treatment_sensitivity(
        _balanced_fixture(),
        tissue_col="tissue",
        group_col="group",
        animal_col="animal_id",
        sex_col="sex",
        age_col="age",
        contrasts=CONTRASTS,
        min_animals_per_group=3,
    )
    assert len(result) == 3
    assert set(result["cluster_unit"]) == {"animal_id"}
    assert set(result["n_animals"]) == {12}
    assert set(result["n_tissues"]) == {3}
    primary = result.loc[result["contrast"].eq("SRC_vs_vehicle")].iloc[0]
    assert primary["mean_effect"] == pytest.approx(-2.0, abs=1e-10)
    assert primary["method"] == "ols_tissue_age_sex_animal_clustered"


def test_clustered_sensitivity_is_not_estimable_without_animal_identity():
    frame = _balanced_fixture()
    frame["animal_id"] = pd.NA

    result = summarize_clustered_treatment_sensitivity(
        frame,
        tissue_col="tissue",
        group_col="group",
        animal_col="animal_id",
        sex_col="sex",
        age_col="age",
        contrasts=CONTRASTS,
        min_animals_per_group=3,
    )

    assert result.empty


def test_tissue_figure_counts_only_the_primary_contrast(tmp_path):
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()
    pd.DataFrame(
        [
            {
                "tissue": "Heart",
                "contrast": "SRC_vs_vehicle",
                "is_primary": True,
                "mean_effect": -1.0,
                "ci_low": -2.0,
                "ci_high": 0.5,
                "n_used": 12,
                "estimable": True,
            },
            {
                "tissue": "Heart",
                "contrast": "WTC_vs_vehicle",
                "is_primary": False,
                "mean_effect": 3.0,
                "ci_low": 1.0,
                "ci_high": 4.0,
                "n_used": 12,
                "estimable": True,
            },
        ]
    ).to_csv(results_dir / "rejuvenation_by_tissue.csv", index=False)

    record = plot_tissue_rejuvenation_forest(results_dir, figures_dir)
    assert record.status == "ok"
    assert record.message.startswith("1 tissues plotted")
    assert record.path.exists()
