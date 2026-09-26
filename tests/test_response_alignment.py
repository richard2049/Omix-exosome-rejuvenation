from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attribution import compute_cross_species_response_alignment
from src.effect_sizes import hedges_g


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mouse_tissue": ["kidney", "liver", "lung", "brain"],
            "primate_tissue": ["Kidney", "Liver", "Lung", "Hippocampus"],
            "compatibility_tier": [
                "organ_compatible",
                "organ_compatible",
                "organ_compatible",
                "weak_context",
            ],
            "include_in_primary": [True, True, True, False],
            "source_status": ["curated"] * 4,
            "mapping_note": ["same organ"] * 3 + ["not anatomically equivalent"],
        }
    )


def _synthetic_outcomes() -> tuple[pd.DataFrame, pd.DataFrame]:
    primate_rows = []
    mouse_rows = []
    tissue_shifts = {"Kidney": -2.0, "Liver": -1.2, "Lung": -0.6, "Hippocampus": 1.5}
    mouse_shifts = {"kidney": -1.7, "liver": -1.0, "lung": -0.5, "brain": -1.5}
    noise = np.array([-0.30, -0.15, -0.05, 0.05, 0.15, 0.30])
    for tissue, shift in tissue_shifts.items():
        for group, offset in (("O_V", 0.0), ("O_GES", shift)):
            for index, residual in enumerate(noise):
                primate_rows.append(
                    {
                        "animal_id": f"{group}_{index}",
                        "group": group,
                        "sex": "F" if index % 2 == 0 else "M",
                        "tissue": tissue,
                        "delta_age": offset + residual,
                    }
                )
    for tissue, shift in mouse_shifts.items():
        for arm, offset in (("Veh", 0.0), ("GES", shift)):
            for index, residual in enumerate(noise):
                mouse_rows.append(
                    {
                        "sample_id": f"{tissue}_{arm}_{index}",
                        "arm": arm,
                        "sex": "F" if index % 2 == 0 else "M",
                        "tissue": tissue,
                        "predicted_age_mouse": 20.0 + offset + residual,
                    }
                )
    return pd.DataFrame(primate_rows), pd.DataFrame(mouse_rows)


def test_hedges_g_matches_known_small_sample_value():
    assert hedges_g([2.0, 3.0, 4.0], [0.0, 1.0, 2.0]) == pytest.approx(1.6)


def test_response_alignment_is_reproducible_and_excludes_weak_context():
    primate, mouse = _synthetic_outcomes()
    kwargs = dict(
        min_common_tissues=3,
        n_bootstrap=40,
        n_permutations=40,
        random_state=17,
    )
    by_tissue, summary = compute_cross_species_response_alignment(
        primate, mouse, _mapping(), **kwargs
    )
    _, repeated = compute_cross_species_response_alignment(
        primate, mouse, _mapping(), **kwargs
    )
    row = summary.iloc[0]
    repeated_row = repeated.iloc[0]

    assert set(by_tissue["primate_tissue"]) == {"Kidney", "Liver", "Lung"}
    assert bool(row["estimable"])
    assert int(row["n_common_tissues"]) == 3
    assert row["cosine_similarity"] > 0.9
    assert row["aligned_response_coefficient"] == pytest.approx(
        row["cosine_similarity"] * row["relative_response_norm"]
    )
    assert row["cosine_similarity"] == pytest.approx(repeated_row["cosine_similarity"])
    assert row["cosine_similarity_ci_low"] == pytest.approx(
        repeated_row["cosine_similarity_ci_low"]
    )
    assert int(row["n_bootstrap_valid"]) > 0
    assert "not an exosome-attributable fraction" in row["interpretation"]


def test_response_alignment_is_explicitly_unestimable_with_too_few_pairs():
    primate, mouse = _synthetic_outcomes()
    mapping = _mapping().loc[_mapping()["mouse_tissue"].isin(["kidney", "liver"])]
    by_tissue, summary = compute_cross_species_response_alignment(
        primate,
        mouse,
        mapping,
        min_common_tissues=3,
        n_bootstrap=0,
        n_permutations=0,
    )

    assert len(by_tissue) == 2
    assert not bool(summary.iloc[0]["estimable"])
    assert "Only 2" in summary.iloc[0]["reason"]


def test_response_alignment_unavailable_inputs_keep_output_contract():
    by_tissue, summary = compute_cross_species_response_alignment(
        pd.DataFrame(),
        pd.DataFrame(),
        _mapping(),
        n_bootstrap=10,
        n_permutations=12,
    )

    assert not bool(summary.iloc[0]["estimable"])
    assert {
        "mouse_tissue",
        "primate_tissue",
        "macaque_standardized_effect",
        "mouse_standardized_effect",
    }.issubset(by_tissue.columns)
    assert {
        "cosine_similarity",
        "relative_response_norm",
        "aligned_response_coefficient",
    }.issubset(summary.columns)
    assert int(summary.iloc[0]["n_bootstrap_requested"]) == 10
    assert int(summary.iloc[0]["n_permutations_requested"]) == 12


def test_response_alignment_rejects_tissue_pseudoreplicates():
    primate, mouse = _synthetic_outcomes()
    primate = pd.concat([primate, primate.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per animal and tissue"):
        compute_cross_species_response_alignment(
            primate,
            mouse,
            _mapping(),
            n_bootstrap=0,
            n_permutations=0,
        )
