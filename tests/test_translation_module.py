"""Tests for the translational/target-prioritization helpers.

The acceptance contract for the FDR correction is the independent reference
implementation ``statsmodels.stats.multitest.multipletests(method="fdr_bh")``,
not this repository's own code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from src.translation_module import (
    benjamini_hochberg,
    differential_expression_ttest,
    get_top_genes,
    load_gmt,
    ora_hypergeom,
    rank_plasma_biomarkers,
    safe_ttest,
)


# -----------------------------
# FDR correction
# -----------------------------


def test_benjamini_hochberg_matches_statsmodels_on_clean_input():
    p = np.array([0.001, 0.02, 0.04, 0.5])
    expected = multipletests(p, method="fdr_bh")[1]
    np.testing.assert_allclose(benjamini_hochberg(p), expected, atol=1e-12)


def test_benjamini_hochberg_nan_does_not_contaminate_finite_pvalues():
    p = np.array([0.001, 0.02, np.nan, 0.04, 0.5])
    q = benjamini_hochberg(p)
    finite = np.isfinite(p)
    # Finite entries are adjusted exactly as if the NaN row did not exist.
    expected = multipletests(p[finite], method="fdr_bh")[1]
    np.testing.assert_allclose(q[finite], expected, atol=1e-12)
    assert np.isnan(q[~finite]).all()


def test_benjamini_hochberg_nonfinite_values_and_ties_match_statsmodels():
    p = np.array([0.04, np.inf, 0.001, np.nan, 0.04, -np.inf, 0.5])
    finite = np.isfinite(p)
    q = benjamini_hochberg(p)
    expected = multipletests(p[finite], method="fdr_bh")[1]
    np.testing.assert_allclose(q[finite], expected, atol=1e-12)
    assert np.isnan(q[~finite]).all()


def test_benjamini_hochberg_edge_cases():
    assert benjamini_hochberg(np.array([], dtype=float)).size == 0
    np.testing.assert_allclose(benjamini_hochberg(np.array([0.03])), [0.03])
    assert np.isnan(benjamini_hochberg(np.array([np.nan, np.nan]))).all()


def test_safe_ttest_clear_difference_is_significant():
    p = safe_ttest(np.array([3.0, 3.1, 2.9, 3.0]), np.array([0.0, 0.1, 0.0, -0.1]))
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
    assert p < 1e-3


# -----------------------------
# Lightweight DE
# -----------------------------


def _de_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = [f"S{i:02d}" for i in range(8)]
    meta = pd.DataFrame(
        {
            "sample_id": samples,
            "group": ["TREATED"] * 4 + ["V"] * 2 + ["WT"] * 2,
            "tissue": ["Liver"] * 8,
        }
    )
    rng = np.random.default_rng(11)
    expr = pd.DataFrame(
        rng.normal(scale=0.1, size=(22, 8)),
        index=[f"gene_{i:02d}" for i in range(22)],
        columns=samples,
    )
    expr.loc["UP_GENE"] = [3.0, 3.1, 2.9, 3.0, 0.0, 0.1, 0.0, -0.1]
    expr.loc["DOWN_GENE"] = [-3.0, -3.1, -2.9, -3.0, 0.0, -0.1, 0.0, 0.1]
    return expr, meta


def test_differential_expression_ttest_recovers_direction_and_valid_adjustment():
    expr, meta = _de_inputs()
    de = differential_expression_ttest(
        expr,
        meta,
        treated_label="TREATED",
        control_labels=["V", "WT"],
        tissue="Liver",
    )

    assert not de.empty
    by_gene = de.set_index("gene")
    assert by_gene.loc["UP_GENE", "logFC"] == pytest.approx(3.0, abs=1e-9)
    assert by_gene.loc["DOWN_GENE", "logFC"] == pytest.approx(-3.0, abs=1e-9)
    assert by_gene.loc["UP_GENE", "n_treated"] == 4
    assert by_gene.loc["UP_GENE", "n_control"] == 4
    assert de["pval"].between(0.0, 1.0).all()
    assert de["qval"].between(0.0, 1.0).all()
    # BH q-values are non-decreasing when sorted by p-value.
    ordered = de.sort_values("pval")
    assert ordered["qval"].is_monotonic_increasing


def test_differential_expression_ttest_respects_min_samples_and_group_column():
    expr, meta = _de_inputs()
    assert differential_expression_ttest(
        expr, meta, "TREATED", ["V", "WT"], min_samples_per_group=10
    ).empty
    assert differential_expression_ttest(
        expr, meta.drop(columns=["group"]), "TREATED", ["V", "WT"]
    ).empty


def test_get_top_genes_direction_and_qval_contract():
    de = pd.DataFrame(
        {
            "gene": ["a", "b", "c", "d"],
            "logFC": [2.0, -3.0, 1.0, -0.5],
            "qval": [0.01, 0.01, 0.5, 0.01],
        }
    )
    assert get_top_genes(de, n=10, direction="up", qval_max=0.2) == {"a"}
    assert get_top_genes(de, n=10, direction="down", qval_max=0.2) == {"b", "d"}
    assert get_top_genes(de, n=10, direction="any", qval_max=0.01) == {"a", "b", "d"}
    assert get_top_genes(pd.DataFrame(), n=5) == set()


# -----------------------------
# GMT + ORA
# -----------------------------


def test_load_gmt_parses_minimal_file(tmp_path):
    gmt = tmp_path / "sets.gmt"
    gmt.write_text("setA\tdesc\tg1\tg2\nsetB\tdesc\tg3\n", encoding="utf-8")
    assert load_gmt(gmt) == {"setA": {"g1", "g2"}, "setB": {"g3"}}


def test_ora_hypergeom_enriches_overlapping_set():
    gene_sets = {
        "path_A": {"g1", "g2", "g3", "g4"},
        "path_B": {"g5", "g6", "g7", "g8"},
    }
    background = {f"g{i}" for i in range(1, 21)}
    out = ora_hypergeom({"g1", "g2", "g3"}, background, gene_sets, min_overlap=2)

    assert not out.empty
    top = out.iloc[0]
    assert top["pathway"] == "path_A"
    assert int(top["overlap"]) == 3
    assert out["pval"].between(0.0, 1.0).all()
    assert out["qval"].between(0.0, 1.0).all()


# -----------------------------
# Plasma biomarker ranking
# -----------------------------


def _biomarker_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = [f"S{i}" for i in range(6)]
    plasma_meta = pd.DataFrame(
        {"sample_id": samples, "group": ["TREATED"] * 3 + ["V"] * 3}
    )
    outcome_meta = pd.DataFrame(
        {
            "sample_id": samples,
            "rejuvenation_score": [3.0, 2.5, 3.5, -1.0, -0.5, -1.5],
        }
    )
    plasma = pd.DataFrame(
        {
            "S0": [3.0, 0.1],
            "S1": [2.5, 0.2],
            "S2": [3.5, 0.0],
            "S3": [-1.0, 0.1],
            "S4": [-0.5, 0.3],
            "S5": [-1.5, 0.2],
        },
        index=["TRACKS_OUTCOME", "FLAT"],
    )
    return plasma, plasma_meta, outcome_meta


def test_rank_plasma_biomarkers_ranks_outcome_tracking_protein_first():
    plasma, plasma_meta, outcome_meta = _biomarker_inputs()
    out = rank_plasma_biomarkers(
        plasma,
        plasma_meta,
        outcome_meta,
        outcome_col="rejuvenation_score",
        treated_label="TREATED",
        control_labels=["V"],
    )

    assert not out.empty
    top = out.iloc[0]
    assert top["protein"] == "TRACKS_OUTCOME"
    assert top["spearman_r"] == pytest.approx(1.0, abs=1e-9)
    # treated mean 3.0 minus control mean -1.0
    assert top["delta_treated_control"] == pytest.approx(4.0, abs=1e-9)
    assert out["qval"].between(0.0, 1.0).all()


def test_rank_plasma_biomarkers_requires_minimum_shared_samples():
    plasma, plasma_meta, outcome_meta = _biomarker_inputs()
    out = rank_plasma_biomarkers(
        plasma.iloc[:, :2],
        plasma_meta.iloc[:2],
        outcome_meta.iloc[:2],
        outcome_col="rejuvenation_score",
        treated_label="TREATED",
        control_labels=["V"],
    )
    assert out.empty
