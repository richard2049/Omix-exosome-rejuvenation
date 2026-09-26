from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rejuvenation import compute_plasma_protein_contrasts
from src.run_pipeline import load_plasma_proteomics_csv


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = [
        "FY_1", "FY_2", "MY_1", "MY_2",
        "FV_1", "FV_2", "MV_1", "MV_2",
        "FWT_1", "FWT_2", "MWT_1", "MWT_2",
        "FGES_1", "FGES_2", "MGES_1", "MGES_2",
    ]
    groups = ["Y"] * 4 + ["V"] * 4 + ["WT"] * 4 + ["GES"] * 4
    sexes = ["F", "F", "M", "M"] * 4
    meta = pd.DataFrame({"sample_id": samples, "group": groups, "sex": sexes})
    expr = pd.DataFrame(
        [
            [1.0] * 4 + [1.0] * 4 + [2.0] * 4 + [4.0] * 4,
            [8.0] * 4 + [4.0] * 4 + [4.0] * 4 + [2.0] * 4,
        ],
        index=["P00001", "P00002"],
        columns=samples,
    )
    annotations = pd.DataFrame(
        {
            "feature_id": ["P00001", "P00002"],
            "protein_accession": ["P00001", "P00002"],
            # Duplicate gene symbols must not collapse distinct protein rows.
            "gene_name": ["DUP", "DUP"],
        }
    )
    return expr, meta, annotations


def test_plasma_contrasts_use_prespecified_groups_and_preserve_accessions():
    expr, meta, annotations = _fixture()
    result = compute_plasma_protein_contrasts(
        expr,
        meta,
        feature_annotations=annotations,
        min_per_group=3,
        n_bootstrap=20,
        stability_top_k=2,
        random_state=7,
    )

    assert set(result["contrast"]) == {"GES_vs_V", "WT_vs_V", "GES_vs_WT"}
    assert result["feature_id"].nunique() == 2
    assert result["protein_accession"].nunique() == 2
    assert set(result["gene_name"]) == {"DUP"}
    assert not {"spearman_r", "qval", "pro-aging", "pro-rejuvenation"}.intersection(result.columns)

    primary = result.loc[
        result["contrast"].eq("GES_vs_V") & result["feature_id"].eq("P00001")
    ].iloc[0]
    assert bool(primary["is_primary"])
    assert primary["log2_fold_change"] == pytest.approx(2.0)
    assert int(primary["n_group_a"]) == 4
    assert int(primary["n_group_b"]) == 4
    assert primary["covariates_used"] == "sex"
    assert primary["claim_class"] == "observed_association"
    assert primary["source_scale_status"] == "SOURCE_VERIFICATION_PENDING"


def test_young_values_do_not_change_old_group_treatment_contrasts():
    expr, meta, annotations = _fixture()
    baseline = compute_plasma_protein_contrasts(
        expr,
        meta,
        feature_annotations=annotations,
        n_bootstrap=0,
    )
    changed = expr.copy()
    changed.loc[:, meta.loc[meta["group"].eq("Y"), "sample_id"]] *= 1000.0
    rerun = compute_plasma_protein_contrasts(
        changed,
        meta,
        feature_annotations=annotations,
        n_bootstrap=0,
    )

    keys = ["feature_id", "contrast"]
    left = baseline.set_index(keys)["log2_fold_change"].sort_index()
    right = rerun.set_index(keys)["log2_fold_change"].sort_index()
    np.testing.assert_allclose(left.to_numpy(), right.to_numpy(), atol=1e-12)


def test_plasma_contrasts_reject_nonpositive_values_before_log2():
    expr, meta, annotations = _fixture()
    expr.iloc[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        compute_plasma_protein_contrasts(
            expr,
            meta,
            feature_annotations=annotations,
            n_bootstrap=0,
        )


def test_plasma_loader_uses_unique_accessions_and_retains_duplicate_gene_symbols(tmp_path):
    path = tmp_path / "plasma.csv"
    pd.DataFrame(
        {
            "Protein accession": ["P00001", "P00002"],
            "Gene name": ["DUP", "DUP"],
            "FV_1": [1.0, 2.0],
            "FGES_1": [4.0, 3.0],
        }
    ).to_csv(path, index=False)

    matrix, annotations = load_plasma_proteomics_csv(path, return_annotations=True)

    assert matrix.index.tolist() == ["P00001", "P00002"]
    assert matrix.index.is_unique
    assert annotations["gene_name"].tolist() == ["DUP", "DUP"]
