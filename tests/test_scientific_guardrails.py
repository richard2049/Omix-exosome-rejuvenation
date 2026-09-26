import warnings
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
import zipfile
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.exosome_effect import estimate_exosome_fraction_with_uncertainty
from src.clocks import train_transcriptomic_clock
from src.attribution import (
    build_mouse_exosome_metadata,
    compute_exosome_alignment_tables,
    compute_mouse_exosome_tissue_effects,
    load_cross_species_tissue_mapping,
)
from src.omix007582_audit import build_omix007582_sample_map_audit
from src.omix009284_audit import build_omix009284_audit
from src.omix009283_metadata import build_omix009283_metadata_table
from src.omix_io import load_omix_matrix
from src.preprocessing import build_proxy_rejuvenation_score
from src.plasma_axis import build_oriented_plasma_aging_axis, correlate_plasma_axis_with_delta_age
from src.report_figures import (
    _claim_limitation,
    generate_report_figures,
    plot_evidence_ladder,
    plot_portfolio_estimability_guardrail,
    plot_portfolio_multimodal_evidence,
)
from src.rejuvenation import annotate_effect_uncertainty
from src.linkage_audit import audit_primate_plasma_linkage, build_estimability_report
from src.viz import plot_mediation_effects_bar
from src.reason_codes import (
    AUTHOR_KEY_NOT_REQUIRED,
    CONFIG_DISABLED,
    INSUFFICIENT_LINKED_ANIMALS,
    OMIX007582_SENTRIX_SAMPLE_SHEET_MISSING,
    PLASMA_ANIMAL_LINKAGE_COLLISION,
    PLASMA_BULK_ANIMAL_LINKAGE_MISSING,
    PLASMA_LINKAGE_CONFIDENCE_MISSING,
    AUTHOR_KEY_OMIX007582_SENTRIX,
    AUTHOR_KEY_PLASMA_BULK_LINKAGE,
    AUTHOR_KEY_PLASMA_TO_ANIMAL,
)
from src.run_pipeline import (
    PROFILE_SPECS,
    STANDARD_RESULT_COLUMNS,
    _attach_primary_mediation_interval,
    _build_animal_level_mediation_table,
    _build_default_config,
    _build_mediation_stub,
    _causal_gate_reason,
    _causal_exosome_fraction_status,
    _ensure_standard_schema,
    _evidence_level,
    _map_plasma_sample_to_bulk_animal_id,
    build_plasma_metadata_from_columns,
)


def _workspace_tempdir() -> TemporaryDirectory:
    base = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    base.mkdir(exist_ok=True)
    return TemporaryDirectory(dir=base)


def test_unlinked_runs_do_not_pass_causal_gate():
    estimability = build_estimability_report(
        {
            "has_prim_animal_id_col": False,
            "has_plasma_animal_id_col": False,
            "n_overlap_animal_ids": 0,
        },
        min_samples_for_mediation=12,
    )
    reason = _causal_gate_reason(
        enable_mediation=True,
        enable_causal_decomposition=True,
        estimability_row=estimability,
    )
    assert reason is not None
    assert "animal_id" in reason.lower() or "overlap" in reason.lower()


def test_unresolved_samples_are_not_silently_linked():
    metadata = build_plasma_metadata_from_columns(["FV_2", "FY_1", "MX_3"])
    mapped = metadata.set_index("sample_id")

    assert mapped.loc["FV_2", "animal_id"] == "F-V-2"
    # Public naming establishes a candidate, not cross-modal animal identity.
    assert mapped.loc["FV_2", "animal_id_confidence"] == "inferred"
    for sample_id in ("FY_1", "MX_3"):
        assert pd.isna(mapped.loc[sample_id, "animal_id"])
        assert mapped.loc[sample_id, "animal_id_source"] == "unresolved"
        assert mapped.loc[sample_id, "animal_id_confidence"] == "low"

    animal_id, rule, confidence, reason = _map_plasma_sample_to_bulk_animal_id("not-a-sample")
    assert animal_id is None
    assert rule == "none"
    assert confidence == "low"
    assert reason


def test_linkage_collisions_fail_estimability():
    prim = pd.DataFrame(
        {
            "animal_id": ["A", "B", "C"],
            "group": ["O_GES", "O_V", "O_WT"],
            "sex": ["F", "F", "M"],
        }
    )
    plasma = pd.DataFrame(
        {
            "animal_id": ["A", "A", "B", "C"],
            "animal_id_confidence": ["high"] * 4,
            "group": ["GES", "GES", "V", "WT"],
            "sex": ["F", "F", "F", "M"],
        }
    )

    audit = audit_primate_plasma_linkage(prim, plasma)
    estimability = build_estimability_report(
        audit,
        min_samples_for_mediation=3,
        min_overlap_animals=3,
        min_treated_overlap=1,
        min_control_overlap=2,
    )

    assert audit["mapping_collision_count"] == 1
    assert estimability["tier"] == "partially_linked"
    assert estimability["can_do_mediation"] is False
    assert estimability["can_do_linked_decomposition"] is False
    assert estimability["reason_code"] == PLASMA_ANIMAL_LINKAGE_COLLISION
    assert "collision" in estimability["reason"].lower()


def test_partially_linked_data_do_not_run_causal_decomposition():
    estimability = build_estimability_report(
        {
            "has_prim_animal_id_col": True,
            "has_plasma_animal_id_col": True,
            "has_plasma_animal_id_confidence_col": True,
            "n_overlap_animal_ids_high_conf": 3,
            "n_overlap_treated_animals": 1,
            "n_overlap_control_animals": 2,
            "mapping_collision_count": 0,
        },
        min_samples_for_mediation=3,
        min_overlap_animals=4,
        min_treated_overlap=1,
        min_control_overlap=2,
    )

    reason = _causal_gate_reason(
        enable_mediation=True,
        enable_causal_decomposition=True,
        estimability_row=estimability,
    )
    assert estimability["tier"] == "partially_linked"
    assert estimability["can_do_linked_decomposition"] is False
    assert reason is not None


def test_mediation_requires_animal_level_rows():
    prim = pd.DataFrame(
        {
            "animal_id": ["A", "A", "B", "B"],
            "rejuvenation_score": [1.0, 3.0, 4.0, 8.0],
            "group_binary": [0, 0, 1, 1],
            "group": ["O_V", "O_V", "O_GES", "O_GES"],
        }
    )
    plasma = pd.DataFrame(
        {
            "animal_id": ["A", "B"],
            "plasma_state_score": [0.5, 1.5],
            "animal_id_confidence": ["high", "metadata_exact"],
        }
    )

    linked = _build_animal_level_mediation_table(
        prim,
        plasma,
        high_conf_values=("high", "metadata", "metadata_exact"),
    )

    assert len(linked) == 2
    assert linked["animal_id"].is_unique
    assert linked.set_index("animal_id").loc["A", "rejuvenation_score"] == 2.0
    assert linked.set_index("animal_id").loc["B", "rejuvenation_score"] == 6.0


def test_rejuvenation_score_positive_values_are_younger_like():
    meta = pd.DataFrame(
        {
            "sample_id": ["younger_like", "older_like"],
            "age": [10.0, 10.0],
            "predicted_age_cv": [8.0, 12.0],
        }
    )

    scored = build_proxy_rejuvenation_score(meta)

    assert scored.loc[0, "rejuvenation_score"] == 2.0
    assert scored.loc[1, "rejuvenation_score"] == -2.0


def test_transcriptomic_clock_uses_stable_float64_ridge_inputs():
    rng = np.random.default_rng(42)
    sample_ids = [f"S{i:02d}" for i in range(40)]
    base = rng.normal(size=(100, 40)).astype(np.float32)
    expression = pd.DataFrame(
        np.repeat(base, repeats=10, axis=0),
        index=[f"gene_{i:04d}" for i in range(1000)],
        columns=sample_ids,
    )
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "age": np.repeat(np.linspace(4.0, 20.0, 20), 2),
            "animal_id": np.repeat([f"A{i:02d}" for i in range(20)], 2),
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, predictions, metrics = train_transcriptomic_clock(
            expression,
            metadata,
            age_col="age",
            cv_group_col="animal_id",
        )

    assert metrics["input_dtype"] == "float64"
    assert int(metrics["n_features"]) == 1000
    assert predictions["predicted_age"].notna().all()
    assert not [item for item in caught if type(item.message).__name__ == "LinAlgWarning"]


def test_nonestimable_mediation_writes_structured_stub():
    stub = _build_mediation_stub(
        reason="Only 3 overlapping animals for mediation.",
        tier="partially_linked",
        n_overlap_animal_ids=3,
        n_used=3,
    )
    with _workspace_tempdir() as tmpdir:
        path = Path(tmpdir) / "mediation_summary.csv"
        stub.to_csv(path, index=False)
        saved = pd.read_csv(path)

    assert set(STANDARD_RESULT_COLUMNS).issubset(saved.columns)
    assert bool(saved.loc[0, "estimable"]) is False
    assert int(saved.loc[0, "evidence_level"]) == 0
    assert saved.loc[0, "tier"] == "partially_linked"
    assert int(saved.loc[0, "n_overlap_animal_ids"]) == 3
    assert saved.loc[0, "reason_code"] == INSUFFICIENT_LINKED_ANIMALS
    assert saved.loc[0, "missing_author_key"] == AUTHOR_KEY_PLASMA_BULK_LINKAGE


def test_mediation_diagnostic_accepts_current_and_historical_columns():
    canonical = pd.DataFrame(
        {
            "Total": [-0.6],
            "Total_CI": [(-1.0, 0.2)],
            "ADE": [-0.4],
            "ADE_CI": [(-0.8, -0.1)],
            "ACME": [-0.2],
            "ACME_CI": [(-0.6, 0.1)],
        }
    )
    normalized = _attach_primary_mediation_interval(canonical)
    assert normalized.loc[0, "ci_low"] == -1.0
    assert normalized.loc[0, "ci_high"] == 0.2

    historical = pd.DataFrame(
        {
            "total_effect": [-0.6],
            "total_ci_low": [-1.0],
            "total_ci_high": [0.2],
            "direct_effect": [-0.4],
            "direct_ci_low": [-0.8],
            "direct_ci_high": [-0.1],
            "indirect_effect": [-0.2],
            "indirect_ci_low": [-0.6],
            "indirect_ci_high": [0.1],
        }
    )
    historical_normalized = _attach_primary_mediation_interval(historical)
    assert historical_normalized.loc[0, "ci_low"] == -1.0
    assert historical_normalized.loc[0, "ci_high"] == 0.2

    with _workspace_tempdir() as tmpdir:
        canonical_path = Path(tmpdir) / "canonical_mediation.png"
        historical_path = Path(tmpdir) / "historical_mediation.png"
        plot_mediation_effects_bar(canonical, canonical_path)
        plot_mediation_effects_bar(historical, historical_path)
        assert canonical_path.exists() and canonical_path.stat().st_size > 0
        assert historical_path.exists() and historical_path.stat().st_size > 0


def test_exosome_fraction_not_promoted_when_gate_fails():
    direct_mediation_level = _evidence_level(
        estimable=True,
        tier="fully_linked",
        has_linked_mediation=True,
    )
    fraction_status = _causal_exosome_fraction_status()

    assert direct_mediation_level == 4
    assert fraction_status["estimable"] is False
    assert fraction_status["evidence_level"] == 0
    assert np.isnan(fraction_status["ratio"])
    assert "response_alignment_summary.csv" in fraction_status["reason"]


def test_missing_linkage_confidence_does_not_default_to_high_confidence():
    prim = pd.DataFrame(
        {
            "animal_id": ["A", "B", "C"],
            "group": ["O_GES", "O_V", "O_WT"],
        }
    )
    plasma_without_confidence = pd.DataFrame(
        {
            "animal_id": ["A", "B", "C"],
            "group": ["GES", "V", "WT"],
        }
    )

    audit = audit_primate_plasma_linkage(prim, plasma_without_confidence)
    estimability = build_estimability_report(
        audit,
        min_samples_for_mediation=3,
        min_overlap_animals=3,
        min_treated_overlap=1,
        min_control_overlap=2,
    )

    assert audit["has_plasma_animal_id_confidence_col"] is False
    assert audit["n_plasma_high_conf_animal_id_non_null"] == 0
    assert audit["n_overlap_animal_ids_high_conf"] == 0
    assert estimability["tier"] == "unlinked"
    assert estimability["can_do_mediation"] is False
    assert estimability["reason_code"] == PLASMA_LINKAGE_CONFIDENCE_MISSING
    assert estimability["missing_author_key"] == AUTHOR_KEY_PLASMA_TO_ANIMAL


def test_exosome_fraction_unestimable_has_explicit_reason():
    cells = pd.DataFrame({"mean_effect": [0.2, -0.1]}, index=["liver", "heart"])
    exo = pd.DataFrame(columns=["mean_effect"])
    out = estimate_exosome_fraction_with_uncertainty(
        effect_cells=cells,
        effect_exosomes=exo,
        min_common_tissues=3,
        n_bootstrap=100,
        n_permutations=100,
        random_state=1,
    )
    assert out["estimable"] is False
    assert isinstance(out["reason"], str)
    assert len(out["reason"]) > 0
    assert out["reason_code"]
    assert out["missing_author_key"]


def test_standard_schema_columns_always_present():
    df = pd.DataFrame([{"metric": 1.23}])
    out = _ensure_standard_schema(
        df,
        available=False,
        estimable=False,
        reason="stub",
        n_used=0,
        method="unit_test",
        ci_low=np.nan,
        ci_high=np.nan,
        evidence_level=0,
    )
    for col in STANDARD_RESULT_COLUMNS:
        assert col in out.columns
    assert out.loc[0, "reason"] == "stub"
    assert out.loc[0, "reason_code"]
    assert out.loc[0, "missing_author_key"]


def test_tissue_effect_uncertainty_annotation_keeps_nominal_claims_separate():
    df = pd.DataFrame(
        [
            {"tissue": "A", "effect_median": -1.0, "ci_low": -2.0, "ci_high": 0.5},
            {"tissue": "B", "effect_median": 1.2, "ci_low": 0.4, "ci_high": 2.0},
        ]
    )
    out = annotate_effect_uncertainty(df)

    assert bool(out.loc[0, "ci_crosses_zero"]) is True
    assert out.loc[0, "effect_direction"] == "younger_shift"
    assert out.loc[0, "interpretation_label"] == "nominal_younger_shift_ci_crosses_zero"
    assert bool(out.loc[1, "ci_crosses_zero"]) is False
    assert out.loc[1, "interpretation_label"] == "supported_older_shift_ci_excludes_zero"
    assert out["signal_to_uncertainty"].notna().all()


def test_non_estimable_outputs_receive_author_metadata_reason_codes():
    out = _ensure_standard_schema(
        pd.DataFrame(
            [
                {
                    "estimable": False,
                    "reason": "No plasma samples could be linked to bulk animal IDs.",
                    "method": "plasma_linkage_qc",
                }
            ]
        )
    )
    assert out.loc[0, "reason_code"] == PLASMA_BULK_ANIMAL_LINKAGE_MISSING
    assert out.loc[0, "missing_author_key"] == AUTHOR_KEY_PLASMA_BULK_LINKAGE

    estimability = build_estimability_report(
        {
            "has_prim_animal_id_col": True,
            "has_plasma_animal_id_col": True,
            "has_plasma_animal_id_confidence_col": True,
            "n_overlap_animal_ids_high_conf": 1,
            "n_overlap_treated_animals": 1,
            "n_overlap_control_animals": 0,
        },
        min_overlap_animals=20,
    )
    assert estimability["reason_code"] == INSUFFICIENT_LINKED_ANIMALS
    assert estimability["missing_author_key"] == AUTHOR_KEY_PLASMA_BULK_LINKAGE


def test_mouse_exosome_parser_recovers_expected_fields():
    meta = build_mouse_exosome_metadata(["21_M_brain_GES_1", "8_W_liver_Ctrl_10"])
    first = meta.loc[meta["sample_id"] == "21_M_brain_GES_1"].iloc[0]
    second = meta.loc[meta["sample_id"] == "8_W_liver_Ctrl_10"].iloc[0]

    assert bool(first["parse_ok"]) is True
    assert float(first["age"]) == 21.0
    assert str(first["sex"]) == "M"
    assert str(first["tissue"]) == "brain"
    assert str(first["arm"]) == "GES"

    assert bool(second["parse_ok"]) is True
    assert str(second["sex"]) == "F"
    assert str(second["arm"]) == "Ctrl"


def test_mouse_exosome_tissue_clock_uses_current_clock_interface():
    rng = np.random.default_rng(7)
    reference_ids = [f"ref_{i:02d}" for i in range(20)]
    ges_ids = [f"ges_{i}" for i in range(3)]
    vehicle_ids = [f"veh_{i}" for i in range(3)]
    sample_ids = reference_ids + ges_ids + vehicle_ids
    reference_ages = np.tile([4.0, 8.0, 12.0, 16.0], 5)
    ages = np.concatenate([reference_ages, np.repeat(20.0, 6)])
    signal = np.tile(ages, (30, 1)) * np.linspace(0.02, 0.08, 30)[:, None]
    expression = pd.DataFrame(
        signal + rng.normal(scale=0.2, size=signal.shape),
        index=[f"gene_{i}" for i in range(30)],
        columns=sample_ids,
    )
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "tissue": "brain",
            "arm": ["Baseline"] * 20 + ["GES"] * 3 + ["Veh"] * 3,
            "age": ages,
        }
    )

    effects, sample_outcomes = compute_mouse_exosome_tissue_effects(
        expression,
        metadata,
        contrasts=(("GES", "Veh"),),
        n_bootstrap=20,
        n_permutations=20,
        return_sample_outcomes=True,
    )

    assert len(effects) == 1
    assert bool(effects.loc[0, "estimable"]) is True
    assert effects.loc[0, "reason_code"] == "OK"
    assert int(effects.loc[0, "n_used"]) == 6
    assert len(sample_outcomes) == len(sample_ids)
    assert sample_outcomes["sample_id"].is_unique
    assert sample_outcomes["predicted_age_mouse"].notna().all()


def test_omix009283_metadata_builder_reads_header_sample_ids():
    with _workspace_tempdir() as tmpdir:
        tmp_path = Path(tmpdir)
        matrix_path = tmp_path / "OMIX009283-01.txt"
        matrix_path.write_text(
            "gene\t21_M_brain_GES_1\t8_W_liver_Ctrl_10\n"
            "GeneA\t1\t2\n",
            encoding="utf-8",
        )

        metadata = build_omix009283_metadata_table(matrix_path)
        assert metadata["sample_id"].tolist() == ["21_M_brain_GES_1", "8_W_liver_Ctrl_10"]
        assert metadata["parse_ok"].fillna(False).all()
        assert metadata.loc[metadata["sample_id"] == "21_M_brain_GES_1", "arm"].iloc[0] == "GES"


def test_exosome_alignment_unestimable_has_explicit_reason():
    prim = pd.DataFrame(
        {"mean_effect": [-0.4, -0.2]},
        index=["Hippocampus", "Liver_L"],
    )
    prim.index.name = "tissue"
    mouse = pd.DataFrame(
        [
            {
                "tissue": "brain",
                "contrast": "GES_vs_Veh",
                "mean_effect": -0.3,
                "estimable": True,
                "available": True,
                "ci_low": -0.5,
                "ci_high": -0.1,
                "permutation_p_value": 0.04,
            },
            {
                "tissue": "liver",
                "contrast": "WT_vs_Veh",
                "mean_effect": -0.1,
                "estimable": True,
                "available": True,
                "ci_low": -0.2,
                "ci_high": 0.0,
                "permutation_p_value": 0.2,
            },
        ]
    )
    by_tissue, summary = compute_exosome_alignment_tables(
        prim,
        mouse,
        tissue_map={"brain": "Hippocampus", "liver": "Liver_L"},
        contrasts=["GES_vs_Veh"],
        min_common_tissues=3,
        n_bootstrap=50,
        n_permutations=50,
        random_state=1,
    )
    assert bool(summary.loc[0, "estimable"]) is False
    assert isinstance(summary.loc[0, "reason"], str)
    assert len(summary.loc[0, "reason"]) > 0
    assert "common tissues" in summary.loc[0, "reason"]
    assert "reason" in by_tissue.columns


def test_exosome_alignment_propagates_disabled_mouse_block_reason():
    prim = pd.DataFrame({"mean_effect": [-0.4]}, index=["Heart"])
    prim.index.name = "tissue"
    mouse = pd.DataFrame(
        [
            {
                "tissue": "NA",
                "contrast": "NA",
                "mean_effect": np.nan,
                "available": False,
                "estimable": False,
                "reason": "Mouse exosome block disabled by config.",
                "reason_code": CONFIG_DISABLED,
                "missing_author_key": AUTHOR_KEY_NOT_REQUIRED,
            }
        ]
    )

    by_tissue, summary = compute_exosome_alignment_tables(
        prim,
        mouse,
        contrasts=["GES_vs_Veh"],
        min_common_tissues=3,
        n_bootstrap=10,
        n_permutations=10,
        random_state=1,
    )

    for output in (by_tissue, summary):
        assert len(output) == 1
        assert bool(output.loc[0, "estimable"]) is False
        assert output.loc[0, "reason_code"] == CONFIG_DISABLED
        assert output.loc[0, "missing_author_key"] == AUTHOR_KEY_NOT_REQUIRED
        assert "disabled" in output.loc[0, "reason"].lower()


def test_cross_species_alignment_preserves_evidence_tier():
    prim = pd.DataFrame(
        {"mean_effect": [-0.4, -0.2, -0.1, 0.3]},
        index=["Renal_cortex", "Liver_L", "Quadriceps_muscle", "Hippocampus"],
    )
    prim.index.name = "tissue"
    mouse = pd.DataFrame(
        {
            "tissue": ["kidney", "liver", "muscle", "brain"],
            "contrast": ["GES_vs_Veh"] * 4,
            "mean_effect": [-0.3, -0.1, -0.05, 0.2],
            "estimable": [True] * 4,
            "available": [True] * 4,
            "ci_low": [-0.5, -0.3, -0.2, 0.0],
            "ci_high": [-0.1, 0.1, 0.1, 0.4],
            "permutation_p_value": [0.04, 0.2, 0.3, 0.1],
        }
    )

    by_tissue, summary = compute_exosome_alignment_tables(
        prim,
        mouse,
        tissue_mapping=(
            Path(__file__).resolve().parents[1]
            / "config"
            / "cross_species_tissue_map.csv"
        ),
        contrasts=["GES_vs_Veh"],
        min_common_tissues=3,
        n_bootstrap=20,
        n_permutations=20,
        random_state=1,
    )

    assert summary["estimable"].astype(bool).all()
    assert by_tissue["estimable"].astype(bool).all()
    assert int(summary.loc[0, "n_common_tissues"]) == 3
    assert int(summary.loc[0, "n_weak_context_tissues"]) == 1
    brain = by_tissue.loc[by_tissue["mouse_tissue"] == "brain"].iloc[0]
    assert brain["compatibility_tier"] == "weak_context"
    assert bool(brain["include_in_primary"]) is False
    assert brain["analysis_scope"] == "weak_context_sensitivity"
    assert set(summary["evidence_level"].astype(int)) == {3}
    assert set(by_tissue["evidence_level"].astype(int)) == {3}
    assert not (summary["evidence_level"].astype(int) == 4).any()


def test_tissue_mapping_contract_rejects_weak_context_in_primary():
    mapping = pd.DataFrame(
        [
            {
                "mouse_tissue": "brain",
                "primate_tissue": "Hippocampus",
                "compatibility_tier": "weak_context",
                "include_in_primary": True,
                "source_status": "SOURCE_VERIFICATION_PENDING",
                "mapping_note": "Not anatomically equivalent.",
            }
        ]
    )
    with pytest.raises(ValueError, match="cannot enter the primary"):
        load_cross_species_tissue_mapping(mapping)


def test_evidence_level_ladder_supports_alignment_and_mediation():
    assert _evidence_level(estimable=False) == 0
    assert _evidence_level(estimable=True) == 1
    assert _evidence_level(estimable=True, has_linkage_support=True) == 2
    assert _evidence_level(estimable=True, has_exosome_alignment=True) == 3
    assert _evidence_level(
        estimable=True,
        tier="fully_linked",
        has_linked_mediation=True,
    ) == 4


def test_omix007582_audit_refuses_underdetermined_biological_mapping():
    with _workspace_tempdir() as tmpdir:
        tmp_path = Path(tmpdir)
        matrix_path = tmp_path / "OMIX007582_beta_matrix.csv"
        metadata_path = tmp_path / "OMIX007582-02.csv"
        idat_dir = tmp_path / "OMIX007582_idat"
        zip_path = tmp_path / "OMIX007582-03.zip"
        idat_dir.mkdir()

        pd.DataFrame(
            {
                "207925070004_R01C01": [0.1, 0.2],
                "207925070004_R01C02": [0.3, 0.4],
            },
            index=["cg1", "cg2"],
        ).to_csv(matrix_path)
        pd.DataFrame(
            {
                "OriginalSampleName": ["FV1-Heart", "FWT2-Heart"],
                "OriginalSampleName.1": ["FV1-Heart", "FWT2-Heart"],
                "sample": ["FV1", "FWT2"],
                "tissue": ["Heart", "Heart"],
                "group": ["O_V", "O_WT"],
            }
        ).to_csv(metadata_path, index=False)

        for suffix in ("Grn", "Red"):
            (idat_dir / f"207925070004_R01C01_{suffix}.idat").write_text("", encoding="utf-8")
            (idat_dir / f"207925070004_R01C02_{suffix}.idat").write_text("", encoding="utf-8")

        with zipfile.ZipFile(zip_path, "w") as handle:
            handle.writestr("207925070004_R01C01_Grn.idat", "")
            handle.writestr("207925070004_R01C01_Red.idat", "")
            handle.writestr("207925070004_R01C02_Grn.idat", "")
            handle.writestr("207925070004_R01C02_Red.idat", "")

        summary, overlap_audit, technical_inventory, metadata_inventory = build_omix007582_sample_map_audit(
            matrix_path=matrix_path,
            metadata_path=metadata_path,
            idat_dir=idat_dir,
            zip_path=zip_path,
        )

        assert bool(summary.loc[0, "available"]) is True
        assert bool(summary.loc[0, "estimable"]) is False
        assert str(summary.loc[0, "mapping_status"]) == "technical_ids_only"
        assert int(summary.loc[0, "exact_overlap_ids"]) == 0
        assert bool(summary.loc[0, "beta_equals_idat_directory"]) is True
        assert bool(summary.loc[0, "beta_equals_idat_archive"]) is True
        assert "cannot be recovered" in str(summary.loc[0, "reason"])
        assert summary.loc[0, "reason_code"] == OMIX007582_SENTRIX_SAMPLE_SHEET_MISSING
        assert summary.loc[0, "missing_author_key"] == AUTHOR_KEY_OMIX007582_SENTRIX
        assert int(overlap_audit["n_exact_overlaps"].sum()) == 0
        assert technical_inventory["mapped_sample_id"].isna().all()
        assert metadata_inventory["mapped_technical_id"].isna().all()


def test_build_default_config_supports_demo_profile():
    with _workspace_tempdir() as tmpdir:
        tmp_path = Path(tmpdir)
        processed = tmp_path / "data" / "PROCESSED"
        processed.mkdir(parents=True)
        (processed / "OMIX007580_01_example.txt").write_text("gene\ts1\nG1\t1\n", encoding="utf-8")
        pd.DataFrame({"sample_id": ["s1"], "group": ["Y_C"], "age": [4.0], "tissue": ["Brain"]}).to_csv(
            processed / "OMIX007580-02_example.csv",
            index=False,
        )
        pd.DataFrame({"Gene name": ["P1"], "FY_1": [1.0]}).to_csv(
            processed / "OMIX007581-01_example.csv",
            index=False,
        )
        pd.DataFrame({"s1": [0.1]}, index=["cg1"]).to_csv(processed / "OMIX007582_beta_matrix_example.csv")
        pd.DataFrame({"OriginalSampleName": ["s1"], "sample": ["s1"], "tissue": ["Brain"], "group": ["Y_C"]}).to_csv(
            processed / "OMIX007582-02_example.csv",
            index=False,
        )

        cfg = _build_default_config(tmp_path, profile="demo")
        assert cfg.data_profile == "demo"
        assert cfg.primate_bulk.matrix.name == "OMIX007580_01_example.txt"
        assert cfg.primate_plasma.matrix.name == "OMIX007581-01_example.csv"
        assert cfg.enable_mouse_exosome_block is False
        assert cfg.enable_subset_validation_block is False


def test_demo_profile_inputs_are_distributed():
    repo_root = Path(__file__).resolve().parents[1]
    demo_spec = PROFILE_SPECS["demo"]
    expected_assets = {
        f"data/PROCESSED/{value}"
        for key, value in demo_spec.items()
        if key.endswith(("_matrix", "_metadata")) and value
    }

    for rel_path in expected_assets:
        assert (repo_root / rel_path).is_file(), rel_path


def test_demo_methylation_asset_is_parseable_csv():
    repo_root = Path(__file__).resolve().parents[1]
    matrix_path = repo_root / "data" / "PROCESSED" / "OMIX007582_beta_matrix_example.csv"
    matrix = pd.read_csv(matrix_path, index_col=0)

    assert matrix.shape[0] > 0
    assert matrix.shape[1] > 0
    assert str(matrix.index[0]).startswith("cg")


def test_omix007582_support_script_is_parameterized():
    repo_root = Path(__file__).resolve().parents[1]
    script_text = (repo_root / "src" / "scripts" / "process_OMIX007582_Mammal40.R").read_text(encoding="utf-8")

    assert "--idat-dir" in script_text
    assert "--output-dir" in script_text
    assert "--max-prefixes" in script_text
    assert "--prefixes" in script_text
    assert "--enable-partial-mapping" in script_text
    assert "D:/DATA" not in script_text
    assert "data/RAW/data/OMIX007582_beta_matrix.csv" not in script_text


def test_optional_r_environment_spec_bootstraps_biocmanager():
    repo_root = Path(__file__).resolve().parents[1]
    env_text = (repo_root / "environment-omix007582-r.yml").read_text(encoding="utf-8")
    main_env_text = (repo_root / "environment.yml").read_text(encoding="utf-8")

    assert "name: srsc-omix007582-r" in env_text
    assert "conda-forge" in env_text
    assert "bioconda" in env_text
    assert "r-base" in env_text
    assert "r-biocmanager" in env_text
    assert "r-base" in main_env_text
    assert "r-biocmanager" in main_env_text


def test_public_data_ceiling_document_names_required_author_keys():
    repo_root = Path(__file__).resolve().parents[1]
    doc_text = (repo_root / "docs" / "scientific_limitations.md").read_text(
        encoding="utf-8"
    )

    required_terms = [
        "plasma proteomics",
        "animal_id",
        "OMIX007582",
        "Sentrix",
        "exosome cargo",
        "causal quantity remains",
        "aligned-response coefficient",
        "non-estimable",
        "reason_code",
        "missing_author_key",
    ]
    for term in required_terms:
        assert term in doc_text


def test_oriented_plasma_axis_reports_young_like_ges_shift():
    samples = ["Y1", "Y2", "V1", "V2", "WT1", "WT2", "GES1", "GES2"]
    expr = pd.DataFrame(
        [
            [1.0, 1.2, 4.0, 4.2, 3.9, 4.1, 2.1, 2.0],
            [1.1, 1.0, 3.8, 4.1, 4.2, 4.0, 2.2, 2.1],
            [1.0, 1.1, 3.7, 4.0, 3.8, 4.1, 1.9, 2.0],
        ],
        index=["POSTN", "CRP", "IGF1"],
        columns=samples,
    )
    meta = pd.DataFrame(
        {
            "sample_id": samples,
            "group": ["Y", "Y", "V", "V", "WT", "WT", "GES", "GES"],
        }
    )

    scores, loadings, summary = build_oriented_plasma_aging_axis(
        expr,
        meta,
        young_groups=("Y",),
        old_control_groups=("V",),
        treated_groups=("GES",),
        n_top_proteins=3,
        n_bootstrap=50,
        n_permutations=50,
        random_state=1,
    )

    row = summary.iloc[0]
    assert bool(row["estimable"])
    assert row["axis_orientation"] == "higher_older_like"
    assert int(row["evidence_level"]) == 1
    assert row["old_control_median"] > row["young_control_median"]
    assert row["treated_median"] < row["old_control_median"]
    assert row["treated_shift_label"] == "young_like_shift_vs_old_controls"
    assert {"sample_id", "plasma_age_axis_score"}.issubset(scores.columns)
    assert {"protein", "loading", "abs_loading"}.issubset(loadings.columns)


def test_plasma_axis_fit_is_independent_of_projected_treatment_values():
    samples = ["Y1", "Y2", "V1", "V2", "WT1", "WT2", "GES1", "GES2"]
    expr = pd.DataFrame(
        [
            [1.0, 1.2, 3.0, 3.2, 2.9, 3.1, 1.1, 1.0],
            [1.1, 1.0, 2.8, 3.1, 3.2, 3.0, 1.2, 1.1],
            [1.0, 1.1, 2.7, 3.0, 2.8, 3.1, 0.9, 1.0],
        ],
        index=["P1", "P2", "P3"],
        columns=samples,
    )
    meta = pd.DataFrame(
        {
            "sample_id": samples,
            "group": ["Y", "Y", "V", "V", "WT", "WT", "GES", "GES"],
        }
    )

    first_scores, first_loadings, _ = build_oriented_plasma_aging_axis(
        expr,
        meta,
        young_groups=("Y",),
        old_control_groups=("V",),
        treated_groups=("GES",),
        n_top_proteins=3,
        n_bootstrap=0,
        n_permutations=0,
        random_state=3,
    )
    changed = expr.copy()
    changed.loc[:, ["WT1", "WT2", "GES1", "GES2"]] *= 100.0
    second_scores, second_loadings, _ = build_oriented_plasma_aging_axis(
        changed,
        meta,
        young_groups=("Y",),
        old_control_groups=("V",),
        treated_groups=("GES",),
        n_top_proteins=3,
        n_bootstrap=0,
        n_permutations=0,
        random_state=3,
    )

    reference_ids = ["Y1", "Y2", "V1", "V2"]
    first_reference = first_scores.set_index("sample_id").loc[reference_ids, "plasma_age_axis_score"]
    second_reference = second_scores.set_index("sample_id").loc[reference_ids, "plasma_age_axis_score"]
    np.testing.assert_allclose(first_reference, second_reference, atol=1e-12)
    first_loading = first_loadings.set_index("feature_id")["loading"].sort_index()
    second_loading = second_loadings.set_index("feature_id")["loading"].sort_index()
    np.testing.assert_allclose(first_loading, second_loading, atol=1e-12)


def test_plasma_axis_delta_age_correlation_uses_high_confidence_links_only():
    prim_meta = pd.DataFrame(
        {
            "animal_id": ["A", "B", "C", "D", "E"],
            "delta_age": [1.0, 2.0, 3.0, 4.0, 5.0],
            "group": ["O_V", "O_V", "O_WT", "O_GES", "O_GES"],
        }
    )
    plasma_meta = pd.DataFrame(
        {
            "animal_id": ["A", "B", "C", "D", "E"],
            "plasma_age_axis_score": [1.0, 2.0, 3.0, 4.0, 100.0],
            "animal_id_confidence": ["high", "high", "metadata", "metadata_exact", "low"],
            "group": ["V", "V", "WT", "GES", "GES"],
        }
    )

    out = correlate_plasma_axis_with_delta_age(
        prim_meta,
        plasma_meta,
        min_animals=3,
        n_bootstrap=50,
        n_permutations=50,
        random_state=1,
    )

    row = out.iloc[0]
    assert bool(row["estimable"])
    assert int(row["n_animals"]) == 3
    assert row["spearman_rho"] > 0.99
    assert "hypothesis-generating" in row["reason"]


def test_scientific_objectives_document_separates_replication_from_claims():
    repo_root = Path(__file__).resolve().parents[1]
    doc_text = (repo_root / "docs" / "scientific_scope.md").read_text(
        encoding="utf-8"
    )

    required_terms = [
        "not only a reproduction",
        "aligned response component",
        "do not estimate an exosome-attributable percentage",
        "Therapeutic insight",
        "Hypothesis-generating result",
        "Public-Data Ceiling",
        "request author metadata",
    ]
    for term in required_terms:
        assert term in doc_text


def test_omix009284_audit_detects_pbmc_sc_export_and_suffix_mapping():
    with _workspace_tempdir() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        (data_dir / "OMIX009284-01.txt").write_text(
            "AAAC-1_1\tAAAG-1_1\nGeneA\t1\t0\n",
            encoding="utf-8",
        )
        (data_dir / "OMIX009284-02.txt").write_text(
            "TTTC-1_2\tTTTG-1_2\tTTTA-1_2\nGeneA\t0\t1\t2\n",
            encoding="utf-8",
        )

        metadata = pd.DataFrame(
            {
                "orig.ident": ["SeuratProject"] * 5,
                "nCount_RNA": [10, 11, 12, 13, 14],
                "nFeature_RNA": [5, 5, 6, 6, 7],
                "percent.mt": [1.0, 1.2, 1.3, 1.1, 1.4],
                "tissue": ["PBMC"] * 5,
                "group": ["GES", "GES", "Vehical", "Vehical", "Vehical"],
                "sample": ["G1", "G1", "V1", "V1", "V1"],
                "cell": ["AAAC-1_1", "AAAG-1_1", "TTTC-1_2", "TTTG-1_2", "TTTA-1_2"],
                "nCount_SCT": [9, 10, 11, 12, 13],
                "nFeature_SCT": [4, 4, 5, 5, 6],
            },
            index=["AAAC-1_1", "AAAG-1_1", "TTTC-1_2", "TTTG-1_2", "TTTA-1_2"],
        )
        metadata.to_csv(data_dir / "OMIX009284-27.txt", sep="\t")

        summary, file_inventory, sample_mapping = build_omix009284_audit(data_dir=data_dir)

        assert bool(summary.loc[0, "available"]) is True
        assert bool(summary.loc[0, "estimable"]) is False
        assert str(summary.loc[0, "dataset_scope"]) == "PBMC single-cell RNA export"
        assert int(summary.loc[0, "n_expression_matrices"]) == 2
        assert int(summary.loc[0, "n_metadata_tables"]) == 1
        assert bool(summary.loc[0, "matrix_suffixes_match_metadata"]) is True
        assert str(summary.loc[0, "tissue_labels"]) == "PBMC"
        assert set(file_inventory["role"]) == {"gene_by_cell_matrix", "cell_metadata_table"}
        assert sample_mapping["mapping_confidence"].eq("high").all()
        assert sample_mapping["sample"].tolist() == ["G1", "V1"]


def test_load_omix_matrix_accepts_string_feature_ids():
    with _workspace_tempdir() as tmpdir:
        tmp_path = Path(tmpdir)
        matrix_path = tmp_path / "OMIX007580_01_example.txt"
        matrix_path.write_text(
            "gene_id\ts1\ts2\nGene_0001\t1\t2\nGene_0002\t3\t4\n",
            encoding="utf-8",
        )

        matrix = load_omix_matrix(matrix_path, allowed_samples={"s1", "s2"})
        assert matrix.index.tolist() == ["Gene_0001", "Gene_0002"]
        assert matrix.columns.tolist() == ["s1", "s2"]
        assert float(matrix.loc["Gene_0001", "s1"]) == 1.0


def test_report_figure_layer_generates_manifest_and_pngs():
    with _workspace_tempdir() as tmpdir:
        tmp_path = Path(tmpdir)
        results_dir = tmp_path / "results"
        out_dir = tmp_path / "figures" / "report"
        assets_dir = tmp_path / "docs" / "assets"
        results_dir.mkdir()

        pd.DataFrame(
            [
                {
                    "n_samples": 40,
                    "MAE": 2.1,
                    "RMSE": 2.8,
                    "pearson_r": 0.82,
                    "spearman_r": 0.85,
                    "calibration_slope": 0.71,
                    "cv_strategy": "GroupKFold(animal_id)",
                    "cv_n_splits": 5,
                    "cv_n_groups": 10,
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 40,
                    "method": "transcriptomic_clock_cv",
                    "evidence_level": 1,
                }
            ]
        ).to_csv(results_dir / "clock_metrics_primates.csv", index=False)

        pd.DataFrame(
            [
                {
                    "tissue": "Heart",
                    "effect_median": -2.0,
                    "ci_low": -3.0,
                    "ci_high": -1.0,
                    "available": True,
                    "estimable": True,
                    "reason": "",
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 12,
                    "method": "delta_age_group_bootstrap",
                    "evidence_level": 1,
                }
            ]
        ).to_csv(results_dir / "rejuvenation_by_tissue.csv", index=False)
        pd.DataFrame(
            [
                {
                    "contrast": "GES_vs_Veh",
                    "n_common_tissues": 4,
                    "mean_standardized_effect_similarity": 0.6,
                    "ci_low": 0.3,
                    "ci_high": 0.8,
                    "spearman_rho": 0.4,
                    "fraction_signed_concordant": 0.75,
                    "available": True,
                    "estimable": True,
                    "reason": "",
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 4,
                    "method": "cross_species_effect_alignment_summary",
                    "evidence_level": 3,
                }
            ]
        ).to_csv(results_dir / "exosome_alignment_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "contrast": "GES_vs_Veh",
                    "mouse_tissue": "heart",
                    "primate_tissue": "Heart",
                    "macaque_effect": -2.0,
                    "mouse_effect": -1.5,
                    "signed_concordance": 1,
                    "rank_concordance": 1,
                    "standardized_effect_similarity": 0.8,
                    "residual_component": 0.2,
                    "available": True,
                    "estimable": True,
                    "reason": "",
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 4,
                    "method": "cross_species_effect_alignment",
                    "ci_low": -0.2,
                    "ci_high": 0.3,
                    "evidence_level": 3,
                },
                {
                    "contrast": "GES_vs_Veh",
                    "mouse_tissue": "liver",
                    "primate_tissue": "Liver",
                    "macaque_effect": -1.0,
                    "mouse_effect": 0.6,
                    "signed_concordance": 0,
                    "rank_concordance": 0,
                    "standardized_effect_similarity": 0.3,
                    "residual_component": 1.4,
                    "available": True,
                    "estimable": True,
                    "reason": "",
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 4,
                    "method": "cross_species_effect_alignment",
                    "ci_low": -0.5,
                    "ci_high": 0.8,
                    "evidence_level": 3,
                },
            ]
        ).to_csv(results_dir / "exosome_alignment_by_tissue.csv", index=False)
        pd.DataFrame(
            [
                {
                    "tier": "fully_linked",
                    "can_do_mediation": True,
                    "n_overlap_animal_ids": 24,
                    "n_overlap_treated_animals": 8,
                    "n_overlap_control_animals": 16,
                    "min_overlap_animals": 20,
                    "min_treated_overlap": 6,
                    "min_control_overlap": 12,
                    "min_samples_for_mediation": 12,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "available": True,
                    "estimable": True,
                    "reason": "",
                    "n_used": 24,
                    "method": "estimability_gate",
                    "evidence_level": 0,
                }
            ]
        ).to_csv(results_dir / "estimability_report.csv", index=False)
        pd.DataFrame(
            [
                {
                    "n_plasma_total": 30,
                    "n_mapped_non_null": 24,
                    "n_mapped_high_conf": 24,
                    "n_mapped_valid_in_bulk": 24,
                    "n_unresolved": 6,
                    "mapping_collision_count": 0,
                    "mapping_coverage": 0.8,
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 30,
                    "method": "plasma_linkage_qc",
                    "evidence_level": 2,
                }
            ]
        ).to_csv(results_dir / "linkage_qc_report.csv", index=False)
        pd.DataFrame(
            [
                {
                    "ACME": -0.2,
                    "ACME_CI": "(-0.6, 0.1)",
                    "ADE": -0.4,
                    "ADE_CI": "(-0.8, -0.1)",
                    "Total": -0.6,
                    "Total_CI": "(-1.0, 0.2)",
                    "PropMediated": 0.33,
                    "PropMediated_CI": "(-0.2, 0.9)",
                    "tier": "fully_linked",
                    "n_overlap_animal_ids": 24,
                    "available": True,
                    "estimable": True,
                    "reason": "",
                    "n_used": 24,
                    "method": "linear_mediation_bootstrap",
                    "evidence_level": 4,
                    "reason_code": "OK",
                    "missing_author_key": "",
                }
            ]
        ).to_csv(results_dir / "mediation_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "feature_id": "P00001",
                    "protein_accession": "P00001",
                    "gene_name": "POSTN",
                    "protein": "POSTN",
                    "contrast": "GES_vs_V",
                    "is_primary": True,
                    "log2_fold_change": -0.7,
                    "q_value": 0.02,
                    "fdr_significant": True,
                    "bootstrap_ci_low": -0.9,
                    "bootstrap_ci_high": -0.4,
                    "stable_association": True,
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 10,
                    "method": "pairwise_ols_hc3_log2_abundance_group_sex",
                    "evidence_level": 1,
                }
            ]
        ).to_csv(results_dir / "plasma_biomarkers.csv", index=False)
        pd.DataFrame(
            [
                {
                    "sample_id": "Y1",
                    "group": "Y",
                    "plasma_age_axis_score": -1.5,
                    "raw_pc1_score": -1.5,
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 4,
                    "method": "reference_trained_oriented_plasma_pc1_age_axis",
                    "evidence_level": 1,
                },
                {
                    "sample_id": "V1",
                    "group": "V",
                    "plasma_age_axis_score": 1.0,
                    "raw_pc1_score": 1.0,
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 4,
                    "method": "reference_trained_oriented_plasma_pc1_age_axis",
                    "evidence_level": 1,
                },
                {
                    "sample_id": "WT1",
                    "group": "WT",
                    "plasma_age_axis_score": 1.2,
                    "raw_pc1_score": 1.2,
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 4,
                    "method": "reference_trained_oriented_plasma_pc1_age_axis",
                    "evidence_level": 1,
                },
                {
                    "sample_id": "GES1",
                    "group": "GES",
                    "plasma_age_axis_score": -0.2,
                    "raw_pc1_score": -0.2,
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 4,
                    "method": "reference_trained_oriented_plasma_pc1_age_axis",
                    "evidence_level": 1,
                },
            ]
        ).to_csv(results_dir / "plasma_age_axis_scores.csv", index=False)
        pd.DataFrame(
            [
                {
                    "loading_rank": 1,
                    "protein": "POSTN",
                    "loading": 0.7,
                    "abs_loading": 0.7,
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 3,
                    "method": "reference_trained_oriented_plasma_pc1_age_axis",
                    "evidence_level": 1,
                },
                {
                    "loading_rank": 2,
                    "protein": "IGF1",
                    "loading": -0.5,
                    "abs_loading": 0.5,
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 3,
                    "method": "reference_trained_oriented_plasma_pc1_age_axis",
                    "evidence_level": 1,
                },
                {
                    "loading_rank": 3,
                    "protein": "CRP",
                    "loading": 0.4,
                    "abs_loading": 0.4,
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 3,
                    "method": "reference_trained_oriented_plasma_pc1_age_axis",
                    "evidence_level": 1,
                },
            ]
        ).to_csv(results_dir / "plasma_age_axis_loadings.csv", index=False)
        pd.DataFrame(
            [
                {
                    "pc1_explained_variance_ratio": 0.62,
                    "treated_shift_label": "young_like_shift_vs_old_controls",
                    "treated_fraction_of_old_young_gap": 0.4,
                    "treated_vs_old_permutation_p_value": 0.08,
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 4,
                    "method": "reference_trained_oriented_plasma_pc1_age_axis",
                    "evidence_level": 1,
                }
            ]
        ).to_csv(results_dir / "plasma_age_axis_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "n_animals": 4,
                    "spearman_rho": 0.5,
                    "ci_low": -0.2,
                    "ci_high": 0.9,
                    "permutation_p_value": 0.2,
                    "available": True,
                    "estimable": True,
                    "reason": "Linked animal count is modest.",
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 4,
                    "method": "linked_plasma_age_axis_delta_age_spearman",
                    "evidence_level": 2,
                }
            ]
        ).to_csv(results_dir / "plasma_age_axis_delta_age_correlation.csv", index=False)
        pd.DataFrame(
            [
                {
                    "analysis_type": "control_set",
                    "scenario": "primary",
                    "effect": -1.0,
                    "direction": "negative",
                    "available": True,
                    "estimable": True,
                    "reason_code": "OK",
                    "missing_author_key": "",
                    "n_used": 10,
                    "method": "global_delta_age_bootstrap",
                    "evidence_level": 1,
                }
            ]
        ).to_csv(results_dir / "sensitivity_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "available": False,
                    "estimable": False,
                    "reason": "Mammal40 biological sample map unavailable.",
                    "reason_code": "OMIX007582_SENTRIX_SAMPLE_SHEET_MISSING",
                    "missing_author_key": "omix007582_sentrix_barcode_position_to_biological_sample_tissue_group_sex_age",
                    "n_used": 0,
                    "method": "multimodal_concordance",
                    "evidence_level": 0,
                }
            ]
        ).to_csv(results_dir / "multimodal_concordance_summary.csv", index=False)

        manifest = generate_report_figures(
            results_dir=results_dir,
            out_dir=out_dir,
            top_n_plasma=5,
            portfolio_assets_dir=assets_dir,
        )

        assert len(manifest) == 15
        assert (out_dir / "report_figure_manifest.csv").exists()
        mediation_limitation = _claim_limitation(
            "Linked mediation",
            pd.read_csv(results_dir / "mediation_summary.csv"),
            estimable=True,
            reason_code="OK",
        )
        assert "OK" not in mediation_limitation
        assert "unstable" in mediation_limitation
        expected_figures = {
            "report_mediation_uncertainty.png",
            "report_exosome_alignment_by_tissue.png",
            "report_top_tissue_priority.png",
            "report_evidence_ladder.png",
            "report_plasma_biomarker_categories.png",
            "report_oriented_plasma_age_axis.png",
            "report_portfolio_aging_rejuvenation.png",
            "report_portfolio_multimodal_evidence.png",
            "report_portfolio_estimability_guardrail.png",
        }
        assert expected_figures.issubset(set(manifest["figure"]))
        for figure in manifest["figure"]:
            assert (out_dir / figure).exists(), figure
        assert {path.name for path in assets_dir.glob("*.png")} == {
            "aging_rejuvenation_signal.png",
            "multimodal_evidence_architecture.png",
            "estimability_guardrail.png",
        }
        portfolio_rows = manifest.set_index("figure")
        assert "methylation=blocked" in portfolio_rows.loc[
            "report_portfolio_multimodal_evidence.png", "message"
        ]
        assert "gate=pass" in portfolio_rows.loc[
            "report_portfolio_estimability_guardrail.png", "message"
        ]
        evidence_message = portfolio_rows.loc["report_evidence_ladder.png", "message"]
        assert "Observed=1" in evidence_message
        assert "Exploratory=3" in evidence_message

        readme_text = (Path(__file__).resolve().parents[1] / "README.md").read_text(
            encoding="utf-8"
        )
        assert readme_text.count("](docs/assets/") == 3
        for claim_status in ("Observed", "Supported", "Exploratory", "Not estimable", "Not established"):
            assert claim_status in readme_text
        assert "no current mechanism-facing claim meets that threshold" in readme_text


def test_portfolio_estimability_figure_preserves_fail_branch():
    with _workspace_tempdir() as tmpdir:
        tmp_path = Path(tmpdir)
        results_dir = tmp_path / "results"
        out_dir = tmp_path / "figures"
        results_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "tier": "unlinked",
                    "can_do_mediation": False,
                    "n_overlap_animal_ids": 0,
                    "n_overlap_treated_animals": 0,
                    "n_overlap_control_animals": 0,
                    "mapping_collision_count": 0,
                    "min_overlap_animals": 20,
                    "min_treated_overlap": 6,
                    "min_control_overlap": 12,
                    "reason_code": "PLASMA_LINKAGE_CONFIDENCE_MISSING",
                    "missing_author_key": "omix007581_plasma_sample_id_to_animal_id_group_sex_age",
                    "estimable": False,
                }
            ]
        ).to_csv(results_dir / "estimability_report.csv", index=False)
        pd.DataFrame(
            [
                {
                    "n_plasma_total": 12,
                    "n_mapped_high_conf": 0,
                    "mapping_collision_count": 0,
                }
            ]
        ).to_csv(results_dir / "linkage_qc_report.csv", index=False)
        _build_mediation_stub(
            reason="Plasma animal_id confidence column is missing.",
            tier="unlinked",
            n_overlap_animal_ids=0,
        ).to_csv(results_dir / "mediation_summary.csv", index=False)

        record = plot_portfolio_estimability_guardrail(results_dir, out_dir)

        assert record.status == "ok"
        assert "gate=fail" in record.message
        assert "mediation=blocked" in record.message
        assert record.path.exists()


def test_evidence_ladder_separates_within_plasma_from_linkage_dependent_claims():
    with _workspace_tempdir() as tmpdir:
        tmp_path = Path(tmpdir)
        results_dir = tmp_path / "results"
        out_dir = tmp_path / "figures"
        results_dir.mkdir()
        pd.DataFrame(
            [{"protein": "POSTN", "estimable": True, "stable_association": True}]
        ).to_csv(results_dir / "plasma_biomarkers.csv", index=False)
        pd.DataFrame([{"tier": "unlinked"}]).to_csv(
            results_dir / "estimability_report.csv", index=False
        )

        record = plot_evidence_ladder(results_dir, out_dir)

        assert record.status == "ok"
        assert "Exploratory=1" in record.message
        assert "Not estimable=3" in record.message
        assert record.path.exists()


def test_multimodal_figure_does_not_treat_candidate_mappings_as_valid_linkage():
    with _workspace_tempdir() as tmpdir:
        tmp_path = Path(tmpdir)
        results_dir = tmp_path / "results"
        out_dir = tmp_path / "figures"
        results_dir.mkdir()
        pd.DataFrame(
            [{"tissue": "Liver", "estimable": True, "ci_low": -1.0, "ci_high": 1.0}]
        ).to_csv(results_dir / "rejuvenation_by_tissue.csv", index=False)
        pd.DataFrame(
            [
                {
                    "n_plasma_total": 32,
                    "n_mapped_high_conf": 24,
                    "n_mapped_valid_in_bulk": 0,
                    "estimable": False,
                }
            ]
        ).to_csv(results_dir / "linkage_qc_report.csv", index=False)

        record = plot_portfolio_multimodal_evidence(results_dir, out_dir)

        assert record.status == "ok"
        assert "plasma gate=blocked" in record.message
        assert "valid links=0" in record.message
        assert record.path.exists()
