"""Validate the public demo's machine-readable result contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STANDARD_COLUMNS = {
    "available",
    "estimable",
    "reason",
    "n_used",
    "method",
    "ci_low",
    "ci_high",
    "evidence_level",
}

REQUIRED_OUTPUTS = {
    "clock_metrics_primates.csv": {"n_samples", "MAE", "RMSE", "cv_strategy"},
    "rejuvenation_by_tissue.csv": {
        "tissue",
        "contrast",
        "mean_effect",
        "p_value_adjusted",
        "ci_crosses_zero",
    },
    "global_rejuvenation_summary.csv": {"contrast", "mean_effect", "p_value"},
    "global_rejuvenation_clustered_sensitivity.csv": {
        "contrast",
        "mean_effect",
        "cluster_unit",
    },
    "tissue_expression_effects.csv": {"tissue", "mean_effect", "p_value"},
    "mouse_exosome_effects.csv": {"tissue", "contrast"},
    "mouse_exosome_signature_summary.csv": {"contrast"},
    "methylation_rejuvenation_by_tissue.csv": {"tissue"},
    "multimodal_concordance_summary.csv": {"n_common_tissues"},
    "ovary_subset_validation.csv": {"subset"},
    "hippocampus_subset_validation.csv": {"subset"},
    "plasma_to_animal_map.csv": {"sample_id", "animal_id_confidence"},
    "plasma_linkage_manifest_audit.csv": {
        "manifest_configured",
        "manifest_accepted",
        "expected_sha256",
        "observed_sha256",
        "n_validated_links",
    },
    "linkage_qc_report.csv": {"mapping_coverage", "mapping_collision_count"},
    "plasma_biomarkers.csv": {
        "feature_id",
        "protein_accession",
        "contrast",
        "log2_fold_change",
        "q_value",
        "is_primary",
    },
    "plasma_age_axis_scores.csv": {"sample_id", "plasma_age_axis_score"},
    "plasma_age_axis_loadings.csv": {"feature_id", "loading"},
    "plasma_age_axis_summary.csv": {"axis_orientation"},
    "plasma_age_axis_delta_age_correlation.csv": {"spearman_rho"},
    "linkage_audit.csv": {"mapping_collision_count", "n_overlap_animal_ids"},
    "estimability_report.csv": {"tier", "can_do_mediation"},
    "mediation_summary.csv": {"tier"},
    "exosome_alignment_by_tissue.csv": {"contrast", "primate_tissue"},
    "exosome_alignment_summary.csv": {"contrast", "n_common_tissues"},
    "cross_species_response_alignment_by_tissue.csv": {
        "mouse_tissue",
        "primate_tissue",
        "macaque_standardized_effect",
        "mouse_standardized_effect",
    },
    "cross_species_response_alignment_summary.csv": {
        "contrast_pair",
        "n_common_tissues",
        "cosine_similarity",
        "relative_response_norm",
        "aligned_response_coefficient",
    },
    "exosome_fraction_summary.csv": {"ratio", "tier"},
    "sensitivity_summary.csv": {"analysis_type", "scenario"},
}


def validate_demo_outputs(results_dir: Path) -> int:
    """Return the number of validated tables or raise with all detected errors."""
    errors: list[str] = []

    for filename, specific_columns in REQUIRED_OUTPUTS.items():
        path = results_dir / filename
        if not path.is_file():
            errors.append(f"{filename}: file is missing")
            continue

        try:
            table = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - exact parser error varies
            errors.append(f"{filename}: could not be read ({exc})")
            continue

        if table.empty:
            errors.append(f"{filename}: table has no rows")

        required = STANDARD_COLUMNS | specific_columns
        missing = sorted(required.difference(table.columns))
        if missing:
            errors.append(f"{filename}: missing columns {missing}")
            continue

        levels = pd.to_numeric(table["evidence_level"], errors="coerce").dropna()
        invalid_levels = sorted(set(levels[~levels.isin(range(5))].tolist()))
        if invalid_levels:
            errors.append(f"{filename}: invalid evidence levels {invalid_levels}")

    if errors:
        details = "\n - ".join(errors)
        raise ValueError(f"Demo output validation failed:\n - {details}")

    return len(REQUIRED_OUTPUTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing CSV files from the demo pipeline.",
    )
    args = parser.parse_args()
    count = validate_demo_outputs(args.results_dir)
    print(f"Validated {count} demo result tables in {args.results_dir.resolve()}.")


if __name__ == "__main__":
    main()
