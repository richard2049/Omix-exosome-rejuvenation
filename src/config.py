"""
config.py

Typed configuration objects and defaults for the OMIX Exosome Rejuvenation pipeline, including
dataset paths, metadata-column candidates, and analysis hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class OmixPaths:
    """Container for OMIX matrix + metadata paths."""
    matrix: Path
    metadata: Optional[Path] = None


@dataclass
class PipelineConfig:
    """
    Configuration for the OMIX Exosome Rejuvenation pipeline.

    The pipeline operates on released OMIX-derived matrices and metadata,
    while keeping linkage-sensitive and attribution-sensitive analyses
    behind explicit estimability gates.
    """

    # Core OMIX
    primate_bulk: OmixPaths
    data_profile: str = "auto"
    data_root: Optional[Path] = None
    primate_bulk_input_mode: str = "feature_indexed"
    primate_bulk_representation: str = "log1p_counts"
    primate_bulk_age_col: str = "age"
    primate_plasma: Optional[OmixPaths] = None
    plasma_linkage_manifest: Optional[Path] = None
    plasma_linkage_manifest_sha256: Optional[str] = None
    linkage_high_conf_values: Optional[List[str]] = None
    primate_methylation: Optional[OmixPaths] = None
    mouse_exosome_bulk: Optional[OmixPaths] = None
    max_allowed_samples: int = 5000
    min_samples_for_mediation: int = 12
    min_samples_per_group_for_rejuv: int = 2
    mediation_bootstrap: int = 500
    clock_model: str = "ridge"
    clock_protocol: str = "legacy_all_samples_ridge"
    clock_cv_folds: int = 5
    clock_inner_folds: int = 3
    clock_inner_seed: int = 20260917
    clock_reference_groups: Optional[List[str]] = None
    clock_ridge_alphas: Optional[List[float]] = None
    clock_diagnostic_seeds: Optional[List[int]] = None
    clock_excluded_sample_ids: Optional[List[str]] = None
    clock_fold_assignments: Optional[Path] = None
    clock_fold_assignments_sha256: Optional[str] = None
    control_label: str = "Control"
    primate_treated_label: str = "O_GES"
    primate_vehicle_label: str = "O_V"
    primate_wtc_label: str = "O_WT"
    primate_control_labels: Optional[List[str]] = None
    primate_treatment_contrasts: Optional[List[Tuple[str, str, str]]] = None
    primate_primary_contrast: str = "SRC_vs_vehicle"
    top_genes_per_tissue: int = 100
    top_plasma_biomarkers: int = 50
    random_seed: int = 42
    focus_genes: tuple[str, ...] = ("FOXO3", "SRC")
    tissue_weighting: str = "uniform"
    exosome_fraction_method: str = "correlation_ratio"
    exosome_fraction_bootstrap: int = 2000
    exosome_fraction_permutations: int = 1000
    exosome_min_common_tissues: int = 3
    exosome_min_cells_median_abs: float = 1e-8
    response_alignment_bootstrap: int = 1000
    response_alignment_permutations: int = 1000
    # The public plasma aliases are candidates until cross-modal identity is confirmed.
    enable_mediation: bool = False
    enable_causal_decomposition: bool = True
    # Optional modules flags
    enable_methylation_block: bool = True
    enable_mouse_exosome_block: bool = True
    enable_subset_validation_block: bool = True
    enable_translation_module: bool = False

    # Mouse exosome alignment
    mouse_reference_arms: Optional[List[str]] = None
    mouse_contrasts: Optional[List[str]] = None
    mouse_min_reference_samples: int = 20
    mouse_min_reference_ages: int = 4
    mouse_min_samples_per_group: int = 3
    mouse_tissue_bootstrap: int = 1000
    mouse_tissue_permutations: int = 1000
    mouse_alignment_contrasts: Optional[List[str]] = None
    mouse_to_primate_tissue_map: Optional[Dict[str, str]] = None
    mouse_tissue_mapping_path: Optional[Path] = Path("config/cross_species_tissue_map.csv")

    # Methylation validation
    methylation_group_age_map: Optional[Dict[str, float]] = None
    methylation_to_primate_tissue_map: Optional[Dict[str, str]] = None
    methylation_min_common_tissues: int = 3

    # Column candidates for auto-detection
    sample_id_col_candidates: Optional[List[str]] = None
    group_col_candidates: Optional[List[str]] = None
    tissue_col_candidates: Optional[List[str]] = None
    age_col_candidates: Optional[List[str]] = None
    sex_col_candidates: Optional[List[str]] = None
    animal_id_col_candidates: Optional[List[str]] = None

    # Labels
    mouse_treated_label: str = "GES"
    mouse_control_labels: Optional[List[str]] = None

    # Feature selection
    n_top_features_expr: int = 0
    n_top_features_proxy: int = 1000

    # Plasma score
    n_top_plasma_features: int = 50
    enable_plasma_age_axis: bool = True
    plasma_axis_young_labels: Optional[List[str]] = None
    plasma_axis_old_control_labels: Optional[List[str]] = None
    plasma_axis_min_group_samples: int = 2
    plasma_axis_min_non_nan_frac: float = 0.8
    plasma_axis_min_linked_animals: int = 8
    sensitivity_top_feature_thresholds: Optional[List[int]] = None
    plasma_treatment_contrasts: Optional[List[Tuple[str, str, str]]] = None
    plasma_primary_contrast: str = "GES_vs_V"
    plasma_contrast_min_per_group: int = 3
    plasma_biomarker_min_pairs: int = 8
    plasma_biomarker_bootstrap: int = 120
    plasma_biomarker_sign_agreement_min: float = 0.8
    plasma_biomarker_stability_top_k: int = 250

    # Statistics
    random_state: int = 42
    n_bootstrap: int = 2000
    treatment_n_permutations: int = 2000

    # Output
    results_dir: Path = Path("results")
    figures_dir: Path = Path("figures")

    # Optional gene sets
    gmt_path: Optional[Path] = None

    # Plasma ranking plot
    plasma_ranking_requires_spearman: bool = False
    plasma_ranking_fallback_metric: str = "variance"  # "variance" | "abs_mean" | "abs_diff"
    tissue_effect_covariates: Optional[List[str]] = None

    def __post_init__(self):
        self.sample_id_col_candidates = self.sample_id_col_candidates or [
            "sample_id", "Sample", "sample", "SampleID", "sample_name"
        ]
        self.group_col_candidates = self.group_col_candidates or [
            "group", "Group", "treatment", "Treatment", "condition", "Condition"
        ]
        self.tissue_col_candidates = self.tissue_col_candidates or [
            "tissue", "Tissue", "organ", "Organ"
        ]
        self.age_col_candidates = self.age_col_candidates or [
            "agenumb",
            "age_num",
            "age_years",
            "Age (years)",
            "Age(years)",
            "age",
            "Age",
            "chrono_age",
            "chronological_age",
        ]
        self.sex_col_candidates = self.sex_col_candidates or [
            "sex", "Sex", "gender", "Gender"
        ]
        self.animal_id_col_candidates = self.animal_id_col_candidates or [
            "animal_id",
            "AnimalID",
            "donor_id",
            "Donor",
            "subject_id",
            "Subject",
            "orig.ident",
            "orig_ident",
            "orig.ident.id",
        ]

        self.primate_control_labels = self.primate_control_labels or [
            "Y_C", "M_C", "O_C", "O_V"
        ]
        self.primate_treatment_contrasts = self.primate_treatment_contrasts or [
            ("SRC_vs_vehicle", "O_GES", "O_V"),
            ("WTC_vs_vehicle", "O_WT", "O_V"),
            ("SRC_vs_WTC", "O_GES", "O_WT"),
        ]
        self.clock_reference_groups = self.clock_reference_groups or [
            "Y_C", "M_C", "O_C", "O_V"
        ]
        self.clock_ridge_alphas = self.clock_ridge_alphas or [
            0.0001,
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
            1000.0,
            10000.0,
            100000.0,
            1000000.0,
        ]
        self.clock_diagnostic_seeds = self.clock_diagnostic_seeds or [17, 29, 43]
        self.clock_excluded_sample_ids = self.clock_excluded_sample_ids or []
        self.plasma_axis_young_labels = self.plasma_axis_young_labels or ["Y", "Y_C"]
        self.linkage_high_conf_values = self.linkage_high_conf_values or [
            "high",
            "metadata_exact",
        ]
        normalized_linkage_confidence = {
            str(value).strip().lower() for value in self.linkage_high_conf_values
        }
        unsupported_linkage_confidence = normalized_linkage_confidence.difference(
            {"high", "metadata_exact"}
        )
        if unsupported_linkage_confidence:
            raise ValueError(
                "linkage_high_conf_values contains unverified confidence labels: "
                f"{sorted(unsupported_linkage_confidence)}"
            )
        self.linkage_high_conf_values = sorted(normalized_linkage_confidence)
        self.plasma_axis_old_control_labels = self.plasma_axis_old_control_labels or [
            "O_V",
            "V",
        ]
        self.plasma_treatment_contrasts = self.plasma_treatment_contrasts or [
            ("GES_vs_V", "GES", "V"),
            ("WT_vs_V", "WT", "V"),
            ("GES_vs_WT", "GES", "WT"),
        ]
        self.mouse_control_labels = self.mouse_control_labels or [
            "Veh", "WT", "Ctrl", "Baseline"
        ]
        self.tissue_effect_covariates = self.tissue_effect_covariates or ["age", "sex", "batch"]
        self.sensitivity_top_feature_thresholds = self.sensitivity_top_feature_thresholds or [
            20,
            int(self.n_top_plasma_features),
            100,
        ]
        self.mouse_reference_arms = self.mouse_reference_arms or ["Baseline"]
        self.mouse_contrasts = self.mouse_contrasts or ["GES_vs_Veh", "WT_vs_Veh", "GES_vs_WT"]
        self.mouse_alignment_contrasts = self.mouse_alignment_contrasts or ["GES_vs_Veh", "WT_vs_Veh"]
        self.mouse_to_primate_tissue_map = self.mouse_to_primate_tissue_map or {
            "brain": "Hippocampus",
            "kidney": "Renal_cortex",
            "liver": "Liver_L",
            "lung": "Lung_R3_3",
            "muscle": "Quadriceps_muscle",
        }
        self.methylation_group_age_map = self.methylation_group_age_map or {
            "Y_C": 4.5,
            "Y_WT": 4.5,
            "Y_WS": 4.5,
            "M_C": 11.0,
            "O_C": 17.0,
            "O_V": 21.0,
            "O_WT": 21.0,
            "O_GES": 21.0,
            "O_CR": 21.0,
            "O_Met": 21.0,
            "O_VC": 21.0,
        }
        self.methylation_to_primate_tissue_map = self.methylation_to_primate_tissue_map or {
            "Brain": "Hippocampus",
            "Kidney": "Renal_cortex",
            "Liver": "Liver_L",
            "Lung": "Lung_R3_3",
            "Quadriceps muscle": "Quadriceps_muscle",
        }
