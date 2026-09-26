# Outputs

The pipeline writes machine-readable analysis tables to `results/`, diagnostic
plots to `figures/`, and interpretation-facing plots to `figures/report/`.

## Output Families

| Family | Main purpose | Start with |
|---|---|---|
| Clock | Evaluate chronological-age prediction under animal-grouped validation | `clock_metrics_primates.csv` |
| Rejuvenation | Estimate animal-aware, uncertainty-aware treatment contrasts | `rejuvenation_by_tissue.csv`; `global_rejuvenation_summary.csv` |
| Multimodal validation | Assess orthogonal methylation and cross-species alignment evidence | `multimodal_concordance_summary.csv`; `exosome_alignment_summary.csv` |
| Linkage and estimability | Establish whether animal-level cross-modal analyses may be interpreted | `estimability_report.csv`; `linkage_qc_report.csv` |
| Sensitivity | Test tissue/age/sex adjustment and feature-threshold dependence | `global_rejuvenation_clustered_sensitivity.csv`; `sensitivity_summary.csv` |

The sections below provide the complete file-level reference. These families
organize outputs by scientific question; they are not evidence-strength ranks.

## Primary Scientific Outputs

| Output | Purpose | Interpretation boundary |
|---|---|---|
| `clock_metrics_primates.csv` | Cross-validated transcriptomic clock performance | Model validation, not treatment evidence |
| `clock_cv_predictions_primates.csv` | Cross-fitted predictions for the historical and three frozen diagnostic partitions | Only the historical partition supplies the primary downstream residuals |
| `clock_fit_audit_primates.csv` | Outer-fold membership, training composition, selected alpha, and leakage checks | Every training fold must contain controls only and zero overlapping animals |
| `clock_hyperparameter_audit_primates.csv` | Inner-fold validity and animal-balanced MAE for every frozen Ridge candidate | Boundary selection is retained and flagged, not described as a proven optimum |
| `clock_exclusions_primates.csv` | Explicit sample-level clock exclusion ledger | Exclusion from modeling does not delete the immutable source record |
| `rejuvenation_by_tissue.csv` | Animal-level tissue contrasts in `delta_age`: primary `SRC - vehicle`, secondary `WTC - vehicle` and `SRC - WTC` | Pointwise bootstrap intervals and multiplicity-adjusted permutation p-values are distinct; both operate on animals |
| `global_rejuvenation_summary.csv` | Secondary across-tissue contrasts after averaging observations within each animal | Equal animal weight does not turn the clock endpoint into proof of biological rejuvenation |
| `global_rejuvenation_clustered_sensitivity.csv` | Tissue-, age-, and sex-adjusted sensitivity with covariance clustered by animal | Model-dependent sensitivity, not the primary estimator |
| `tissue_expression_effects.csv` | Historical filename for covariate-adjusted, per-tissue effects on `delta_age` for `SRC - vehicle` | This is a clock-outcome model, not gene-level differential expression |
| `plasma_biomarkers.csv` | Accession-preserving plasma protein effects for primary `GES - V` and secondary `WT - V` / `GES - WT` contrasts | Exploratory within-plasma treatment associations; not validated rejuvenation biomarkers or therapeutic targets |
| `plasma_age_axis_summary.csv` | Plasma PC1 learned from young and vehicle references, oriented older-like, then projected onto WTC/SRC samples | Age-state axis, not a biological clock or causal mediator |
| `plasma_age_axis_delta_age_correlation.csv` | Linked plasma-axis versus tissue `delta_age` association | Conditional on animal linkage and sample size |
| `cross_species_tissue_mapping.csv` | Run-specific copy of the tiered mouse-to-macaque anatomical contract | `organ_compatible` is not exact anatomical equivalence; `weak_context` cannot enter the primary summary |
| `exosome_alignment_by_tissue.csv` | Pairwise macaque/mouse alignment with anatomical tier and analysis scope | Cross-species mechanism support only; weak-context rows are sensitivities |
| `exosome_alignment_summary.csv` | Contrast-level summary over primary-compatible tissue pairs only | Does not include `brain` versus `Hippocampus` and is not a headline causal or attribution fraction |
| `cross_species_response_alignment_by_tissue.csv` | Hedges' g effects for each primary-compatible macaque/mouse tissue pair | Standardized response comparison; not gene-level concordance or exact anatomical equivalence |
| `cross_species_response_alignment_summary.csv` | Direction (C), relative magnitude (R), and aligned response component (A), with resampling and leave-one-tissue-out diagnostics | Exploratory cross-species resemblance; A is not a percentage, mediation effect, or causal fraction |
| `mediation_summary.csv` | Animal-level mediation estimates or structured stub | Causal interpretation requires stable estimates and assumptions |
| `multimodal_concordance_summary.csv` | Transcriptome/methylation validation status | Blocked when biological sample mapping is unavailable |

`clock_metrics_primates.csv` records the reference and evaluation sample
counts, animal-balanced primary metrics, sample-weighted secondary metrics,
input representation, feature count, Ridge alpha, boundary flag, and solver.
The full protocol uses float64 model inputs after `log2(CPM)`, with variance
filtering and standardization fitted independently inside each training fold.
This prevents held-out animals from influencing preprocessing.

`plasma_biomarkers.csv` retains the historical filename for output
compatibility, but no longer contains the arbitrary ordinal
`V=-1/Y=0/WT=1/GES=2` ranking. Abundances are transformed with `log2` and
analysed using sex-adjusted pairwise OLS with HC3 covariance. Benjamini-Hochberg
FDR is calculated separately for each prespecified contrast. Protein accession
is the stable feature identifier; gene symbol is annotation because symbols are
not unique. The precise upstream normalization of `OMIX007581` remains
`SOURCE_VERIFICATION_PENDING`.

For estimable mediation, `Total`, `ADE`, and `ACME` are the canonical effect
fields and their `*_CI` columns contain effect-specific bootstrap intervals.
The common `ci_low` and `ci_high` fields repeat the `Total_CI` bounds so the
table retains the repository-wide result contract. ADE and ACME must not be
relabeled as cellular and exosome-causal effects without stronger identifying
evidence.

The response-alignment summary deliberately keeps three components visible.
`cosine_similarity` compares direction, `relative_response_norm` compares the
overall standardized magnitude, and `aligned_response_coefficient` equals their
product. The last quantity measures how much of the mouse response vector lies
along the macaque response direction; it does not estimate how much of the
macaque treatment effect was caused by exosomes. Its uncertainty is summarized
by animal/sample bootstrap intervals, treatment-label permutations, and
leave-one-tissue-out ranges.

## Linkage and Estimability Diagnostics

| Output | Purpose |
|---|---|
| `run_manifest.json` | Run status, command, environment, configuration, input checksums, dirty-source fingerprint, and output inventory |
| `plasma_to_animal_map.csv` | Mapping provenance, confidence, and validity per plasma sample |
| `plasma_linkage_manifest_audit.csv` | Checksum, schema, one-to-one identity, provenance, group, and sex validation for an explicitly confirmed linkage key |
| `linkage_qc_report.csv` | Coverage and collision diagnostics |
| `linkage_audit.csv` | Bulk/plasma animal-overlap diagnostics |
| `estimability_report.csv` | Gate for mediation and linked decomposition |
| `omix007580_input_audit.csv` | Version, retained dimensions, quarantine, and anonymous-feature status for the full-profile bulk input |

These are required scientific diagnostics. A mechanism-facing result should not
be interpreted without its linkage and estimability context.

## Supporting and Compatibility Outputs

The standalone `src.data_refresh` command writes an audit subdirectory
containing `dataset_inventory.csv`, `project_index.csv`, `sample_manifest.csv`,
`plasma_linkage_candidates.csv`, `shared_gene_annotations.csv`,
`source_inventory.csv`, and `audit_summary.json`. These are source/provenance
tables, not additional estimable biological results. Detailed metadata coverage
and the distinction between candidate aliases and confirmed identities are
explicit. See [the refresh audit](data_refresh.md).

| Output | Purpose |
|---|---|
| `mouse_exosome_effects.csv` | Tissue-level `OMIX009283` effects |
| `mouse_exosome_signature_summary.csv` | Mouse contrast summaries |
| `methylation_rejuvenation_by_tissue.csv` | Methylation validation or structured limitation |
| `ovary_subset_validation.csv` | Targeted ovary subset audit |
| `hippocampus_subset_validation.csv` | Targeted hippocampus subset audit |
| `sensitivity_summary.csv` | Animal-level treatment-contrast and plasma feature-threshold sensitivity |
| `exosome_fraction_summary.csv` | Structured record that the causal exosome-attributable fraction is not identified by the current cross-species design; points readers to the separate quantitative response-alignment profile |
| `omix007582_sample_map_summary.csv` | Mammal40 mapping audit |
| `omix009284_audit_summary.csv` | PBMC single-cell structural audit |

## Standard Result Contract

Interpretation-facing tables use the following fields where applicable:

- `available`: whether the required source data were available.
- `estimable`: whether the configured analysis could be evaluated.
- `reason` and `reason_code`: human- and machine-readable status.
- `missing_author_key`: exact external metadata required when blocked.
- `n_used`: observations used by the analysis.
- `method`: compact method identifier.
- `ci_low` and `ci_high`: uncertainty interval where defined.
- `evidence_level`: repository-specific integer code used for internal reporting
  and compatibility. It is not a standardized scientific evidence grade,
  a statistical significance score, or proof of causality. Interpret it with
  `method`, `estimable`, and the documented requirements for each analysis.
  Standalone within-modality analyses use level 1; level 2 is reserved for
  analyses that use sufficiently supported biological linkage between
  modalities.

## Report Figures

Run `python -m src.report_figures` after the pipeline. The generated
`figures/report/report_figure_manifest.csv` records each figure's source tables,
status, and interpretation note. The report set includes:

- tissue rejuvenation and tissue-priority views;
- exosome-alignment summary and tissue drivers;
- estimability, evidence-ladder, and public-data-ceiling views;
- mediation uncertainty;
- plasma biomarker stability and heuristic categories;
- the oriented plasma aging axis;
- analysis sensitivity summaries.

Three report figures are curated separately for the public README:

| Asset | Source tables | Interpretation boundary |
|---|---|---|
| `docs/assets/aging_rejuvenation_signal.png` | `clock_metrics_primates.csv`; `rejuvenation_by_tissue.csv` | Grouped clock validation plus nominal tissue prioritization; not confirmed tissue effects |
| `docs/assets/multimodal_evidence_architecture.png` | Rejuvenation, linkage, alignment, and multimodal status tables | Modality-to-analysis flow with estimability gates; not a pooled multimodal effect |
| `docs/assets/estimability_guardrail.png` | `linkage_qc_report.csv`; `estimability_report.csv`; `mediation_summary.csv` | A passed gate means estimable, not causally established |

These tracked images are release snapshots. Regenerate them from reviewed
full-profile results with:

```bash
python -m src.report_figures --portfolio-assets-dir docs/assets
```

Protein categories are heuristic symbol groupings unless a dedicated annotation
field is supplied; they are not formal pathway-enrichment results.
