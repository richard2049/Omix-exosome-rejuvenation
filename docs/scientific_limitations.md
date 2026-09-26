# Public Data Limitations and Evidence Ceiling

## Purpose

This page defines which conclusions can be supported by the public OMIX
data used in this repository and which require additional metadata or new
experimental evidence. Its purpose is to prevent additional model complexity
from being mistaken for stronger biological identification.

## Defensible Uses of the Public Data

The guarded workflow can support:

- evaluation of a macaque transcriptomic age model and uncertainty-aware
  tissue treatment effects;
- covariate-aware tissue expression analyses, subject to released metadata;
- exploratory within-plasma treatment contrasts and separate plasma-to-animal linkage audits;
- animal-level mediation as an exploratory analysis when its linkage and
  sample-support gate passes;
- mouse `OMIX009283` comparisons as cross-species, non-causal mechanism
  support or contradiction. The primary summary is restricted to explicitly
  classified organ-compatible pairs; whole mouse brain versus macaque
  hippocampus is retained only as a weak-context sensitivity;
- a quantitative cross-species response profile based on standardized tissue
  effects: direction, relative magnitude, and the mouse component aligned with
  the macaque pattern, with bootstrap, permutation, and leave-one-tissue-out
  diagnostics;
- technical audits of `OMIX007582` and `OMIX009284` with explicit structured
  limitations.

These outputs support reproduction, falsification, and hypothesis
prioritization. They do not establish that exosomes caused the macaque tissue
response.

This distinction still permits biologically informative conclusions. If the
three response components are directionally consistent and stable to
resampling and tissue omission, they strengthen the hypothesis that the mouse
exosome perturbation reproduces part of the tissue-level pattern associated
with macaque SRC treatment. If they are weak, opposite, or unstable, they
provide evidence against a simple shared-response account. Neither outcome is
a causal percentage, and the conclusion remains conditional on cross-species
tissue compatibility and the clock-derived endpoint.

## Claims Not Established

The current public study design does not establish:

- a numeric causal partition into exosome-derived and tissue-intrinsic effects;
- exosomes as the principal cause of macaque rejuvenation;
- causal FOXO3A necessity or sufficiency;
- biological methylation concordance from `OMIX007582`, because the Sentrix
  technical identifiers lack a defensible biological sample map;
- a cross-tissue mechanism from the PBMC-only `OMIX009284` release.

## Metadata Required for Stronger Inference

The [September 2026 refresh](data_refresh.md) improves public specimen
provenance and identifies additional exosome cargo matrices. These are useful
inputs to a biological comparison of SRC and WTC responses. They do not supply
the missing cross-modal identity keys. The current limitation is specific
identity and preparation metadata, not a general absence of public metadata.

Also request the F3/WT preparation and replicate labels for OMIX009654 and the
sample-1...12 assignments for OMIX009655. Gene-annotation overlap between cargo
and plasma provides candidates to test, not evidence of cargo transfer.

The following author-side records would materially increase evidentiary
strength:

- a complete `OMIX007581` plasma proteomics
  `sample_id -> animal_id -> group -> sex -> age` mapping;
- confirmation of the `OMIX007580` bulk RNA-seq
  `sample_id -> animal_id -> tissue -> group -> sex -> age` mapping;
- an author- or repository-supplied ordered feature map for the released
  `OMIX007580-01` rows. Until then, its anonymous variables support bounded
  prediction but not gene, pathway, orthology, mechanism, or target claims;
- exosome cargo, preparation, donor, batch, dose, treatment, recipient, and
  outcome linkage;
- the `OMIX007582` Sentrix barcode and position to biological sample, tissue,
  group, sex, and age mapping;
- assay-specific QC and exclusion tables;
- direct exosome uptake, biodistribution, or cargo-transfer measurements linked
  to recipient animals and tissues.

## Analysis-Specific Inference Requirements

- Transcriptomic age prediction requires animal-grouped validation; tissue
  treatment effects require comparisons and uncertainty estimates that respect
  the biological unit and repeated measurements. The primary intervention
  contrast is `SRC - vehicle`; age-stage controls are not pooled with the
  intervention-era vehicle arm. Across-tissue summaries are secondary and
  reduce repeated tissue observations to one mean per animal before inference.
  Their intervals remain conditional on the fitted cross-validated clock and
  do not propagate every source of clock-selection uncertainty.
- Plasma–tissue associations require confirmed shared animal identities and
  adequate sample support. Within-plasma summaries do not require those
  cross-modal links.
- Cross-species alignment requires explicit treatment contrasts and compatible
  tissues, with species-aware feature matching when comparing molecular
  programs. The primary response profile requires at least three valid
  organ-compatible tissue pairs. Its aligned-response coefficient is a vector
  projection and may be negative or exceed 1; it must not be formatted or
  interpreted as an exosome-attributable percentage.
- The controls-only Ridge clock remains a provisional predictive reference,
  not a validated measure of rejuvenation. Its frozen primary benchmark chose
  the weakest tested penalty in every outer fold, so the boundary condition,
  age-calibration bias, partition sensitivity, and pending representation/age
  sensitivities must remain visible.
- Animal-level mediation requires valid linkage and adequate sample support;
  causal interpretation additionally requires a study design and explicit
  assumptions that identify the relevant causal effects.

These are analysis-specific requirements, not a hierarchy of scientific
certainty. Meeting computational requirements does not establish effect
significance or causality. A computable mediation estimate alone does not
establish an exosome-attributable fraction; that causal quantity remains
unidentified even when the separate response-alignment profile is precise.

## Operational Rule

When an analysis depends on unavailable information, the pipeline must emit a
structured non-estimable row with a stable `reason_code` and the exact external
requirement in `missing_author_key`, or label the result exploratory and state
the limitation. Missing biological identity must not be replaced by positional
joins, row order, group matching, or cross-species substitution.

If an author-validated plasma-to-animal key becomes available, it must be
preserved as a checksum-pinned manifest with explicit provenance and per-row
identity confirmation. A generic metadata table or a syntactic match between
sample aliases is not sufficient evidence of shared biological identity.
