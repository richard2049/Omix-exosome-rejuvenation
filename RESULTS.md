# Results: What the Public OMIX Data Currently Support

These results come from the corrected September 2026 full-profile run. The
[evidence snapshot](docs/evidence/2026-09-26/README.md) contains the aggregate
tables and run provenance. The [source audit](docs/data_refresh.md) documents
how public records were reconciled.

## Executive Interpretation

The controls-only clock tracked chronological age in held-out animals. The
SRC-versus-vehicle estimate was slightly older-like and uncertain. None of
the 39 tissues met the prespecified family-wise threshold.

The plasma protein score shifted toward the young reference, and the macaque
SRC and mouse SRC-exosome responses showed some directional agreement. Both
estimates were imprecise and remain exploratory. The available data leave
mediation, cargo transfer, and the fraction attributable to exosomes unresolved.

## 1. Transcriptomic Aging Signal

The Ridge transcriptomic clock was trained only on untreated reference groups
and evaluated by nested, animal-grouped cross-fitting. The reference set
contains 1,499 tissue samples from 44 animals; predictions cover 2,058 samples
from 60 animals after one unresolved sample alias is excluded.

| Metric | Corrected full-profile value | Interpretation |
|---|---:|---|
| Animal-balanced MAE | 3.47 years | Mean absolute error after giving each held-out control animal equal weight |
| Animal-balanced RMSE | 3.99 years | Error with greater weight on large deviations |
| Pearson correlation | 0.83 | Positive linear age association |
| Spearman correlation | 0.87 | Strong rank-order age association |
| Calibration slope | 0.49 | Substantial compression of predictions toward the mean age |

Across the historical and three frozen diagnostic partitions, all 20 outer
fits had zero animal overlap and zero treated samples in training. Every fold
selected the weakest prespecified Ridge penalty (`alpha = 0.0001`). This
supports use of the model as an age-associated predictive reference, while the
boundary selection and calibration compression remain important limitations.
It does not, by itself, prove biological rejuvenation or establish causal
treatment effects.

Evidence from the [clock metrics](docs/evidence/2026-09-26/clock_metrics_primates.csv)
and [fold audit](docs/evidence/2026-09-26/clock_fit_audit_primates.csv).

## 2. Tissue Treatment Contrasts

The tissue analysis estimates animal-level treated-versus-control differences
in cross-fitted `delta_age`. The global contrasts are uncertain:

| Contrast | Mean effect (years) | 95% pointwise bootstrap CI | Permutation p-value |
|---|---:|---:|---:|
| SRC minus vehicle | +0.46 | -0.76 to 1.61 | 0.476 |
| WTC minus vehicle | +0.73 | -0.51 to 1.86 | 0.313 |
| SRC minus WTC | -0.27 | -1.17 to 0.59 | 0.617 |

Among 39 primary tissue contrasts, seven point younger-like and 32 older-like.
Pointwise intervals exclude zero for hippocampus and pancreas in the older-like
direction and `Trachea_4` in the younger-like direction. Only hippocampus has a
nominal permutation `p < 0.05`, and no tissue passes the prespecified primary
family-wise correction. These are prioritization signals, not confirmed
tissue-specific rejuvenation or aging effects.

The table reports the numbers of control and treated animals for each tissue.
For follow-up, `signal_to_uncertainty` divides the effect magnitude by the
confidence interval width.

Evidence from the [global contrasts](docs/evidence/2026-09-26/global_rejuvenation_summary.csv)
and [tissue contrasts](docs/evidence/2026-09-26/rejuvenation_by_tissue.csv).

## 3. Plasma State and Linkage

The naming rule produces 24 candidate aliases among 32 plasma samples, but none
is an author-confirmed cross-modal identity. The corrected run therefore has
zero high-confidence overlapping animals and withholds plasma-to-tissue
correlation.

The reference-trained plasma PC1 explains 50.4% of the selected protein
variance. Relative to vehicle, SRC has a younger-like median displacement of
`-1.69` axis units (95% CI `-4.20` to `0.56`; permutation `p = 0.068`). The
old-versus-young reference gap itself has an interval crossing zero, so this
directional signal is exploratory and its ratio to the reference gap is not a
rejuvenation fraction.

In the primary `GES - V` contrast, 1,333 of 3,047 protein accessions met the
Benjamini–Hochberg false discovery rate threshold for that contrast. All 250
candidates selected for bootstrap
stability testing retained the direction of their estimated difference.
Associations included lower TLN1 and higher APOC1B, IGFBP2, APOA2, and ALOX15 in
GES. Each group contains eight samples, and the original normalization of
OMIX007581 remains unverified (`SOURCE_VERIFICATION_PENDING`). The protein
associations remain exploratory.

Evidence from the [linkage audit](docs/evidence/2026-09-26/linkage_qc_report.csv),
[plasma axis summary](docs/evidence/2026-09-26/plasma_age_axis_summary.csv),
[correlation status](docs/evidence/2026-09-26/plasma_age_axis_delta_age_correlation.csv),
and [primary protein contrast summary](docs/evidence/2026-09-26/plasma_primary_contrast_summary.json).

## 4. Linked Mediation

The [linkage check](docs/evidence/2026-09-26/estimability_report.csv) reports
zero confirmed animal links and records `PLASMA_BULK_ANIMAL_LINKAGE_MISSING`.
The [mediation output](docs/evidence/2026-09-26/mediation_summary.csv) records
`CONFIG_DISABLED`, the default setting used for the full run.

Earlier estimates based on sample names are retained in the repository history.
A validated animal key would let the workflow assess whether mediation is
estimable. A causal interpretation would still require the assumptions in
the [inference framework](docs/inference_framework.md).

## 5. Cross-Species Exosome Alignment

The primary response profile compares three organ-compatible pairs—kidney,
liver, and skeletal muscle—and excludes whole mouse brain versus macaque
hippocampus from the primary summary. Kidney and muscle point in the same
direction across species. Liver points in opposite directions.

| Component | Estimate | Uncertainty / test |
|---|---:|---|
| Direction cosine (C) | 0.73 | 95% bootstrap CI -0.89 to 0.99; permutation `p = 0.285` |
| Relative response magnitude (R) | 2.56 | 95% bootstrap CI 0.55 to 5.51 |
| Aligned response coefficient (A = C × R) | 1.87 | 95% bootstrap CI -2.01 to 3.09; permutation `p = 0.061` |

The direction and alignment intervals both cross zero. The comparison
generates a hypothesis about shared responses across the three tissues.

Evidence from the [response profile](docs/evidence/2026-09-26/cross_species_response_alignment_summary.csv),
[three tissue pairs](docs/evidence/2026-09-26/cross_species_response_alignment_by_tissue.csv),
and [causal attribution status](docs/evidence/2026-09-26/exosome_fraction_summary.csv).

## 6. Orthogonal Validation

Methylation validation requires a map from the Mammal40 matrix's technical
Sentrix identifiers to biological sample, tissue, group, sex, and age.
The map is unavailable, so the workflow records
`OMIX007582_SENTRIX_SAMPLE_SHEET_MISSING`.

`OMIX009284` is auditable as PBMC single-cell RNA data, but its PBMC-only scope
does not resolve cross-tissue exosome attribution. It remains a candidate for a
separate immune-cell-state analysis.

See the [concordance status](docs/evidence/2026-09-26/multimodal_concordance_summary.csv),
[Mammal40 sample-map audit](docs/OMIX007582_sample_map_audit.md), and
[PBMC dataset audit](docs/OMIX009284_audit.md).

## Biological priorities

The corrected observations favor three specific follow-up questions:

1. Can the strongest plasma protein changes be reproduced in an independent
   cohort and interpreted after the upstream normalization is clarified?
2. Once macaque feature identities are recovered, do SRC-responsive genes and
   pathways explain the discordant liver and aligned kidney/muscle responses?
3. Does a directly matched SRC-Exo experiment in primates reproduce the
   cross-species response profile, and can a validated animal key support
   plasma-to-tissue association?

Treatment contrasts for OMIX009654 and OMIX009655 require verified preparation
labels. The [source audit](docs/data_refresh.md) lists the missing feature
identities, preparation labels, and animal-level links.
