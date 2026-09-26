# Results: What the Public OMIX Data Currently Support

This document reports the corrected September 2026 full-data rerun. The clock,
treatment contrasts, plasma analyses, cross-species response profile,
estimability outputs, and public figures were all regenerated under the current
animal-level and fail-closed linkage contracts. The run is bound to its exact
inputs, configuration, Git state, and outputs in `results/run_manifest.json`.

Historical naming-based plasma–tissue correlations and mediation estimates are
not treated as current results because cross-modal identity remains
unconfirmed. See [the refresh audit](docs/data_refresh.md) for source
reconciliation and [the reproducibility guide](docs/reproducibility.md) for the
current execution contract.

## Executive Interpretation

The controls-only clock captures age-associated expression in held-out animals,
but the corrected SRC-versus-vehicle analysis does not support robust
transcriptomic-clock rejuvenation. The global point estimate is slightly
older-like and uncertain, and no tissue survives the prespecified family-wise
correction.

That negative primary result does not make the multimodal analysis empty.
Within plasma, SRC samples show an uncertain younger-like displacement on a
reference-trained protein axis and many treatment-associated proteins. Across
species, the macaque whole-cell and mouse SRC-Exo tissue responses have a
positive but imprecise directional alignment. Together these results motivate
specific follow-up hypotheses, while neither establishes rejuvenation,
mediation, cargo transfer, or an exosome-attributable fraction.

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

Source: `results/clock_metrics_primates.csv`.

## 2. Tissue Rejuvenation

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

Differences in tissue sample size are represented through uncertainty and the
reported `n_ctrl` and `n_trt`; effects are not rescaled by sample size. The
`signal_to_uncertainty` field is a prioritization aid, not a second effect size.

Sources: `results/rejuvenation_by_tissue.csv` and
`figures/report/report_tissue_rejuvenation_forest.png`.

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

For the primary `GES - V` protein contrast, 1,333 of 3,047 tested accessions
pass within-contrast BH FDR; the 250 highest-ranked candidates subjected to
bootstrap stability testing retain their direction. Strong associations
include lower TLN1 and higher APOC1B, IGFBP2, APOA2, and ALOX15 in GES. The
scale of this signal makes the plasma dataset biologically interesting, but
the groups contain only eight samples each and the exact upstream normalization
of OMIX007581 remains `SOURCE_VERIFICATION_PENDING`. These are exploratory
treatment-associated protein candidates, not validated aging biomarkers or
therapeutic targets.

Sources: `results/linkage_qc_report.csv`,
`results/plasma_age_axis_summary.csv`, and
`results/plasma_age_axis_delta_age_correlation.csv`.

## 4. Linked Mediation

The current linkage tier is `unlinked`: zero treated and zero control animals
have confirmed high-confidence overlap between plasma and bulk tissue data.
The pipeline therefore emits a structured non-estimable mediation record with
reason code `PLASMA_BULK_ANIMAL_LINKAGE_MISSING`.

Earlier naming-based mediation estimates are retained only in repository
history for provenance; they are not current scientific evidence. A validated
plasma-to-animal key would make a statistical mediation analysis technically
possible, but causal interpretation would still require its identifying
assumptions and would not automatically estimate an exosome-attributable
fraction.

## 5. Cross-Species Exosome Alignment

The primary response profile compares three organ-compatible pairs—kidney,
liver, and skeletal muscle—and excludes whole mouse brain versus macaque
hippocampus from the primary summary. Kidney and muscle are directionally
aligned; liver is discordant.

| Component | Estimate | Uncertainty / test |
|---|---:|---|
| Direction cosine (C) | 0.73 | 95% bootstrap CI -0.89 to 0.99; permutation `p = 0.285` |
| Relative response magnitude (R) | 2.56 | 95% bootstrap CI 0.55 to 5.51 |
| Aligned response coefficient (A = C × R) | 1.87 | 95% bootstrap CI -2.01 to 3.09; permutation `p = 0.061` |

The positive point estimates are compatible with partial directional
resemblance, but the intervals are broad, the result depends on only three
tissues, and no permutation test reaches the conventional 0.05 threshold.
Cross-species alignment therefore supports a testable exosome-related
hypothesis, not a robust concordance claim, quantitative transfer estimate, or
causal exosome fraction in macaques.

Sources: `results/cross_species_response_alignment_summary.csv`,
`results/cross_species_response_alignment_by_tissue.csv`, and
`results/exosome_alignment_summary.csv`.

## 6. Orthogonal Validation

Biological methylation validation is not currently estimable. The released
Mammal40 beta matrix uses technical Sentrix identifiers, while the public files
available to this project do not provide the required mapping to biological
sample, tissue, group, sex, and age. The workflow records this as
`OMIX007582_SENTRIX_SAMPLE_SHEET_MISSING` instead of applying an order-based
join.

`OMIX009284` is auditable as PBMC single-cell RNA data, but its PBMC-only scope
does not resolve cross-tissue exosome attribution. It remains a candidate for a
separate immune-cell-state analysis.

Sources: `results/multimodal_concordance_summary.csv`,
`docs/OMIX007582_sample_map_audit.md`, and
`docs/OMIX009284_audit.md`.

## Biological priorities

The corrected observations favor three specific follow-up questions:

1. Can the strongest plasma protein changes be reproduced in an independent
   cohort and interpreted after the upstream normalization is clarified?
2. Once macaque feature identities are recovered, do SRC-responsive genes and
   pathways explain the discordant liver and aligned kidney/muscle responses?
3. Does a directly matched SRC-Exo experiment in primates reproduce the
   cross-species response profile, and can a validated animal key support
   plasma-to-tissue association?

These questions become more concrete with OMIX009654 and OMIX009655, but their
preparation labels must be resolved before treatment contrasts. None requires
creating a new composite causal score. The [author questions](docs/data_refresh.md)
target the feature identities and individual links needed for stronger
molecular and animal-level inference.
