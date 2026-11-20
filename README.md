# OMIX Exosome Rejuvenation

**A multi-omics reanalysis of aging signals after cell therapy in macaques,
with a comparison to a mouse exosome experiment.**

[![Technical integrity on main](https://github.com/richard2049/Omix-exosome-rejuvenation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/richard2049/Omix-exosome-rejuvenation/actions/workflows/ci.yml)

## What This Project Asks and Why It Matters

After treatment with senescence-resistant cells (SRCs), do age-related changes
in macaques reflect responses within tissues, or could circulating exosomes
(small vesicles released by cells) help explain them? The distinction matters for
which biomarkers to measure and which follow-up experiments to run.

The project uses public OMIX data to build a tissue RNA-based age predictor,
estimate treatment effects at the animal level, and compare exploratory plasma
protein and mouse exosome responses. **The clock tracked chronological age, but
the treatment analysis found no robust evidence of transcriptomic age reversal.**

### Start Here

| To explore | Start with |
|---|---|
| Findings and the data behind them | [Scientific results](RESULTS.md) · [Full-run evidence tables](docs/evidence/2026-09-26/README.md) |
| Study design, methods, and limits | [Study design](docs/study_design.md) · [Inference framework](docs/inference_framework.md) |
| Running and testing the software | [Reproducibility guide](docs/reproducibility.md) · [Automated checks](https://github.com/richard2049/Omix-exosome-rejuvenation/actions/workflows/ci.yml) |

## Evidence at a Glance

Results from the [corrected September 2026 full-data run](docs/evidence/2026-09-26/README.md).

| Evidence layer | Question | Claim status | Current result |
|---|---|---|---|
| Transcriptomic age model | Does tissue RNA predict chronological age? | **Observed** | A Ridge model evaluated on held-out controls gave Spearman `r = 0.87` and animal-balanced mean absolute error of `3.47` years, with a compressed age range. |
| Tissue treatment effects | Do treated tissues show younger-like transcriptomic age? | **Observed, statistically uncertain** | The global SRC-minus-vehicle estimate was `+0.46` years (95% CI `-0.76` to `1.61`). None of the 39 tissue comparisons passed correction for multiple testing. |
| Plasma proteomics | Is the circulating protein state associated with age or intervention? | **Exploratory** | The protein score shifted younger-like after SRC (`-1.69` units versus vehicle). The interval crossed zero, and the difference between the young and old reference groups was uncertain. |
| Macaque methylation | Is there independent epigenetic evidence of rejuvenation? | **Not estimable** | The Mammal40 technical-to-biological sample map is unavailable. |
| Macaque-mouse exosome alignment | Are macaque tissue effects concordant with mouse exosome perturbations? | **Exploratory** | Three organ-compatible pairs give positive directional alignment (`C = 0.73`), but uncertainty is wide and the permutation test is not significant (`p = 0.285`). |
| Plasma-tissue mediation | Can a linked plasma state statistically mediate the transcriptomic response? | **Not estimable** | Plasma and tissue samples lack confirmed matches to the same animals. |
| Causal attribution | Did exosomes cause the macaque rejuvenation response? | **Not established** | The public design lacks direct exosome exposure or cargo-to-recipient outcome linkage. |

`Observed` means estimated, not necessarily statistically supported;
`Exploratory` means hypothesis-generating. `Supported` requires corroborating
evidence; no current mechanism-facing claim meets that threshold.
`Not estimable` denotes a missing design or data requirement; `Not established`
denotes a conclusion the analysis cannot justify. See the
[inference framework](docs/inference_framework.md) for the full definitions.

Across the three primary-compatible cross-species pairs, kidney and muscle are
directionally aligned while liver is discordant. The positive aggregate
alignment therefore motivates a more specific molecular question—what programs
are shared between macaque SRC treatment and mouse SRC-Exo perturbation?—but
does not estimate an exosome-attributable fraction or establish transfer to
macaques.

Read the full interpretation in [RESULTS.md](RESULTS.md).

![Transcriptomic aging and tissue-level treatment effects](docs/assets/aging_rejuvenation_signal.png)

*Figure 1. The controls-only clock is evaluated with animal-grouped nested
cross-fitting before estimating animal-level SRC-minus-vehicle shifts in
`delta_age`. Three of 39 pointwise intervals exclude zero, but none of the
tissue effects passes the primary family-wise correction; displayed tissues
are prioritization candidates, not confirmed rejuvenation effects.*

## Evidence Architecture

The workflow combines scientific analysis with explicit estimability controls:

- A grouped cross-validated transcriptomic aging clock and uncertainty-aware
  tissue summaries.
- Conservative plasma-to-animal linkage with collision, coverage, and
  confidence audits.
- Animal-level mediation that cannot run silently when linkage assumptions
  fail.
- Cross-species exosome-alignment analysis kept separate from direct primate
  mediation.
- Orthogonal validation modules that emit structured limitations instead of
  filling metadata gaps by positional inference.
- A report layer that converts machine-readable result tables into
  interpretation-facing figures without recomputing the analysis.

### Analysis Requirements and Estimability

| Analysis | Design requirement | Current status |
|---|---|---|
| Methylation validation | Biological sample identities | Not estimable: sample map unavailable |
| Macaque age and treatment effects | Animal-grouped prediction and within-tissue comparisons | Estimable, tissue effects uncertain |
| Plasma–tissue association | Confirmed shared animal identities | Not estimable: cross-modal identity remains unconfirmed |
| Cross-species alignment | Compatible tissues and treatment contrasts | Estimable for three organ-compatible pairs; alignment remains exploratory |
| Linked mediation | Linked biological units and causal identification assumptions | Not estimable with current public identity evidence |

This table describes analysis-specific requirements, not a hierarchy of
scientific certainty. An analysis being estimable does not establish effect
significance or causal interpretation.

## Data and Workflow

The primary datasets and supporting public resources are:

- `OMIX007580`: macaque bulk transcriptomics.
- `OMIX007581`: macaque plasma proteomics.
- `OMIX007582`: Mammal40 methylation, currently limited by the missing sample
  map.
- `OMIX009283`: mouse exosome-related mechanism support.
- `OMIX009654` and `OMIX009655`: candidate exosome proteomics and metabolomics,
  audited separately from recipient plasma and awaiting sample labels.
- `OMIX007583`, `OMIX007586`, and `OMIX009284`: targeted validation or audited
  supporting resources, not additional primary cohorts.

The [source audit](docs/data_refresh.md) documents additional public specimen
metadata, article-linked sequencing deposits, and the remaining identity gaps.

The analysis proceeds through auditable, gated stages:

```text
Public OMIX data
  -> metadata, group-label, and identity audits
  -> grouped transcriptomic aging clock
  -> tissue treatment-response summaries
  -> plasma linkage and oriented aging axis
  -> gated animal-level mediation

Mouse exosome perturbation -> cross-species alignment
Validation datasets        -> guarded orthogonal checks
Result CSVs                 -> interpretation-facing report figures
```

![Status-aware multimodal evidence architecture](docs/assets/multimodal_evidence_architecture.png)

*Figure 2. Each modality follows a distinct analysis path before contributing
to interpretation. Estimability gates block unsupported transitions;
plasma-to-tissue identity and Mammal40 biological validation remain unresolved,
while within-plasma summaries remain available. The three-tissue mouse
alignment is exploratory cross-species evidence, not direct macaque validation.*

Group-label interpretation follows
[the canonical crosswalk](docs/group_label_crosswalk.md). A modality-by-modality
summary is available in the [dataset guide](docs/datasets.md). Raw inputs are
treated as immutable; derived tables and figures are written to `results/` and
`figures/`.

## Evidence and Claim Boundaries

A computable result is not necessarily scientifically estimable or biologically
established.

- **Linkage:** cross-modal animal identity must be supported by an explicit,
  auditable mapping. Unresolved plasma samples remain unresolved.
- **Estimability:** linkage-dependent analyses run only when predefined linkage
  and sample-size requirements are met.
- **Mediation:** animal-level mediation is treated as conditional evidence, not
  proof that plasma or exosomes caused rejuvenation.
- **Cross-species evidence:** mouse exosome perturbations can strengthen or
  challenge a mechanistic hypothesis, but they do not establish causality in
  macaques or humans.

![Animal-linkage and mediation estimability guardrail](docs/assets/estimability_guardrail.png)

*Figure 3. The refreshed identity audit retains 24 candidate aliases but no
confirmed cross-modal links. The gate therefore blocks animal-level mediation
and writes a structured non-estimable result. The alternative branch shows
the requirements for a future linked analysis.*

See the [scientific inference and estimability framework](docs/inference_framework.md)
for the complete claim rules.

## Reproduce the Analysis

The public demo checks that the software runs on reduced example data and
produces the expected output tables.

```bash
conda env create -f environment.yml
conda activate srsc
python -m src.run_pipeline --profile demo --safe
python -m src.demo_validation
```

Run the scientific guardrail tests with:

```bash
python -m pytest -q tests/test_scientific_guardrails.py
```

Run the full-data profile against an independently downloaded, read-only data
directory:

```bash
python -m src.run_pipeline --profile full --data-root /path/to/omix-data
python -m src.report_figures
```

Recomputing the reported analyses requires the standard full profile and the
matching source data and configuration. Use a separate checkout to keep demo and
full outputs apart. See the [profile comparison and reproduction checks](docs/reproducibility.md#data-profiles)
for input requirements and the limits of recreating the September run.

## Repository and Documentation

| Path | Purpose |
|---|---|
| `src/` | Analysis, audit, attribution, and reporting modules |
| `tests/` | Automated scientific guardrails and smoke tests |
| `data/PROCESSED/` | Reduced example inputs for the demo profile |
| `results/`, `figures/` | Generated machine-readable outputs and figures |
| `docs/` | Study context, evidence rules, dataset audits, reproducibility, and output reference |

Key reading paths:

- [Documentation index](docs/README.md)
- [Output and figure reference](docs/outputs.md)
- [Citing the work and release provenance](docs/release.md)

Questions, reproducibility problems, and metadata corrections can be reported
through [GitHub Issues](https://github.com/richard2049/Omix-exosome-rejuvenation/issues).

## Current Status and Next Milestone

The core multimodal workflow, scientific guardrails, public demo, and
interpretation-facing report layer are implemented. The corrected full-data
run does not support robust transcriptomic-clock rejuvenation, but it identifies
testable plasma-protein and cross-species response candidates while preserving
their uncertainty and mechanism boundaries.

The next scientific milestone is to obtain or independently validate the
author-side plasma-to-animal identity map and the missing macaque transcript
feature map. These resources would determine whether plasma and tissue data can
support individual-level association and whether macaque molecular programs
can enter gene-, pathway-, and target-level analysis. In parallel, the current
protein and compatible-tissue contrasts can prioritize hypotheses without
pretending to provide a measured exosome-attributable fraction.

## Citation and License

Please cite the repository using the metadata in [CITATION.cff](CITATION.cff).
The source code is distributed under the [MIT License](LICENSE).
