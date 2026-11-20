# Documentation

This directory contains the scientific, methodological, and reproducibility
documentation for the project. The root [README](../README.md) is the concise
project overview, while [RESULTS](../RESULTS.md) presents the scientific story.

## Study and Data

| Page | Purpose |
|---|---|
| [Study design](study_design.md) | Source study, cohorts, interventions, endpoints, and known ambiguities |
| [Dataset guide](datasets.md) | Role, modality, organism, and current use of each OMIX dataset |
| [Group-label crosswalk](group_label_crosswalk.md) | Article-to-OMIX cohort naming and confidence |
| [Scientific scope](scientific_scope.md) | Questions prioritized by the repository and rules for adding analyses |

## Evidence and Interpretation

| Page | Purpose |
|---|---|
| [Inference and estimability framework](inference_framework.md) | Canonical rules for linkage, estimability, mediation, and evidence states |
| [Data linkage](data_linkage.md) | Biological units, linkage artifacts, and permitted downstream uses |
| [Multimodal validation](multimodal_validation.md) | How transcriptomic, plasma, methylation, and mouse evidence are compared without causal overreach |
| [Scientific limitations](scientific_limitations.md) | Public-data ceiling and metadata needed for stronger inference |

## Reproducibility and Outputs

| Page | Purpose |
|---|---|
| [Reproducibility](reproducibility.md) | Environments, data profiles, validation commands, and troubleshooting |
| [Outputs](outputs.md) | Five output families, complete result inventory, report figures, and interpretation boundaries |
| [Figure provenance](assets/README.md) | Sources and release rules for the three curated README figures |
| [September 2026 evidence snapshot](evidence/2026-09-26/README.md) | Selected full-profile results, checksums, and recorded run settings |
| [Citation and release](release.md) | How to cite the software and prepare a versioned scientific release |

## Dataset Audits

- [OMIX007582 Mammal40 sample-map audit](OMIX007582_sample_map_audit.md)
- [OMIX009284 PBMC single-cell audit](OMIX009284_audit.md)

The audits remain separate because they preserve dataset-specific provenance
and explain why particular analyses are available, exploratory, or blocked.
