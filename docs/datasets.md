# Dataset Guide

## Purpose

This page summarizes how each public OMIX dataset contributes to the project.
Availability of a modality does not imply that cross-modal or causal analysis
is estimable; those decisions follow the
[inference framework](inference_framework.md).

## Dataset Roles

| Dataset | Organism and modality | Repository role | Current interpretation boundary |
|---|---|---|---|
| `OMIX007580` | Macaque bulk transcriptomics across tissues | Primary age-model and tissue treatment-effect analysis | Released rows currently lack recoverable gene identities; prediction is possible with anonymous variables, but gene, pathway, orthology, mechanism, and target interpretation is blocked |
| `OMIX007581` | Macaque plasma proteomics | Exploratory within-plasma treatment contrasts and a `Y/V` reference-trained age-state axis | Exact upstream normalization is not documented; cross-modal animal integration separately requires an explicit plasma-to-animal key |
| `OMIX007582` | Macaque Mammal40 DNA methylation | Intended orthogonal validation layer | Biological validation is blocked without a technical-to-biological sample map |
| `OMIX007583` | Macaque ovary expression subset | Targeted validation and naming support | Intervention-era `O_V` and stage control `O_C` must remain distinct |
| `OMIX007586` | Macaque hippocampus expression subset | Targeted validation and strongest article-to-OMIX naming bridge | Supports subset validation, not a new independent cohort |
| `OMIX009283` | Mouse exosome perturbation transcriptomics | Cross-species mechanism-supportive alignment | Cannot establish macaque or human exosome causality |
| `OMIX009284` | Macaque PBMC single-cell RNA data | Read-only structural audit and possible future PBMC analysis | PBMC-only scope does not resolve cross-tissue attribution |
| `OMIX009654` | Candidate exosome proteomics; source title conflicts with article/material annotations | Audited additional cargo resource; 4,506 protein rows, six F3/WT identifiers | Preparation and replicate identity require clarification; not a replacement for OMIX007581 |
| `OMIX009655` | Candidate exosome metabolomics; source title conflicts with article | Audited exploratory resource; 64 metabolite rows, 12 anonymous sample columns | Group assignments are unavailable in the matrix; no treatment contrast is run |

## September 2026 refresh

See the [data refresh audit](data_refresh.md) for source URLs, checksums,
matrix overlap, and a reproducible BioSample manifest. PRJCA035748 provides
additional specimen metadata. The detailed manifest covers a focused panel,
with the full project index retained separately; neither establishes a
plasma-to-animal key automatically.

CRA022788, CRA023573, and CRA023595 provide raw sequencing provenance for
macaque tissues, mouse exosome response, and PBMC. Their correspondence to
existing processed OMIX inputs must be reconciled before counting them as
new samples or independent evidence. The FOXO1-LHX1 follow-up is a candidate
mechanistic reference, deferred pending an independence and scope check.

## Data Profiles

- `demo` uses reduced, tracked examples from `data/PROCESSED` and is intended
  for reproducibility and smoke testing.
- `full` uses locally downloaded source data under `data/RAW/data`.
- `auto` selects the full profile when those inputs are present and otherwise
  falls back to the demo profile.

Raw source data are not distributed through Git and must remain immutable.
Generated tables and figures belong in `results/` and `figures/`.

### OMIX007580 release-specific ingestion

The released `OMIX007580-01.txt` has 2,059 sample identifiers in its header
and no leading feature-identifier field. It contains 23,716 structurally
regular count rows plus one malformed physical record at line 22,429. The
full-data profile is bound to the audited source and metadata checksums. It
retains all regular rows, quarantines only that checksum-matched record, and
assigns stable source-line identifiers with `feature_identity=unknown`. Any
different checksum, additional malformed record, non-finite value, negative
value, or non-integer count stops the load.

This repair restores the first sample previously consumed as a row index; it
does not recover gene names. A deliberate standard full-profile rerun using
this corrected 2,059-sample input completed on 26 September 2026. The current
result tables and public interpretation derive from that run; its exact inputs,
configuration, source state, and output inventory are recorded in
`results/run_manifest.json`.

## Reading Dataset Status

Dataset-specific sample counts and estimability can differ by profile. Consult
the generated linkage, audit, and estimability tables rather than treating this
guide as a frozen numerical results report. See the
[output reference](outputs.md) for the relevant files.
