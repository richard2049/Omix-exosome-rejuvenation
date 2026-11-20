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

- `demo` uses synthetic, tracked examples from `data/PROCESSED` for software
  smoke testing.
- `full` uses locally downloaded source data under `data/RAW/data`.
- `auto` selects the full profile when those inputs are present and otherwise
  falls back to the demo profile.

Obtain full source datasets separately and keep them read-only. Generated
tables and figures belong in `results/` and `figures/`.

### Demo inputs and redistribution

The five bundled demo inputs use generated values and fictional identifiers.
They exercise the readers and output contracts, including the unresolved
methylation sample-map path. To regenerate the inputs from the repository root:

```bash
python -m src.generate_demo_inputs --output-dir data/PROCESSED
```

The generator uses a fixed seed and reads no source datasets. The historical
examples listed below precede these synthetic inputs. A request to OMIX for
written confirmation about redistributing those excerpts is awaiting a
response.

Credit the study to [Lei et al., *Cell* (2025)](https://doi.org/10.1016/j.cell.2025.05.021)
and cite each OMIX accession for the corresponding source data.

| Historical demo file | Source and verified selection | Public redistribution |
|---|---|---|
| `OMIX007581-01_example.csv` | First 101 data rows of the deposited `OMIX007581-01.csv`, with all 34 columns. [OMIX007581](https://ngdc.cncb.ac.cn/omix/release/OMIX007581) | Pending verification |
| `OMIX007582_beta_matrix_example.csv` | Values for 99 CpG sites and 10 technical sample columns match the locally prepared `OMIX007582_beta_matrix.csv`. The source accession is cited in the study's data availability statement. [OMIX007582](https://ngdc.cncb.ac.cn/omix/release/OMIX007582) | Pending verification |

The comparison used the demo files at commit `da62d6b` and the source files
identified by SHA-256 in the [September full-run manifest](evidence/2026-09-26/manifest.json).

On 20 November 2026, both accession pages listed open access, and OMIX described
free academic use in its [licensing guidance](https://ngdc.cncb.ac.cn/omix/docs?lang=en_GB).
The current demo and release candidate use synthetic inputs while the request
is pending. If permission is confirmed, record the applicable licence or written
confirmation from OMIX or the rights holder, the exact files it covers, and
any attribution or use conditions. The repository's [MIT licence](../LICENSE)
applies to the project software. See the [ELIXIR guidance on third-party data
licensing](https://rdmkit.elixir-europe.org/licensing).

### Inputs for the September full run

The [full-run manifest](evidence/2026-09-26/manifest.json) lists the exact
input filenames and SHA-256 values. Obtain source files from the corresponding
OMIX accessions. The archived run also used a prepared
`OMIX007582_beta_matrix.csv` and a two-column `OMIX009283_metadata.csv`.
The current [mouse metadata helper](../src/omix009283_metadata.py) parses the
same sample identifiers into a different column schema. Verify prepared
files against the manifest before claiming an exact input match. The
[Mammal40 audit](OMIX007582_sample_map_audit.md) describes the methylation files
and the unresolved biological sample map.

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
result tables and public interpretation derive from that run. Its aggregate
results and sanitized provenance are in the [September evidence snapshot](evidence/2026-09-26/README.md).

## Reading Dataset Status

Dataset-specific sample counts and estimability can differ by profile. Consult
the generated linkage, audit, and estimability tables rather than treating this
guide as a frozen numerical results report. See the
[output reference](outputs.md) for the relevant files.
