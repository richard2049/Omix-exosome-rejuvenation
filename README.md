# SRSC: Cross-species translational analysis of exosome interventions (exploratory v0.1, Phase 1)

> **Status:** exploratory / work in progress (v0.1, Phase 1).  
> This repository is intended as a portfolio / research project in aging and translational bioinformatics.

## Overview

Omix-exosome-rejuvenation is an exploratory pipeline to analyze primate rejuvenation and exosome-mediated effects from public OMIX datasets.

It implements:

- robust data loading / guardrails for OMIX matrices and metadata,
- a lightweight **transcriptomic clock** trained on bulk tissues,
- plasma proteomics–based state scores,
- and a first-pass estimate of the exosome-attributable fraction of rejuvenation.

The code is research-oriented, not an official pipeline, and focuses on transparency and reproducibility rather than definitive biological claims.

This project explores cross-species translational signals between mouse and primate
datasets in the context of exosome-based interventions. The pipeline integrates:

- mouse bulk RNA-seq (tissue-level effects),
- plasma proteomics,
- and primate methylation data (OMIX007582, Mammal40 array),

to derive exploratory translational insights about rejuvenation-like signatures.

The code is intentionally modular and experiment-focused rather than production-grade.

---

## What’s new in Phase 1

Phase 1 focuses on having a **fully runnable, small-scale end-to-end example** with realistic guardrails, rather than on biological completeness. The main additions are:

- **Transcriptomic clock (primate bulk expression)**
  - Simple, regression-based age predictor trained on bulk RNA-seq.
  - Uses numeric ages from metadata; includes strict checks so the pipeline fails fast if there are too few valid training samples.
  - Produces per-sample predicted age and residuals (e.g. ΔAge-style scores) for exploratory rejuvenation readouts.

- **Improved OMIX example files**
  - New small, consistent, NaN-safe example files for primate bulk expression and plasma proteomics, with matching metadata:
    - `data/processed/OMIX007580_01_example.*` (bulk transcriptomics)
    - `data/processed/OMIX007580_01_metadata_example.*` (matching metadata, including numeric age)
    - `data/processed/OMIX007581-01_example.*` (plasma proteomics for a subset of the same samples)
  - Example files are subsetted from the original OMIX matrices (e.g. ~30 samples) to:
    - avoid memory issues on laptops,
    - demonstrate the expected column naming conventions,
    - prevent all-NaN columns/rows after QC.

- **Safer OMIX I/O layer (`src/omix_io.py`)**
  - Robust metadata loader that handles:
    - real Excel (`.xlsx`, `.xls`),
    - mislabeled “fake xlsx” that are actually CSV/TSV/HTML,
    - plain CSV/TSV/TXT with conservative size checks.
  - Matrix loader with:
    - header-only preflight checks (to avoid accidentally loading massive tables),
    - explicit sample-whitelisting via metadata,
    - hard stops on absurd column counts.
  - Alignment helper that:
    - detects sample ID columns in metadata by name/overlap,
    - aligns matrix columns to metadata rows,
    - enforces reasonable sample counts and overlapping IDs before proceeding.

- **Plasma proteomics cleaning for PCA-based scores**
  - `clean_plasma_matrix` now:
    - drops all-NaN rows/columns,
    - filters features/samples by minimum non-NaN fraction,
    - performs robust median imputation per feature,
    - guarantees a non-empty, numeric matrix after QC when example files are used.

These changes are aimed at making the pipeline easier to run end-to-end on a laptop with **only the small example files**, while surfacing clear, informative errors when real OMIX data are incomplete or inconsistent.

---

## Data

The project uses public datasets from OmicsDI / NGDC, including:

- **OMIX007580 / OMIX007581** (expression / proteomics)
- **OMIX007582** (Mammal40 methylation, IDAT format)

Raw data files (IDAT, full matrices, etc.) are **not** stored in this repository.

Instead, the repo includes **small illustrative examples** that conform to the internal I/O guardrails:

- `data/processed/OMIX007580_01_example.*`  
  Small bulk RNA-seq matrix (genes × samples) with realistic column names (e.g. `01-YM-C_Abdominal_subcutaneous_fat`) used for the transcriptomic clock.
- `data/processed/OMIX007580_01_metadata_example.*`  
  Matching metadata with (at minimum) a sample ID column and a **numeric age** column, plus optional tissue / group annotations.
- `data/processed/OMIX007581-01_example.*`  
  Small plasma proteomics matrix (proteins × samples) with overlapping sample IDs relative to the bulk expression example, pre-filtered to avoid all-NaN columns after QC.
- `data/processed/OMIX007582_beta_matrix_example.csv`  
  A small beta-value matrix (CpG × sample) illustrating expected methylation matrix format.

These files are meant to:

- document expected shapes and column naming,
- allow the pipeline to run on a “toy SRSC dataset” without downloading the full OMIX data,
- and exercise all the main analysis steps (clock, plasma PCA, basic cross-block summaries) in a reproducible, laptop-friendly way.

---

## Methylation processing (OMIX007582, Mammal40)

The `scripts/process_OMIX007582_Mammal40.R` script uses
[SeSAMe](https://bioconductor.org/packages/release/bioc/html/sesame.html) and
`sesameData` to process Mammal40 IDAT files into a CpG × sample beta-value matrix.

**Important limitations:**

- The beta matrix contains **643 sample columns** (technical IDs).
- The accompanying metadata table `OMIX007582-02.csv` has **620 rows** (biological samples).
- No public mapping is provided that unambiguously links every IDAT file to its exact
  biological sample.
- Current implementation wires a placeholder for exosome-attributable fraction; on the public SRSC OMIX datasets there is no shared tissue axis between bulk and plasma.
- Mouse block and cross-species translational summaries are scaffolded but not yet implemented.
- For the plasma block (OMIX007581) the public deposit only provides a single wide CSV file (`PlasmaProtein_exp_mat`, proteins × samples). There is no separate metadata file linking plasma samples to individual animals or to the tissue RNA-seq samples from OMIX007580. 

Consequently:

- `OMIX007582_beta_matrix.csv` is produced with **technical sample IDs** only and used
  for analyses that do not require exact one-to-one sample annotation.
- An **optional** partially named matrix (`OMIX007582_beta_matrix_named_partial.csv`)
  can be generated by assuming column order ≈ metadata row order for the first
  620 samples. This is clearly marked in the code as *exploratory-only* and is
  not used for strong sample-level conclusions.
- The fraction estimate is still marked as non-informative (n_common_tissues = 0, ratio = NaN).
- The pipeline safely skips the mouse block and the cross-species translational summaries and logs a warning.
- There is **no reliable `animal_id` field** that would allow us to
match “this plasma sample” to “this bulk transcriptomic sample” for the
same monkey using only the public data. Because of this, the following components are **optional** and remain disabled or no-op when `animal_id` is missing in the plasma metadata:

  - Bootstrap mediation of a plasma “state score” between treatment and
    `rejuvenation_score`;
  - Simple causal decomposition of the rejuvenation effect into
    cell-intrinsic vs exosome-mediated components.

All other parts of the pipeline (transcriptomic clock, rejuvenation by
group and by tissue, tissue-level expression shifts, and plasma biomarker
ranking) run on the public data without additional metadata.

---

## Pipeline

The main workflow is implemented in `src/run_pipeline.py` and includes:

- loading and harmonizing omics blocks (expression, proteomics, methylation),
- training and applying a **bulk transcriptomic clock** on primate expression data,
- computing plasma proteomics–based state scores (PCA on cleaned proteomics matrix),
- estimating exosome-attributable fractions (where shared tissue axes exist),
- cross-tissue concordance analyses,
- optional translational insight generation when all required blocks are available.

When methylation sample IDs cannot be aligned to full metadata, the pipeline:

- falls back to minimal metadata (`sample_id` only),
- logs clear warnings,
- and skips analyses that would require fully matched annotation, instead of forcing
  an incorrect mapping.

### Clean plasma matrix for PCA

For the plasma proteomics block, `clean_plasma_matrix` performs:

1. Drop rows and columns that are entirely NaN.
2. Filter features that have too many missing values (configurable fraction).
3. Filter samples that have too many missing values (configurable fraction).
4. Median imputation per feature.
5. Final safety checks to ensure no remaining NaNs.

The resulting matrix is suitable for PCA-based state scoring, even on small example datasets.

---

## Reproducibility

A minimal `environment.yml` is provided with Python dependencies
(e.g. `pandas`, `numpy`, `scanpy`, etc.). Methylation processing depends on:

- R (`>=4.3`)
- Bioconductor packages `sesame`, `sesameData`, `BiocParallel`

See `scripts/process_OMIX007582_Mammal40.R` for details.

---

## Status and future work

This is a **v0.1 exploratory / Phase 1** implementation. The current focus is on:

- robust, explicit handling of real-world data issues (NaNs, mismatched IDs, huge matrices),
- a minimal yet reproducible **transcriptomic clock** for primate bulk expression,
- NaN-safe, laptop-scale example datasets for all major blocks (bulk, plasma, methylation).

Planned directions include:

- obtaining or reconstructing a more reliable sample matching for OMIX007582,
- extending the translational module with additional mouse interventions,
- richer exosome-attributable fraction modeling once shared tissue axes are available on public data,
- refactoring the pipeline into a more generalizable package.

Despite the limitations, this repository documents the end-to-end process of:

- integrating multi-omics public datasets,
- handling real-world metadata inconsistencies,
- building and debugging a transcriptomic clock and plasma PCA scores under strict guardrails,
- and extracting exploratory translational hypotheses in the aging / longevity domain.

## Version history

- v0.1 Phase 1
  - Added primate bulk transcriptomic clock.
  - Added NaN-safe plasma proteomics cleaning and PCA scores.
  - Introduced small, consistent OMIX example files.
  - Hardened OMIX I/O (metadata + matrix guardrails).
  - Documented Mammal40 sample-matching limitations.

- v0.1 Phase 2 - Robust rejuvenation score & tissue signals

Phase 2 extends the example pipeline with:

- A robust definition of `delta_age = predicted_age – chronological_age`.
- Global and tissue-level rejuvenation summaries with bootstrap confidence intervals.
- Simple tissue-level expression effects (treated vs control) based on bulk RNA-seq.
- Robust plotting utilities for rejuvenation by group (with sensible fallbacks when sample sizes are small).

New outputs are written under `results/`:

- `rejuvenation_by_tissue.csv`
- `tissue_expression_effects.csv`
- `figures/rejuvenation_by_group_boxplot.png` (example run)

## What’s new (v0.2 – transcriptomic rejuvenation & plasma)

This update focuses on making the rejuvenation signal more robust and easier to inspect:

- Added a cross-validated transcriptomic clock for primate bulk tissues.
- Added a proxy rejuvenation score (`delta_age` / `rejuvenation_score`) with:
  - Global summary of treated vs. control animals.
  - Tissue-level summaries (`rejuvenation_by_tissue.csv`).
- Exported tissue expression effects (`tissue_expression_effects.csv`) to inspect which genes and tissues move the most.
- Added a minimal plasma proteomics module:
  - Cleaning and filtering of the OMIX007581 plasma matrix.
  - Simple “state” outcome based on Y / WT / V / GES group labels.
  - Ranking of top plasma biomarker candidates (`plasma_biomarkers.csv`) and a summary plot.

The exosome-attributable fraction and translational insights modules are still experimental:
when the necessary inputs (mouse tissue effects, mouse expression log matrix) are not available,
the pipeline now fails gracefully and emits explicit warnings instead of crashing.