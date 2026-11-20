# Reproducibility

## Environment

Create the development and CI environment:

```bash
conda env create -f environment.yml
conda activate srsc
```

If an existing environment must be updated after the specification changes:

```bash
conda env update -n srsc -f environment.yml --prune
```

The [environment specification](../environment.yml) uses Python 3.11 and
OpenBLAS. GitHub Actions tests the public path on Linux. The September full
run used Windows and recorded the following numerical stack in its manifest.

| Python | NumPy | SciPy | pandas | scikit-learn | statsmodels |
|---|---|---|---|---|---|
| 3.11.15 | 2.4.3 | 1.17.1 | 2.3.3 | 1.8.0 | 0.14.6 |

Conda resolves the development recipe at installation. The September manifest
captures core versions. The complete Conda build list and pip artifacts for
that run are unavailable. For future full runs, export the analysis environment:

```bash
conda list --name srsc --explicit --md5 > conda-explicit.txt
conda env export --name srsc > environment-resolved.yml
```

Record the platform and review package URLs and local paths before sharing.
Restore the same Conda builds and checksums on that platform with
`conda create --name srsc-replay --file conda-explicit.txt`. Install any pinned
packages listed under `pip:` separately, then validate the restored environment.
[Conda environment documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

## Data Profiles

| Profile | Input | Purpose |
|---|---|---|
| `demo --safe` | Bundled synthetic examples | Smoke test with reduced feature and resampling settings. Exercises the output tables and records unavailable analyses. |
| `full` | Downloaded and prepared files from the [dataset guide](datasets.md) | Standard fits, resampling, and diagnostics. Recompute the reported analyses with matching inputs and settings. |
| `auto` (default) | Looks for the full profile's bulk input first, then the demo input | Runs the selected profile. Check `data_profile` in the manifest to verify the selection. |

The main commands are:

```bash
python -m src.run_pipeline --profile demo --safe
python -m src.run_pipeline --profile full --data-root /path/to/omix-data
python -m src.run_pipeline --profile auto
```

Without `--data-root`, `full` looks under `data/RAW/data`. Keep raw inputs
read-only and use separate clean checkouts to keep each run's generated
`results/` and `figures/` distinct. The archived September full invocation
took about 7.5 minutes on its Windows host, excluding data preparation and
report generation. Runtime depends on hardware and the environment.

Every CLI execution starts `results/run_manifest.json` before model fitting.
It records the command, environment, complete configuration, SHA-256 values of
configured inputs, Git HEAD, dirty-worktree status, and a content fingerprint
of the local source tree. On exit it is finalized as `completed` or `failed`
and inventories the outputs present at finalization. Start with empty output
directories so earlier artifacts cannot enter the inventory.

The [OMIX007580 input checks](datasets.md#omix007580-release-specific-ingestion)
verify the matrix and metadata hashes before loading. The audit is written to
`results/omix007580_input_audit.csv`. The [September evidence snapshot](evidence/2026-09-26/README.md)
contains the recorded configuration, source fingerprints, and selected outputs.

The corrected full-profile clock protocol uses `log2(1 + CPM)` and fits Ridge
only on the untreated reference groups `Y_C`, `M_C`, `O_C`, and `O_V`.
`O_WT` and `O_GES` receive cross-fitted predictions but never enter training.
Animals, rather than tissue rows, define the outer and inner folds, and the
primary model-selection metric gives each held-out control animal equal
weight. The 11-value Ridge grid, three-fold inner selection, historical outer
partition, and three diagnostic partitions are frozen. Their animal-level
assignments are stored in `config/primate_clock_folds.csv` and checksum-checked
at runtime so restoration or removal of a tissue cannot silently redistribute
an animal.

Treatment inference is kept separate from clock training. The prespecified
primary macaque contrast is `O_GES - O_V` (`SRC - vehicle`); `O_WT - O_V` and
`O_GES - O_WT` are secondary. Tissue-level bootstrap and permutation resample
or relabel animals, with sex used as the permutation stratum when available.
The primary permutation p-values are Bonferroni-adjusted over the planned
tissue family; a Holm correction over every tissue and all three contrasts and
a per-contrast BH sensitivity are retained alongside the unadjusted values.
The reported 95% percentile-bootstrap intervals are pointwise rather than
family-simultaneous and are labelled accordingly in the result table.
The secondary global summary first averages tissues within animal, so an animal
with more observed tissues does not become additional biological replication.
A separate tissue/age/sex-adjusted OLS sensitivity uses animal-clustered
covariance. These procedures quantify uncertainty in a clock-derived endpoint;
they do not establish functional rejuvenation. Treatment-stage resampling is
conditional on the frozen cross-fitted predictions and does not refit the clock
inside every bootstrap or permutation, so model-selection and clock-training
uncertainty are not fully propagated into these intervals.

The cross-species response profile standardizes each primary-compatible tissue
contrast with Hedges' g and gives each tissue equal weight. It requires at
least three valid pairs and excludes whole mouse brain versus macaque
hippocampus from the primary summary. With random seed `42`, the default run
uses 1,000 bootstrap replicates and 1,000 treatment-label permutations; safe
mode caps each at 250. Macaque animals are resampled jointly across their
tissues. Because public cross-tissue mouse animal linkage is not confirmed,
mouse observations are resampled within tissue, treatment arm, and sex rather
than being joined across tissues. Leave-one-tissue-out ranges expose dependence
on individual tissue pairs. These procedures quantify uncertainty in
cross-species resemblance, not a causal exosome-attributable fraction.

`58-MF-C-Trachea_4` is retained in the immutable source matrix but excluded
from clock fitting and prediction because its declared animal alias remains
unresolved. A full run records that decision in
`results/clock_exclusions_primates.csv`. Selecting the weakest tested Ridge
penalty is reported as a boundary condition; the grid is not expanded after
observing that result.

Mediation is now disabled by default in every profile. Name-derived plasma
aliases are `inferred` and fail the existing high-confidence gate even if a
caller enables mediation. Enabling that flag alone does not validate identity.

Optional `sample_id`/`animal_id` metadata are likewise treated as
`unverified_metadata`. Only a dedicated plasma linkage manifest whose SHA-256
is fixed in configuration can assign `metadata_exact`. The validator requires
explicit per-row confirmation and provenance, one-to-one sample/animal keys,
and concordant group and sex. It rejects the complete manifest on checksum,
schema, duplication, collision, or concordance failure and records the outcome
in `results/plasma_linkage_manifest_audit.csv`.

The plasma protein analysis is independent of that cross-modal gate. It uses
the released plasma sample aliases as the within-assay unit, applies `log2` to
strictly positive abundances, and fits the prespecified `GES - V`, `WT - V`,
and `GES - WT` contrasts with sex adjustment and HC3 covariance. FDR is
controlled separately per contrast. The plasma age-state PCA is fitted only on
young and vehicle-reference samples; WTC and SRC samples are projected without
refitting. Exact upstream normalization provenance for `OMIX007581` remains
`SOURCE_VERIFICATION_PENDING`.

## Repository Workflow and Local Data

The Git repository is the single source of truth for code, configuration,
tests, demo inputs, and public documentation. Develop changes on focused
branches and integrate them through reviewed pull requests; do not maintain a
second code copy through manual file synchronization.

If separate clean and experimental directories are useful, create a Git
worktree so both directories share the same history:

```bash
git switch main
git pull --ff-only
git worktree add ../Omix-exosome-rejuvenation-dev -b analysis/my-change main
```

Keep full OMIX inputs in a read-only directory outside version control.
Maintain a separate checksum-verified backup of irreplaceable source data.

## Validation

Run the focused scientific guardrail suite:

```bash
python -m pytest -q tests/test_scientific_guardrails.py
```

Run the demo and generate the report figures:

```bash
python -m src.run_pipeline --profile demo --safe
python -m src.demo_validation
python -m src.report_figures
```

`src.demo_validation` checks the result tables and their evidence fields.
`src.report_figures` renders figures from the existing CSVs.

CI compiles the source, runs all tests, then runs and checks the demo in Conda.
It runs for pull requests to `main` and pushes to `main` or `release/**`.
A branch push checks out the exact pushed SHA. These checks cover software
behavior. [Technical integrity workflow](https://github.com/richard2049/Omix-exosome-rejuvenation/actions/workflows/ci.yml)

For a full-profile comparison, verify the nine data-root inputs against the
[recorded checksums](evidence/2026-09-26/manifest.json) and verify the two config
file hashes from the LF Git blobs. After running `full`, inspect the completed
manifest for the resolved profile, effective configuration, software versions,
and source state. Compare the outputs to the snapshot by contrast and tissue
labels, checking estimates, intervals, counts, and reason codes. The snapshot's
hashes identify the archived files. Any comparison using a different source
state or environment should state the differences and report numerical agreement
explicitly. See [release provenance](release.md) for the historical source limits.

See [figure provenance](assets/README.md) for regenerating the three README
figures from reviewed full-profile results.

## Optional Audit Workflows

The focused [public-data refresh](data_refresh.md) is independent of the main
pipeline and requires no new dependencies or raw sequencing downloads:

```bash
python -m src.data_refresh --output-dir results/data_refresh --data-root /path/to/omix-data
python -m src.data_refresh --output-dir results/data_refresh --data-root /path/to/omix-data --offline
```

It caches immutable public responses and verifies their checksums on replay.
Choose a new output directory when intentionally inspecting a new source
release. The optional `--data-root` is read-only and permits comparison with
the existing local plasma source and bulk animal aliases.

```bash
python -m src.omix007582_audit
python -m src.omix009283_metadata
python -m src.omix009284_audit
```

The optional Mammal40 IDAT rebuild uses R and Bioconductor. The main `srsc`
environment includes R and `BiocManager` as a bootstrap:

```bash
conda run -n srsc Rscript -e "BiocManager::install(c('sesame', 'sesameData', 'BiocParallel'), ask = FALSE, update = FALSE)"
conda run -n srsc Rscript src/scripts/process_OMIX007582_Mammal40.R --max-prefixes 1 --prep-candidates default --output-dir results/omix007582_rebuild_smoke
```

For isolation, create the optional R-only environment instead:

```bash
conda env create -f environment-omix007582-r.yml
conda run -n srsc-omix007582-r Rscript src/scripts/process_OMIX007582_Mammal40.R --help
```

The first rebuild may cache sesame reference resources through ExperimentHub.
Use `--max-prefixes` or `--prefixes` before attempting the complete archive.

## Troubleshooting

### The full profile cannot find source files

The public repository contains synthetic examples. Obtain the full OMIX
downloads separately.
Use `--profile demo` to verify the installation, or place independently
downloaded source files under the configured full-data root and use
`--profile full`. Do not rename samples or edit raw files to satisfy a loader.

### A linkage-dependent result is not estimable

Inspect `results/plasma_to_animal_map.csv`, `results/linkage_qc_report.csv`,
`results/linkage_audit.csv`, and `results/estimability_report.csv`. A structured
stub is expected when identity, confidence, collision, or sample-support rules
fail; it is not a pipeline crash.

### Methylation validation remains blocked

`OMIX007582` requires a defensible Sentrix technical-to-biological sample map.
Rebuilding beta values from IDAT files does not recover that missing identity.
See the [sample-map audit](OMIX007582_sample_map_audit.md).

### R packages are unavailable

The optional Mammal40 workflow is not required for the Python demo. Install
the Bioconductor packages shown above or use the isolated R environment. Test
one prefix before processing the complete archive.

### Report figures show status panels

The report layer visualizes structured non-estimable outputs rather than
inventing estimates. Read `figures/report/report_figure_manifest.csv` for each
figure's source tables and status message.
