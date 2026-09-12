# Reproducibility

## Environment

Create the main environment once:

```bash
conda env create -f environment.yml
conda activate srsc
```

If an existing environment must be updated after the specification changes:

```bash
conda env update -n srsc -f environment.yml --prune
```

## Data Profiles

The pipeline has one entry point and three data-selection profiles:

```bash
python -m src.run_pipeline --profile demo --safe
python -m src.run_pipeline --profile full
python -m src.run_pipeline --profile auto
```

- `demo` reads reduced public examples from `data/PROCESSED`.
- `full` reads locally downloaded inputs from `data/RAW/data`.
- `auto` prefers the full layout when present and otherwise uses the demo
  layout.
- `--safe` applies conservative laptop settings.
- `--data-root <PATH>` overrides the selected profile's data directory without
  changing source code.

Raw inputs are immutable. Generated tables and figures are written to
`results/` and `figures/`.

## Validation

Run the focused scientific guardrail suite:

```bash
pytest -q tests/test_scientific_guardrails.py
```

Run the pipeline and then regenerate the interpretation-facing report layer:

```bash
python -m src.run_pipeline --profile demo --safe
python -m src.demo_validation
python -m src.report_figures
```

`src.demo_validation` checks that the demo generated the expected result tables
and preserved their machine-readable evidence fields. It does not assess
biological validity or upgrade any evidence claim.

The report command reads existing `results/*.csv`; it does not recompute the
analysis or replace diagnostic plots.

Pull requests and updates to `main` run the same compilation, test, demo, and
output-contract checks in GitHub Actions. Passing CI demonstrates technical
integrity of the public workflow, not biological validity of the underlying
study design, metadata mappings, or causal interpretation.

When preparing a public release from reviewed full-profile results, refresh the
three tracked README figures explicitly:

```bash
python -m src.report_figures --portfolio-assets-dir docs/assets
```

Do not publish these snapshots from a reduced demo run unless they are clearly
labelled as demo outputs.

## Optional Audit Workflows

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

The public repository contains reduced examples, not the full OMIX downloads.
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
