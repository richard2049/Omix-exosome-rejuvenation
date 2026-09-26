# OMIX009284 Read-Only Audit

## Scope

This audit records what can be established from the released `OMIX009284`
files without modifying raw data or integrating the dataset into the canonical
pipeline.

## Main findings

- `OMIX009284` is consistent with a **PBMC single-cell RNA export**, not a
  bulk tissue matrix aligned to the current cross-tissue attribution workflow.
- The raw release contains **32 per-sample gene-by-cell matrices** and **one
  Seurat-style cell metadata table**.
- The metadata table records `group`, `sample`, `tissue`, RNA QC columns, and
  cell barcodes.
- The metadata indicate **PBMC only**, with groups `GES`, `WT`, `Vehical`, and
  `Young`.
- The cell-barcode suffixes used in the expression matrices can be linked back
  to sample labels through the metadata table in a read-only manner.

## Interpretation for the current repository

This dataset is useful for future **PBMC-focused cell-state or mechanistic
follow-up**, but it does not materially strengthen the current
**cross-species macaque whole-cell versus mouse exosome response alignment**
question:

- tissue scope is limited to `PBMC`
- assay granularity is single-cell rather than the repo's current bulk/tissue
  endpoint layer
- no implementation decision in the main pipeline should depend on this audit
  alone

## Machine-readable outputs

Run:

```bash
python -m src.omix009284_audit
```

This writes:

- `results/omix009284_audit_summary.csv`
- `results/omix009284_file_inventory.csv`
- `results/omix009284_suffix_sample_map.csv`

The suffix-to-sample map is informative for future work, but it should remain
an audited support artifact until a concrete PBMC-specific analysis question is
added to the repo.
