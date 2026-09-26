# Public-data refresh: biological scope and provenance

Verified on 14 September 2026 against NGDC and the locally held Cell article.
This is a focused source and identity audit, not a new cohort analysis.

## What changes biologically

The project can now distinguish recipient plasma, exosome cargo, and tissue
responses to exosome perturbation. These layers support a concrete question:
which SRC-associated responses are shared with exosome treatment, and which
distinguish SRCs from wild-type cells?

Existing clock-effect comparisons show directional overlap for both treatment
types. That observation motivates direct SRC-versus-WTC comparisons rather
than attributing all concordance to engineered SRCs. It does not establish
equivalence of SRC and WTC effects. Molecular programs and cargo-to-response
relationships remain the next analysis, not an outcome of this refresh.

## Deposits inspected

| Deposit | Inspected content | Interpretation |
|---|---|---|
| [OMIX007581](https://ngdc.cncb.ac.cn/omix/release/OMIX007581) | 3,608 protein rows; 32 plasma identifiers | Retain as macaque plasma; downloaded file matches the existing local source byte-for-byte |
| [OMIX009654](https://ngdc.cncb.ac.cn/omix/release/OMIX009654) | 4,506 protein rows; six F3/WT identifiers, each with intensity and normalized columns | Additional candidate exosome cargo resource; not a replacement plasma cohort |
| [OMIX009655](https://ngdc.cncb.ac.cn/omix/release/OMIX009655) | 64 metabolite rows; sample-1 through sample-12, plus chemical annotations | Exploratory resource awaiting sample-group and preparation labels |

The proteomic files share 2,239 exact non-missing gene annotations, but zero
exact protein accessions and zero sample identifiers. Gene-annotation overlap
is descriptive: it is neither a validated orthology map nor evidence of
sample identity. The two sets of six numerical columns in OMIX009654 must not
be counted as twelve independent samples.

Both new deposit titles describe macaque plasma, whereas the article's data
availability section identifies exosome proteomics and metabolomics. Its DIA
methods use a human protein reference for SRC-Exo versus WTC-Exo; the protein
file also contains human descriptions. Preserve that discrepancy explicitly.
The biological material, F3 label, and preparation/replicate identities should
be confirmed before differential cargo analysis. The local PDF bears an
in-press banner; this audit does not claim to compare all editorial versions.
Primary article: [Lei et al.](https://doi.org/10.1016/j.cell.2025.05.021).

Source SHA-256 values:

| File | SHA-256 |
|---|---|
| OMIX007581-01.csv | `5e9ee404aed57930910ed3502d9fa22fcf735860aa9defaa19d00495aff4da77` |
| OMIX009654-01.txt | `d0ddc9675bd6a31087a43be8fc26f171632b510489b3329891a8d0811bb2987e` |
| OMIX009655-01.txt | `f3e5c753731ff815b7ed3db337919ec85a1f0a6a3300936c35b4ed292ebc3ac7` |

## A small, reproducible sample manifest

```bash
python -m src.data_refresh --output-dir results/data_refresh --data-root /path/to/omix-data
python -m src.data_refresh --output-dir results/data_refresh --data-root /path/to/omix-data --offline
```

The command reads the full PRJCA035748 index (1,694 records at verification)
and retrieves a deliberately focused panel of 58 detailed BioSamples: 24
intervention hippocampus specimens, all 32 PBMC samples, and two mouse
examples. This panel supports the author query; it is not an analytical
selection or a complete metadata export. Repeated tissues and sequencing runs
are not additional animals.

Outputs under the chosen directory:

- `project_index.csv`: full index and explicit detailed-retrieval coverage.
- `sample_manifest.csv`: reported specimen metadata, candidate animal alias,
  source URL, retrieval time, and checksum. Validated animal identity remains
  blank; treatment descriptions retain the source wording.
- `plasma_linkage_candidates.csv`: one row per plasma sample, with candidate
  bulk-alias existence and BioSample references; 24 candidates, zero identity
  confirmations. Eight young samples remain without a candidate.
- `dataset_inventory.csv`, `shared_gene_annotations.csv`, `audit_summary.json`:
  matrix dimensions, missingness, duplicate identifiers, and exact overlap.
- `source_inventory.csv` and `sources/`: downloaded responses and provenance.

Snapshots are reused only when their URL and hash agree. Use a new output
directory for a new upstream release. `--offline` rebuilds the tables without
network access. Source data are never overwritten, and none of these manifests
automatically changes the primary pipeline's animal or group assignments.

## Linkage correction

Historically the naming rule FGES_1 -> F-GES-1 returned `high` confidence.
It now returns `inferred`: existence of the bulk alias is a candidate check,
not author confirmation that numbering is shared across modalities. Inferred
matches fail the existing high-confidence gates. Mediation is disabled by
default, while plasma group summaries and tissue analyses remain available.

The canonical test expectation for this naming rule was corrected to match
the documented identity requirement, not weakened to preserve old results.
Tests with explicitly supported links continue to exercise positive gates.
Unchanged numerical thresholds and existing result-table contracts remain in
force. Historical linked correlations and mediation in RESULTS are conditional
records, not validated individual-level evidence after this correction.

## Raw sequencing and follow-up studies

[CRA022788](https://ngdc.cncb.ac.cn/gsa/browse/CRA022788),
[CRA023573](https://ngdc.cncb.ac.cn/gsa/browse/CRA023573), and
[CRA023595](https://ngdc.cncb.ac.cn/gsa/browse/CRA023595) provide raw sequencing
provenance for macaque tissues, mouse exosome response, and PBMC, respectively.
CRA023573 lists 295 experiments/590 files; the local mouse matrix has 295
columns. This is a candidate raw-to-processed relationship, not an independent
replication or a completed sample crosswalk. No FASTQs are required for this
refresh. Mouse brain versus macaque hippocampus remains an anatomical
approximation; examine its exclusion before mechanistic interpretation.

The [FOXO1-LHX1 epididymal study](https://doi.org/10.1093/procel/pwag020)
is a useful mechanistic reference, with reported deposits CRA030395 and
HRA014320. Cohort/sample independence must be established before calling it
external validation. Integration is deferred so it does not delay the query.

## Questions ready for the authors

1. Do plasma identifiers such as FGES_1/MGES_1 identify the same individual
   macaques as F-GES-1/M-GES-1 in CRA022788/PRJCA035748? Can you provide a
   cross-modal key including sampling timepoints and assay-specific exclusions?
2. Do OMIX009654 and OMIX009655 measure isolated exosome preparations? What are
   the F3/WT and sample-1...12 assignments, biological/technical replicates,
   preparation batches, and relationships between the assays?
3. Is a Sentrix barcode/position-to-biological sample sheet available for
   OMIX007582, including the explanation for 643 technical IDs versus 620
   metadata rows?

The next analysis can compare compatible tissue responses and candidate cargo
without constructing a new composite mechanism score. Changing the name of
the legacy fraction would not validate its mathematics. The primary interface
remains signed effects, uncertainty, and contrast-specific concordance.

## Verification and review boundary

This is an L3 (claim-critical) change because identity confidence determines
which individual-level results may be interpreted. The acceptance contract is
the existing inference framework: naming is not biological identity, tissues
must not multiply plasma observations, and unsupported links must fail the
existing high-confidence gate. No numerical threshold, CI rule, or output
schema was relaxed.

Verification in the `srsc` environment on 14 September 2026:

- `python -m pytest -q`: 47 passed, including six new refresh tests and the
  explicitly justified correction to one canonical confidence assertion.
- `python -m src.run_pipeline --profile demo --safe`, followed by
  `python -m src.demo_validation`: successful exit and 23 valid result tables.
- Syntax compilation of the three affected Python modules passed.
- Online source retrieval followed by `src.data_refresh --offline`: all six
  CSV tables and the JSON summary reproduced byte-for-byte from 62 snapshots.
- Direct checks against the full source metadata: 2,059 metadata rows and 32
  plasma columns; 24 candidate aliases, zero high-confidence overlaps, and
  mediation not estimable. Candidate group and sex concordance were both 1.0;
  this corroborates naming consistency, not cross-modal identity. The 2,059
  metadata rows are not the 2,058 analyzed clock samples in the historical run.
- Regenerated 15 demo report figures; inspected the linkage gate figure and
  linkage, estimability, mediation, and plasma-axis result tables. Demo figures
  were not copied into public portfolio assets or used as biological findings.

The statements above record the bounded refresh as verified on 14 September
2026. At that stage the full clock, treatment-effect, and mouse workflows had
not been rerun; the refresh tested the source and identity boundary directly.
A later user-authorized standard full-profile run completed on 26 September
2026 using the corrected ingestion and current scientific contracts. Its
provenance is recorded in `results/run_manifest.json`, and the current README,
RESULTS narrative, and three established public figure slots were subsequently
reconciled with those outputs.

Neither the original refresh nor the later full run supplies the missing gene
identities, exosome-cargo group labels, or cross-modal animal key. No pathway
enrichment or differential cargo analysis has therefore been promoted from
these resources. The current cross-species output is the explicitly non-causal
C/R/A response-alignment profile, not a composite mechanism score. Rendering
and reproduction details belong in the [figure documentation](assets/README.md).
No branch has been published and no author has been contacted.
