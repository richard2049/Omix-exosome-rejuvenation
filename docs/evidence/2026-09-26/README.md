# September 2026 full-profile evidence

These aggregate tables support the [scientific results](../../../RESULTS.md)
from the completed standard full-profile run of 26 September 2026. The
[manifest](manifest.json) records the run settings, input hashes, software
versions, source state, and SHA-256 and size of each included file. The
[output reference](../../outputs.md) describes the complete generated outputs.

| Evidence | Tables |
|---|---|
| Chronological-age prediction and validation | [Clock metrics](clock_metrics_primates.csv), [fold audit](clock_fit_audit_primates.csv) |
| Animal-level treatment contrasts | [Global contrasts](global_rejuvenation_summary.csv), [tissue contrasts](rejuvenation_by_tissue.csv) |
| Exploratory plasma response | [Plasma age-state axis](plasma_age_axis_summary.csv), [primary protein contrast summary](plasma_primary_contrast_summary.json) |
| Plasma identity and mediation decisions | [Linkage audit](linkage_qc_report.csv), [correlation status](plasma_age_axis_delta_age_correlation.csv), [estimability gate](estimability_report.csv), [mediation status](mediation_summary.csv) |
| Exploratory macaque–mouse comparison | [Summary](cross_species_response_alignment_summary.csv), [three tissue pairs](cross_species_response_alignment_by_tissue.csv), [causal attribution status](exosome_fraction_summary.csv) |
| Methylation validation status | [Multimodal concordance status](multimodal_concordance_summary.csv) |

The CSVs match the archived pipeline outputs byte for byte. The protein
contrast summary uses estimable `GES_vs_V` rows from the checksum-verified
`plasma_biomarkers.csv`. It counts `fdr_significant` and `stability_tested`
flags, including the number of stability-tested rows with `sign_agreement == 1`,
and lists the accessions and directions reported in RESULTS. The full protein
table, individual predictions, and per-sample linkage records stay with the run.

Input accessions and acquisition notes are in the [dataset guide](../../datasets.md).
The [OMIX data-use policy](https://ngdc.cncb.ac.cn/omix/term_and_policy?lang=en_GB)
describes conditions for access and use of source data.

## Verify the files

Run from the repository root in a Bash-compatible shell:

```bash
python - <<'PY'
from hashlib import sha256
import json
from pathlib import Path

root = Path("docs/evidence/2026-09-26")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
entries = manifest["outputs"] + manifest["derived_outputs"]
for entry in entries:
    data = (root / entry["file"]).read_bytes()
    if len(data) != entry["size_bytes"] or sha256(data).hexdigest() != entry["sha256"]:
        raise SystemExit(f"Checksum or size mismatch: {entry['file']}")
print(f"Verified {len(entries)} files")
PY
```

On PowerShell, `Get-FileHash -Algorithm SHA256` can also check individual files.
The public manifest uses relative paths and retains the archived manifest hash
and source fingerprints. See the [reproducibility guide](../../reproducibility.md#data-profiles)
to verify a new run and the [release notes](../../release.md) for the limits of
recreating the September run.
