# Data Linkage

## Why Linkage Matters

The repository combines tissue transcriptomics, plasma proteomics, methylation,
and supporting datasets. Samples from the same study group are not necessarily
measurements from the same animal. Group membership, file order, and similar
sample names therefore cannot substitute for demonstrated biological identity.

## Biological Units

| Analysis | Required unit |
|---|---|
| Transcriptomic age model | Animal, with repeated tissues kept within the same validation fold |
| Tissue treatment effect | Animal within tissue |
| Plasma protein treatment contrasts | Plasma sample; group and sex come from the released sample alias, while cross-modal animal identity is not required or implied |
| Plasma-to-tissue association | Linked animal |
| Mediation | One valid analytical row per linked animal unless repeated structure is modeled explicitly |
| Cross-species alignment | Tissue and contrast after explicit species-aware matching; not individual linkage |

## Required Evidence Artifacts

The pipeline writes five complementary linkage outputs:

| Output | Question answered |
|---|---|
| `plasma_to_animal_map.csv` | What mapping was proposed, by which rule, and with what confidence? |
| `plasma_linkage_manifest_audit.csv` | Was a checksum-pinned, explicitly confirmed linkage manifest supplied and accepted? |
| `linkage_qc_report.csv` | Are coverage, uniqueness, and collisions acceptable? |
| `linkage_audit.csv` | Which animal identifiers overlap across the relevant modalities? |
| `estimability_report.csv` | May a linkage-dependent analysis proceed under configured requirements? |

Actual coverage and gate status are profile-specific and should be read from
these outputs. A failed gate must produce a structured non-estimable result,
not a group-level surrogate.

## Permitted and Blocked Uses

The September 2026 refresh classifies name-derived plasma links as `inferred`.
The 24 full-data naming matches remain useful candidates, but do not meet the
high-confidence gate. BioSample metadata corroborate bulk specimen attributes;
they do not prove that the individual numbering is shared with plasma.
Mediation is disabled by default. The optional refresh emits a separate
`plasma_linkage_candidates.csv` for the author query, without promoting it to
an accepted identity map. See [data_refresh.md](data_refresh.md).

In `linkage_qc_report.csv`, `n_mapped_valid_in_bulk` and `mapping_coverage`
describe candidate alias existence; `n_mapped_high_conf`, `estimable`, and
`evidence_level` distinguish admissible individual-level links. A positive
candidate coverage is not confirmed biological identity.

Generic optional metadata containing `sample_id` and `animal_id` is also only
candidate evidence. It is labelled `unverified_metadata` and cannot enter a
linked analysis. This prevents a syntactically valid table from silently
turning naming compatibility into confirmed biological identity.

The gate can be opened only through the dedicated
`plasma_linkage_manifest` configuration together with its exact
`plasma_linkage_manifest_sha256`. The CSV must contain `sample_id`, `animal_id`,
`identity_confirmed`, and `evidence_source`. Every row must be explicitly
confirmed, sample and animal identifiers must form a one-to-one mapping, and
group and sex must agree between plasma and bulk metadata. The whole manifest
is rejected if any row or checksum fails; valid-looking rows are not accepted
piecemeal. Configuring such a manifest records acceptance of its stated
provenance—it does not make an unverified author key valid by itself.

- Unlinked plasma data may support descriptive or group-level exploratory
  summaries when labeled accordingly.
- Linked analyses must exclude unresolved, low-confidence, or colliding maps
  according to the configured rules.
- Tissue rows must not be replicated to manufacture multiple observations for
  one animal-level plasma measurement.
- Missing identifiers must not be filled from row order, treatment balance, or
  the nearest-looking sample name.

The [inference framework](inference_framework.md) is authoritative if this
reader-oriented guide and an implementation detail appear to conflict.
