# Citation and release provenance

For software use, cite the author and repository title in
[CITATION.cff](../CITATION.cff) and give the exact Git commit or released tag
you used. For the reported September results, also cite the
[26 September 2026 evidence snapshot](evidence/2026-09-26/README.md).

## Analysis and software history

| Record | Identity |
|---|---|
| Corrected full run, 26 September 2026 | Started at `7b49ef43d1e01ea34ea16da930d7107339dc0874` with local changes. The [run manifest](evidence/2026-09-26/manifest.json) records the source fingerprints, inputs, settings, and core software versions. |
| Corrected workflow merged | [PR #15](https://github.com/richard2049/Omix-exosome-rejuvenation/pull/15) integrated the workflow at `8abd9d0f968236093ae857ae076085ad59de361a` after the September run. |
| Code baseline for release preparation | [PR #16](https://github.com/richard2049/Omix-exosome-rejuvenation/pull/16) added numerical and scientific checks at `8425dc7bb42aae79d5b10aebdb1ad76affb5050a`. |
| Full-profile comparison, 20 November 2026 | A private run at `da62d6b` used the same 11 input hashes and effective settings. All shared values and statuses agreed across 34 tables. The mouse effect table gained three validation columns. |

Exact replay of September requires the recorded source state and environment.
The full reconstruction of those files and package builds remains unverified.
The November run documents numerical agreement on the same inputs.

## Proposed `v0.3.0` release

The proposed minor version follows the exploratory `v0.2` series and marks
the revised analysis workflow and its documented evidence.

### Draft release description

> OMIX Exosome Rejuvenation v0.3.0 documents the corrected full-profile
> workflow, links to aggregate evidence from September 2026, and includes
> a synthetic demo. In macaques, an animal-grouped, controls-only
> transcriptomic clock and animal-level treatment contrasts gave no robust
> evidence of transcriptomic rejuvenation after SRC treatment. Plasma and
> three-tissue mouse comparisons remain exploratory. Confirmed plasma–tissue
> links and a biological methylation sample map remain unavailable.

### Release checklist

1. Check the evidence hashes and [figure sources](assets/README.md). Verify
   that the source archive bundles the [synthetic demo inputs](datasets.md#demo-inputs-and-redistribution).
2. Once the release version and publication date are approved, record both in
   `CITATION.cff` and validate it against [CFF 1.2.0](https://github.com/citation-file-format/citation-file-format/blob/1.2.0/schema-guide.md).
3. Verify CI for the exact reviewed commit. With publication approval, tag
   that commit and include links to the evidence and reproducibility guide in
   the release notes.
