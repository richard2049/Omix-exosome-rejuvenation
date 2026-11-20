# Citation and release provenance

For software use, cite the author and repository title in
[CITATION.cff](../CITATION.cff) and give the exact Git commit or released tag
you used. For the reported September results, also cite the
[26 September 2026 evidence snapshot](evidence/2026-09-26/README.md).

## The analysis and the software history

| Record | Identity |
|---|---|
| Corrected standard full run, 26 September 2026 | Recorded HEAD `7b49ef43d1e01ea34ea16da930d7107339dc0874`, with working-tree changes at runtime. The [run manifest](evidence/2026-09-26/manifest.json) records the source fingerprint, runtime source file hashes, inputs, settings, and core package versions. |
| Subsequent public integration | [PR #15](https://github.com/richard2049/Omix-exosome-rejuvenation/pull/15) integrated the corrected workflow at `8abd9d0f968236093ae857ae076085ad59de361a`. The merge is a review milestone. The recorded run fingerprint describes the earlier working tree. |
| Code baseline for release preparation | `8425dc7bb42aae79d5b10aebdb1ad76affb5050a`, the [PR #16 merge](https://github.com/richard2049/Omix-exosome-rejuvenation/pull/16) that added numerical and scientific guardrail checks. Verified against GitHub `main` on 20 November 2026. |
| Full-profile comparison, 20 November 2026 | A private run of `da62d6b` used the same 11 input hashes and effective analysis settings. All shared values and statuses agreed across 34 tables. The mouse effect table added three validation diagnostics. |

The source hashes identify the files used at runtime. Exact reconstruction of the
dirty working tree and full package builds remains unverified. A checkout of a
later merge can be used to test numerical agreement under its own recorded
source and environment. The archived analysis date and any future software
release date are recorded separately.

## Proposed `v0.3.0` release

The next pre-1.0 tag would follow the exploratory `v0.2` series. Here the minor
version marks a substantive change to the documented analytical workflow. Each
release identifies the reviewed source by its tag and commit, with dated
evidence linked to the run that generated it.

### Draft release description

> OMIX Exosome Rejuvenation v0.3.0 documents the corrected macaque and mouse
> multi-omics analysis and links to aggregate evidence from the September 2026
> full run. For macaques, the workflow uses an animal-grouped, controls-only
> transcriptomic clock and animal-level treatment contrasts. The macaque
> analysis finds no robust transcriptomic-clock rejuvenation after SRC treatment.
> The plasma and three-tissue mouse comparisons remain exploratory. Confirmed
> plasma–tissue identities and a biological methylation sample map remain unavailable.
> The public demo exercises the software and its output contracts.

### Release checklist

1. Review the snapshot, source and environment caveats, and redistribution scope.
   Verify the included hashes and link the associated report figure sources.
   Confirm that the release bundles [synthetic demo inputs](datasets.md#demo-inputs-and-redistribution)
   while OMIX reviews the request about the historical excerpts.
2. Exact replay of the September run requires the recorded runtime source and
   environment. A new full-profile run under a reviewed commit can assess
   numerical agreement with the archived snapshot. Record its date, source
   revision, and environment alongside the comparison.
3. On release approval, record the actual version and publication date in
   `CITATION.cff`, validate it against [CFF 1.2.0](https://github.com/citation-file-format/citation-file-format/blob/1.2.0/schema-guide.md),
   and confirm the technical checks for the commit selected for tagging.
4. With publication approval, tag that reviewed commit and publish the release
   notes with links to the evidence snapshot and reproduction guide.
