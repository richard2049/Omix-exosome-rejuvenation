# Curated README Figures

This directory contains exactly three interpretation-facing figure snapshots
for the public README. They are generated from reviewed full-profile result
tables, not assembled manually and not selected from the diagnostic plot set.

| Figure | Source tables | Scientific boundary |
|---|---|---|
| `aging_rejuvenation_signal.png` | `clock_metrics_primates.csv`; `rejuvenation_by_tissue.csv` | Shows grouped clock validation and uncertainty-aware tissue prioritization; it does not establish tissue-specific rejuvenation |
| `multimodal_evidence_architecture.png` | `rejuvenation_by_tissue.csv`; `linkage_qc_report.csv`; `estimability_report.csv`; `exosome_alignment_summary.csv`; `cross_species_response_alignment_summary.csv`; `multimodal_concordance_summary.csv` | Shows modality-specific analysis paths, estimability gates, and current evidence status; it is not a pooled concordance estimate |
| `estimability_guardrail.png` | `linkage_qc_report.csv`; `estimability_report.csv`; `mediation_summary.csv` | Shows whether mediation may be estimated and how failure is reported; gate passage does not establish causality |

Regenerate both the complete report layer and these public snapshots with:

```bash
python -m src.report_figures --portfolio-assets-dir docs/assets
```

Only run the publication step after reviewing full-profile result tables and
the generated `figures/report/report_figure_manifest.csv`. The remaining report
and diagnostic figures stay generated and are not part of the README image set.

## Current full-profile publication

The three snapshots were regenerated together from the corrected September
2026 full-profile run. They preserve the existing layout and styling while
showing the current controls-only clock, animal-level treatment contrasts,
zero confirmed plasma links, three-tissue response-alignment profile, and
structured non-estimability of mediation.

The older `src.refresh_readme_figures` command remains a historical,
linkage-only checkpoint utility. It deliberately requires the former
61-animal baseline and must not be used to publish current primary results.
