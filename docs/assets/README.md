# Curated README Figures

These three README figures were generated from the reviewed September 2026
full-profile results. The [evidence snapshot](../evidence/2026-09-26/README.md)
contains the associated aggregate tables.

| Figure | Source tables | Scientific boundary |
|---|---|---|
| `aging_rejuvenation_signal.png` | `clock_metrics_primates.csv`; `rejuvenation_by_tissue.csv` | Shows grouped clock validation and uncertainty-aware tissue prioritization; it does not establish tissue-specific rejuvenation |
| `multimodal_evidence_architecture.png` | `rejuvenation_by_tissue.csv`; `linkage_qc_report.csv`; `estimability_report.csv`; `exosome_alignment_summary.csv`; `cross_species_response_alignment_summary.csv`; `multimodal_concordance_summary.csv` | Shows modality-specific analysis paths, estimability gates, and current evidence status; it is not a pooled concordance estimate |
| `estimability_guardrail.png` | `linkage_qc_report.csv`; `estimability_report.csv`; `mediation_summary.csv` | Shows whether mediation may be estimated and how failure is reported; gate passage does not establish causality |

Regenerate the report and update the README figures with:

```bash
python -m src.report_figures --portfolio-assets-dir docs/assets
```

Use reviewed full-profile tables and check
`figures/report/report_figure_manifest.csv` before keeping the regenerated
PNGs. The report is generated after the pipeline finalizes its run inventory.
The historical `src.refresh_readme_figures` utility applies to the earlier
61-animal baseline.
