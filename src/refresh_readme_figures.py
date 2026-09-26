"""Refresh only the two linkage-sensitive README figures from full checkpoints.

This is deterministic figure postprocessing, not a full model rerun. Historical
tissue/alignment tables remain byte-identical; linkage is re-audited from raw
metadata. The aging figure is deliberately neither regenerated nor copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data_refresh import read_matrix, sample_columns
from .linkage_audit import audit_primate_plasma_linkage, build_estimability_report
from .report_figures import plot_portfolio_estimability_guardrail, plot_portfolio_multimodal_evidence
from .run_pipeline import build_plasma_metadata_from_columns, _build_mediation_stub


def refresh(baseline: Path, data_root: Path, output: Path, assets: Path | None = None):
    baseline, data_root, output = (p.resolve() for p in (baseline, data_root, output))
    if output.is_relative_to(baseline) or output.is_relative_to(data_root):
        raise ValueError("Figure snapshots must be outside the input directories")
    # These reviewed full-profile checkpoints are not interchangeable with demo
    # or safe-profile outputs. No baseline acceptance values are regenerated.
    clock = pd.read_csv(baseline / "clock_metrics_primates.csv").iloc[0]
    tissue = pd.read_csv(baseline / "rejuvenation_by_tissue.csv")
    if clock.cv_strategy != "GroupKFold(animal_id)" or clock.cv_n_groups != 61 or clock.n_samples != 2058:
        raise ValueError("Expected the reviewed 61-animal, 2058-sample full checkpoint")
    if len(tissue) != 39 or not ((tissue.ci_low <= 0) & (tissue.ci_high >= 0)).all():
        raise ValueError("Expected the reviewed 39-tissue uncertainty checkpoint")
    bulk_path = data_root / "OMIX007580-02.csv"
    plasma_path = data_root / "OMIX007581-01.csv"
    bulk = pd.read_csv(bulk_path).rename(columns={"orig.ident": "animal_id"})
    matrix = read_matrix(plasma_path.read_bytes(), "OMIX007581")
    plasma = build_plasma_metadata_from_columns(sample_columns(matrix, "OMIX007581"))
    audit = audit_primate_plasma_linkage(bulk, plasma)
    gate = build_estimability_report(audit)
    if len(plasma) != 32 or audit["n_overlap_animal_ids"] != 24 or audit["n_overlap_animal_ids_high_conf"] != 0:
        raise ValueError("This refresh requires the reviewed 32-plasma/24-candidate/zero-confirmed state")
    if gate["can_do_mediation"]:
        raise ValueError("Unexpected estimable mediation; review identity evidence first")

    output.mkdir(parents=True, exist_ok=False)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir()
    figures.mkdir()
    retained = ["clock_metrics_primates.csv", "rejuvenation_by_tissue.csv",
                "exosome_alignment_summary.csv", "multimodal_concordance_summary.csv"]
    sources = [bulk_path, plasma_path] + [baseline / name for name in retained]
    for name in retained:
        shutil.copy2(baseline / name, tables / name)
    pd.DataFrame([audit]).to_csv(tables / "linkage_audit.csv", index=False)
    pd.DataFrame([gate]).to_csv(tables / "estimability_report.csv", index=False)
    # A figure-input snapshot, not a substitute for all primary pipeline tables.
    pd.DataFrame([{
        "n_plasma_total": len(plasma),
        "n_mapped_valid_in_bulk": audit["n_overlap_animal_ids"],
        "n_mapped_high_conf": audit["n_overlap_animal_ids_high_conf"],
        "estimable": False,
    }]).to_csv(tables / "linkage_qc_report.csv", index=False)
    _build_mediation_stub(reason="Mediation disabled by config (enable_mediation=False).",
                         tier=gate["tier"], n_overlap_animal_ids=gate["n_overlap_animal_ids"]).to_csv(
                             tables / "mediation_summary.csv", index=False)
    records = [plot_portfolio_multimodal_evidence(tables, figures),
               plot_portfolio_estimability_guardrail(tables, figures)]
    pd.DataFrame([vars(record) for record in records]).to_csv(output / "figure_manifest.csv", index=False)
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "full_checkpoint_plus_metadata_reaudit_no_model_refit",
        "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if assets is not None:
        assets.mkdir(parents=True, exist_ok=True)
        for record, name in zip(records, ["multimodal_evidence_architecture.png", "estimability_guardrail.png"]):
            shutil.copy2(record.path, assets / name)
    for record in records:
        print(record.message, flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New snapshot directory")
    parser.add_argument("--assets-dir", type=Path, help="Optionally replace only the two affected README PNGs")
    args = parser.parse_args()
    refresh(args.baseline_results, args.data_root, args.output_dir, args.assets_dir)


if __name__ == "__main__":
    main()
