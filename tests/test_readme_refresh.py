"""Regression checks for claim-correct figure refresh without layout changes."""
import pandas as pd
import pytest

from src import report_figures
from src.refresh_readme_figures import refresh


def test_multimodal_figure_counts_confirmed_overlap_not_existing_aliases(tmp_path, monkeypatch):
    pd.DataFrame([{"n_plasma_total": 32, "n_mapped_valid_in_bulk": 24,
                   "n_mapped_high_conf": 0, "estimable": False}]).to_csv(
                       tmp_path / "linkage_qc_report.csv", index=False)
    pd.DataFrame([{"n_overlap_animal_ids": 0, "can_do_mediation": False}]).to_csv(
        tmp_path / "estimability_report.csv", index=False)
    captured = []
    save = report_figures._save

    def inspect_and_save(fig, path):
        captured.extend(text.get_text() for ax in fig.axes for text in ax.texts)
        save(fig, path)

    monkeypatch.setattr(report_figures, "_save", inspect_and_save)
    record = report_figures.plot_portfolio_multimodal_evidence(tmp_path, tmp_path / "figures")
    assert "valid links=0" in record.message
    assert any("Gate blocked: 0/32 valid cross-modal links" in text for text in captured)
    assert not any("24/32 valid" in text for text in captured)


def test_multimodal_figure_uses_primary_tissues_and_response_alignment(tmp_path, monkeypatch):
    rows = []
    for tissue in ("Kidney", "Liver", "Muscle"):
        rows.extend(
            [
                {
                    "tissue": tissue,
                    "contrast": "SRC_vs_vehicle",
                    "is_primary": True,
                    "estimable": True,
                    "ci_low": -1.0 if tissue != "Kidney" else 0.1,
                    "ci_high": 1.0,
                },
                {
                    "tissue": tissue,
                    "contrast": "WTC_vs_vehicle",
                    "is_primary": False,
                    "estimable": True,
                    "ci_low": -1.0,
                    "ci_high": 1.0,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(tmp_path / "rejuvenation_by_tissue.csv", index=False)
    pd.DataFrame(
        [
            {
                "estimable": True,
                "n_common_tissues": 3,
                "cosine_similarity": 0.731,
                "cosine_permutation_p_value": 0.285,
            }
        ]
    ).to_csv(tmp_path / "cross_species_response_alignment_summary.csv", index=False)

    captured = []
    save = report_figures._save

    def inspect_and_save(fig, path):
        captured.extend(text.get_text() for ax in fig.axes for text in ax.texts)
        save(fig, path)

    monkeypatch.setattr(report_figures, "_save", inspect_and_save)
    record = report_figures.plot_portfolio_multimodal_evidence(
        tmp_path, tmp_path / "figures"
    )

    assert "transcriptomic tissues=3" in record.message
    assert any("3 tissues; 1 nominal CIs exclude zero" in text for text in captured)
    assert any("C=0.73, permutation p=0.285" in text for text in captured)
    assert not any("6 tissues" in text for text in captured)


def test_readme_refresh_rejects_demo_before_writing_outputs(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    pd.DataFrame([{"cv_strategy": "KFold", "cv_n_groups": 0,
                   "n_samples": 30}]).to_csv(baseline / "clock_metrics_primates.csv", index=False)
    pd.DataFrame([{"ci_low": -1, "ci_high": 1}]).to_csv(
        baseline / "rejuvenation_by_tissue.csv", index=False)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="reviewed 61-animal"):
        refresh(baseline, tmp_path / "raw", output)
    assert not output.exists()
