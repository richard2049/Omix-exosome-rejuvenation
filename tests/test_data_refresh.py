"""Refresh acceptance checks based on public records and identity invariants."""
import hashlib
import json

import pandas as pd
import pytest

from src.config import OmixPaths, PipelineConfig
from src.data_refresh import Snapshots, candidate_links, parse_biosample, project_index, read_matrix
from src.linkage_audit import audit_primate_plasma_linkage, build_estimability_report
from src.run_pipeline import build_plasma_metadata_from_columns, _causal_gate_reason


def test_public_aliases_cannot_pass_linkage_gate_even_when_all_exist_in_bulk():
    # Independently enumerated study arms: four females and four males per arm.
    samples = [f"{sex}{arm}_{i}" for arm in ("GES", "V", "WT")
               for sex in ("F", "M") for i in range(1, 5)]
    plasma = build_plasma_metadata_from_columns(samples)
    bulk = pd.DataFrame({"animal_id": plasma.animal_id,
                         "group": ["O_GES"] * 8 + ["O_V"] * 8 + ["O_WT"] * 8})
    audit = audit_primate_plasma_linkage(bulk, plasma)
    assert audit["n_overlap_animal_ids"] == 24
    assert audit["n_overlap_animal_ids_high_conf"] == 0
    assert build_estimability_report(audit)["can_do_mediation"] is False


def test_mediation_is_disabled_by_default(tmp_path):
    cfg = PipelineConfig(primate_bulk=OmixPaths(tmp_path / "matrix.txt"))
    assert cfg.enable_mediation is False
    reason = _causal_gate_reason(enable_mediation=cfg.enable_mediation,
                                enable_causal_decomposition=True,
                                estimability_row={"tier": "fully_linked", "can_do_mediation": True})
    assert "disabled" in reason.lower()


def test_biosample_preserves_reported_metadata_without_asserting_animal_identity():
    # Minimal transcription of SAMC4681482 fields, not an invented animal key.
    html = """<table><tr><th>样本编号</th><td>SAMC4681482</td></tr>
    <tr><th>样品名称</th><td>F-GES-1_Hippocampus</td></tr>
    <tr><th>物种名称</th><td>Macaca fascicularis</td></tr>
    <tr><th>描述信息</th><td>Monkey tissue_GES</td></tr>
    <tr><th>Attributes</th><td><table>
    <tr><th>年龄</th><td>21 Year(s)</td></tr>
    <tr><th>性别</th><td>female</td></tr>
    <tr><th>组织器官</th><td>Hippocampus</td></tr></table></td></tr>
    </table><a href='/gsa/browse/CRA022788'>CRA022788</a>""".encode()
    row = parse_biosample(html, "SAMC4681482")
    assert (row["age"], row["age_unit"], row["sex"]) == (21, "year", "female")
    assert row["animal_alias"] == "F-GES-1"
    assert row["animal_id_validated"] == ""
    assert row["gsa_accession"] == "CRA022788"
    with pytest.raises(ValueError, match="accession mismatch"):
        parse_biosample(html, "SAMC0000000")


def test_candidate_table_does_not_multiply_plasma_rows_by_tissue():
    manifest = pd.DataFrame({"biosample_id": ["SAMC1", "SAMC2"],
                             "animal_alias": ["F-GES-1", "F-GES-1"]})
    result = candidate_links(["FGES_1", "FY_1"], manifest,
                             pd.DataFrame({"orig.ident": ["F-GES-1", "F-GES-1"]}))
    assert len(result) == 2
    assert result.plasma_sample_id.is_unique
    assert result.iloc[0].n_biosamples_in_selected_panel == 2
    assert not result.identity_confirmed.any()
    assert result.iloc[1].mapping_status == "unresolved"


def test_duplicate_project_ids_and_matrix_headers_are_rejected():
    row = "<tr><td>SAMC1</td><td>A</td></tr>"
    with pytest.raises(ValueError, match="duplicate"):
        project_index(("<table>" + row * 2 + "</table>").encode())
    with pytest.raises(ValueError, match="Duplicate"):
        read_matrix(b"Protein accession,Gene name,FY_1,FY_1\nP,G,1,2\n", "OMIX007581")


def test_snapshot_replay_verifies_hash_and_url(tmp_path):
    payload = b"source data"
    (tmp_path / "sample.txt").write_bytes(payload)
    record = {"source_url": "https://example.org/sample", "sha256": hashlib.sha256(payload).hexdigest()}
    (tmp_path / "sample.txt.json").write_text(json.dumps(record))
    snapshots = Snapshots(tmp_path, offline=True)
    assert snapshots.get("sample.txt", record["source_url"])[0] == payload
    with pytest.raises(ValueError, match="provenance mismatch"):
        snapshots.get("sample.txt", "https://example.org/different")
    (tmp_path / "sample.txt").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="provenance mismatch"):
        snapshots.get("sample.txt", record["source_url"])
    with pytest.raises(FileNotFoundError, match="Offline snapshot missing"):
        snapshots.get("missing.txt", record["source_url"])
