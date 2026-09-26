import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import OmixPaths, PipelineConfig
from src.run_provenance import begin_run_manifest, finalize_run_manifest


def test_run_manifest_binds_inputs_config_source_and_outputs(tmp_path):
    matrix = tmp_path / "matrix.csv"
    metadata = tmp_path / "metadata.csv"
    matrix.write_text("feature,S1\nG1,2\n", encoding="utf-8")
    metadata.write_text("sample_id,age\nS1,4\n", encoding="utf-8")
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    cfg = PipelineConfig(
        primate_bulk=OmixPaths(matrix, metadata),
        results_dir=results,
        figures_dir=figures,
    )
    repo_root = Path(__file__).resolve().parents[1]

    path = begin_run_manifest(
        cfg,
        repo_root=repo_root,
        argv=["python", "-m", "src.run_pipeline", "--profile", "demo"],
    )
    started = json.loads(path.read_text(encoding="utf-8"))

    assert started["status"] == "started"
    assert started["source"]["git_head"]
    assert len(started["source"]["source_tree_sha256"]) == 64
    matrix_record = next(row for row in started["inputs"] if row["label"] == "primate_bulk.matrix")
    assert matrix_record["sha256"] == hashlib.sha256(matrix.read_bytes()).hexdigest()
    assert started["configuration"]["primate_bulk"]["matrix"] == str(matrix.resolve())

    results.mkdir(exist_ok=True)
    (results / "scientific_result.csv").write_text("estimate\n1.0\n", encoding="utf-8")
    finalize_run_manifest(path, cfg, status="completed")
    completed = json.loads(path.read_text(encoding="utf-8"))

    assert completed["status"] == "completed"
    assert completed["completed_at_utc"]
    assert completed["error"] is None
    assert any(row["path"].endswith("scientific_result.csv") for row in completed["outputs"])
    assert not any(row["path"].endswith("run_manifest.json") for row in completed["outputs"])


def test_run_manifest_preserves_failure_state(tmp_path):
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("feature,S1\nG1,2\n", encoding="utf-8")
    cfg = PipelineConfig(
        primate_bulk=OmixPaths(matrix),
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
    )
    path = begin_run_manifest(
        cfg,
        repo_root=Path(__file__).resolve().parents[1],
        argv=["python", "-m", "src.run_pipeline"],
    )

    error = RuntimeError("intentional diagnostic failure")
    finalize_run_manifest(path, cfg, status="failed", error=error)
    failed = json.loads(path.read_text(encoding="utf-8"))

    assert failed["status"] == "failed"
    assert failed["error"] == {
        "type": "RuntimeError",
        "message": "intentional diagnostic failure",
    }
