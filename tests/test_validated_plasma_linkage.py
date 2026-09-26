import hashlib
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import OmixPaths, PipelineConfig
from src.linkage_audit import (
    audit_primate_plasma_linkage,
    build_estimability_report,
    load_validated_plasma_linkage_manifest,
)


def _metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    plasma = pd.DataFrame(
        {
            "sample_id": ["FGES_1", "MGES_1", "FV_1", "MV_1"],
            "group": ["GES", "GES", "V", "V"],
            "sex": ["F", "M", "F", "M"],
        }
    )
    primate = pd.DataFrame(
        {
            "animal_id": ["F-GES-1", "M-GES-1", "F-V-1", "M-V-1"] * 2,
            "group": ["O_GES", "O_GES", "O_V", "O_V"] * 2,
            "sex": ["F", "M", "F", "M"] * 2,
            "tissue": ["Liver"] * 4 + ["Kidney"] * 4,
        }
    )
    return plasma, primate


def _write_manifest(tmp_path: Path, **overrides) -> tuple[Path, str]:
    frame = pd.DataFrame(
        {
            "sample_id": ["FGES_1", "MGES_1", "FV_1", "MV_1"],
            "animal_id": ["F-GES-1", "M-GES-1", "F-V-1", "M-V-1"],
            "identity_confirmed": [True] * 4,
            "evidence_source": ["author-supplied cross-modal key"] * 4,
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    path = tmp_path / "plasma_linkage_manifest.csv"
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_validated_manifest_can_open_gate_only_after_full_contract(tmp_path):
    plasma, primate = _metadata()
    path, digest = _write_manifest(tmp_path)

    links = load_validated_plasma_linkage_manifest(
        path,
        expected_sha256=digest,
        plasma_meta=plasma,
        primate_meta=primate,
    )
    linked_plasma = plasma.merge(
        links[["sample_id", "animal_id", "animal_id_confidence"]],
        on="sample_id",
        validate="one_to_one",
    )
    audit = audit_primate_plasma_linkage(primate, linked_plasma)
    estimability = build_estimability_report(
        audit,
        min_samples_for_mediation=4,
        min_overlap_animals=4,
        min_treated_overlap=2,
        min_control_overlap=2,
    )

    assert links["animal_id_confidence"].eq("metadata_exact").all()
    assert links["identity_confirmed"].all()
    assert links["linkage_manifest_sha256"].eq(digest).all()
    assert estimability["tier"] == "fully_linked"
    assert estimability["can_do_mediation"] is True


def test_generic_metadata_confidence_cannot_open_gate():
    plasma, primate = _metadata()
    plasma["animal_id"] = ["F-GES-1", "M-GES-1", "F-V-1", "M-V-1"]
    plasma["animal_id_confidence"] = "metadata"

    audit = audit_primate_plasma_linkage(primate, plasma)

    assert audit["n_overlap_animal_ids"] == 4
    assert audit["n_overlap_animal_ids_high_conf"] == 0
    assert build_estimability_report(
        audit,
        min_samples_for_mediation=4,
        min_overlap_animals=4,
        min_treated_overlap=2,
        min_control_overlap=2,
    )["can_do_mediation"] is False


def test_validated_manifest_rejects_checksum_mismatch(tmp_path):
    plasma, primate = _metadata()
    path, _ = _write_manifest(tmp_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_validated_plasma_linkage_manifest(
            path,
            expected_sha256="0" * 64,
            plasma_meta=plasma,
            primate_meta=primate,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"identity_confirmed": [True, True, False, True]}, "identity_confirmed=true"),
        ({"animal_id": ["F-GES-1", "F-GES-1", "F-V-1", "M-V-1"]}, "not one-to-one"),
        ({"evidence_source": ["", "source", "source", "source"]}, "empty evidence_source"),
    ],
)
def test_validated_manifest_rejects_unconfirmed_or_ambiguous_rows(
    tmp_path, overrides, message
):
    plasma, primate = _metadata()
    path, digest = _write_manifest(tmp_path, **overrides)

    with pytest.raises(ValueError, match=message):
        load_validated_plasma_linkage_manifest(
            path,
            expected_sha256=digest,
            plasma_meta=plasma,
            primate_meta=primate,
        )


def test_validated_manifest_rejects_group_mismatch(tmp_path):
    plasma, primate = _metadata()
    path, digest = _write_manifest(
        tmp_path,
        animal_id=["F-V-1", "M-GES-1", "F-GES-1", "M-V-1"],
    )

    with pytest.raises(ValueError, match="group mismatch"):
        load_validated_plasma_linkage_manifest(
            path,
            expected_sha256=digest,
            plasma_meta=plasma,
            primate_meta=primate,
        )


def test_config_rejects_unverified_high_confidence_labels(tmp_path):
    with pytest.raises(ValueError, match="unverified confidence labels"):
        PipelineConfig(
            primate_bulk=OmixPaths(tmp_path / "matrix.txt"),
            linkage_high_conf_values=["metadata", "metadata_exact"],
        )
