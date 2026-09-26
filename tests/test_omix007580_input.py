from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.omix007580 import (
    AnonymousCountMatrixContract,
    anonymous_feature_id,
    load_omix007580_anonymous_counts,
    sha256_file,
)


def _record_sha(record: bytes) -> str:
    return hashlib.sha256(record).hexdigest()


def _write_source(tmp_path: Path, records: list[bytes]) -> Path:
    path = tmp_path / "OMIX007580-01.txt"
    path.write_bytes(b"s1\ts2\n" + b"".join(records))
    return path


def _contract(
    source: Path,
    *,
    expected_rows: int,
    quarantine: dict[int, str] | None = None,
) -> AnonymousCountMatrixContract:
    return AnonymousCountMatrixContract(
        accession="OMIX007580-01",
        source_sha256=sha256_file(source),
        metadata_sha256="metadata-test-sha256",
        expected_samples=2,
        expected_rows=expected_rows,
        quarantine=quarantine or {},
    )


def test_contract_loads_all_regular_rows_and_only_authorized_quarantine(tmp_path: Path):
    bad_record = b"corrupt\trecord\textra\n"
    source = _write_source(tmp_path, [b"1\t2\n", bad_record, b"3\t4\n"])
    source_hash_before = sha256_file(source)
    contract = _contract(
        source,
        expected_rows=2,
        quarantine={3: _record_sha(bad_record)},
    )

    matrix, audit = load_omix007580_anonymous_counts(
        source,
        ["s2", "s1"],
        contract=contract,
    )

    assert matrix.shape == (2, 2)
    assert matrix.columns.tolist() == ["s1", "s2"]
    assert matrix.index.tolist() == [
        anonymous_feature_id(contract.accession, contract.source_sha256, 2),
        anonymous_feature_id(contract.accession, contract.source_sha256, 4),
    ]
    np.testing.assert_array_equal(matrix.to_numpy(), [[1, 2], [3, 4]])
    assert matrix.dtypes.astype(str).unique().tolist() == ["float32"]
    assert audit.retained_rows == 2
    assert audit.retained_samples == 2
    assert audit.quarantined_physical_lines == "3"
    assert audit.feature_identity == "unknown"
    assert matrix.attrs["input_audit"] == audit.to_record()
    assert sha256_file(source) == source_hash_before


def test_contract_rejects_changed_source_checksum(tmp_path: Path):
    source = _write_source(tmp_path, [b"1\t2\n"])
    contract = _contract(source, expected_rows=1)
    source.write_bytes(source.read_bytes() + b"3\t4\n")

    with pytest.raises(ValueError, match="source checksum changed"):
        load_omix007580_anonymous_counts(source, ["s1", "s2"], contract=contract)


def test_contract_rejects_changed_quarantine_record(tmp_path: Path):
    bad_record = b"corrupt\trecord\textra\n"
    source = _write_source(tmp_path, [b"1\t2\n", bad_record])
    contract = _contract(
        source,
        expected_rows=1,
        quarantine={3: "0" * 64},
    )

    with pytest.raises(ValueError, match="quarantine record changed"):
        load_omix007580_anonymous_counts(source, ["s1", "s2"], contract=contract)


def test_contract_rejects_unplanned_structural_defect(tmp_path: Path):
    source = _write_source(tmp_path, [b"1\t2\n", b"3\t4\t5\n"])
    contract = _contract(source, expected_rows=2)

    with pytest.raises(ValueError, match="Unexpected structural defect"):
        load_omix007580_anonymous_counts(source, ["s1", "s2"], contract=contract)


@pytest.mark.parametrize("record", [b"-1\t2\n", b"1.5\t2\n", b"nan\t2\n", b"inf\t2\n"])
def test_contract_rejects_invalid_count_values(tmp_path: Path, record: bytes):
    source = _write_source(tmp_path, [record])
    contract = _contract(source, expected_rows=1)

    with pytest.raises(ValueError, match="Invalid count value"):
        load_omix007580_anonymous_counts(source, ["s1", "s2"], contract=contract)


def test_contract_rejects_metadata_header_mismatch(tmp_path: Path):
    source = _write_source(tmp_path, [b"1\t2\n"])
    contract = _contract(source, expected_rows=1)

    with pytest.raises(ValueError, match="sample identifiers differ"):
        load_omix007580_anonymous_counts(source, ["s1", "different"], contract=contract)


def test_contract_rejects_duplicate_metadata_identifiers(tmp_path: Path):
    source = _write_source(tmp_path, [b"1\t2\n"])
    contract = _contract(source, expected_rows=1)

    with pytest.raises(ValueError, match="duplicate sample identifiers"):
        load_omix007580_anonymous_counts(source, ["s1", "s1"], contract=contract)
