"""Strict, version-bound ingestion for the released OMIX007580 count matrix.

The public matrix has no feature-identifier column: every header field is a
sample identifier.  One source record is malformed.  This module preserves
all regular records under stable, explicitly anonymous row identifiers and
refuses to generalize that one-record quarantine to any other defect.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Collection, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnonymousCountMatrixContract:
    """Immutable expectations for one exact release of a source matrix."""

    accession: str
    source_sha256: str
    metadata_sha256: str
    expected_samples: int
    expected_rows: int
    quarantine: Mapping[int, str]


@dataclass(frozen=True)
class AnonymousCountMatrixAudit:
    """Machine-readable evidence emitted after a contract-valid load."""

    source_accession: str
    source_sha256: str
    metadata_sha256: str
    retained_rows: int
    retained_samples: int
    quarantined_records: int
    quarantined_physical_lines: str
    feature_identity: str
    row_identifier_scheme: str
    source_validation: str

    def to_record(self) -> dict[str, object]:
        return asdict(self)


OMIX007580_RELEASE_CONTRACT = AnonymousCountMatrixContract(
    accession="OMIX007580-01",
    source_sha256="4c5405720300b52351033951db55c6dd80807c6ea7abaf44125e20fb4bf2a6f5",
    metadata_sha256="4717a905deaf76446077c4e5beb85c8d6d5f4e8536afcc37a893f228cd92e159",
    expected_samples=2059,
    expected_rows=23716,
    quarantine={
        22429: "56f3d244a5023982c993a4640f4d6345d0d17265451cdcf73bc40d44140bd58c",
    },
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def anonymous_feature_id(accession: str, source_sha256: str, physical_line: int) -> str:
    """Return a stable provenance identifier, never a biological annotation."""
    return f"{accession}@{source_sha256[:16]}:line{physical_line:08d}"


def _validated_header(
    header_bytes: bytes,
    expected_sample_ids: Collection[str],
    contract: AnonymousCountMatrixContract,
) -> list[str]:
    if b"\x00" in header_bytes:
        raise ValueError("OMIX007580 header contains NUL bytes")
    try:
        header = header_bytes.decode("utf-8-sig").rstrip("\r\n").split("\t")
    except UnicodeDecodeError as exc:
        raise ValueError("OMIX007580 header is not valid UTF-8") from exc

    expected = [str(sample_id) for sample_id in expected_sample_ids]
    if len(header) != contract.expected_samples:
        raise ValueError(
            f"OMIX007580 header has {len(header)} samples; "
            f"expected {contract.expected_samples}"
        )
    if any(not value for value in header) or len(set(header)) != len(header):
        raise ValueError("OMIX007580 header contains empty or duplicate sample identifiers")
    if len(expected) != len(set(expected)):
        raise ValueError("OMIX007580 metadata contains duplicate sample identifiers")
    if set(header) != set(expected):
        missing = sorted(set(expected) - set(header))[:5]
        unexpected = sorted(set(header) - set(expected))[:5]
        raise ValueError(
            "OMIX007580 header and supplied metadata sample identifiers differ; "
            f"missing_from_matrix={missing}, unexpected_in_matrix={unexpected}"
        )
    return header


def load_omix007580_anonymous_counts(
    path: Path,
    expected_sample_ids: Collection[str],
    *,
    contract: AnonymousCountMatrixContract = OMIX007580_RELEASE_CONTRACT,
    dtype: str = "float32",
) -> tuple[pd.DataFrame, AnonymousCountMatrixAudit]:
    """Load the exact audited release without inventing gene identities.

    The source checksum and the checksum of the single authorized malformed
    physical record bind this behavior to the audited file.  Any other
    structural or numerical defect is fatal.
    """
    source = Path(path)
    source_hash_before = sha256_file(source)
    if source_hash_before != contract.source_sha256:
        raise ValueError(
            "OMIX007580 source checksum changed; the curation contract is version-specific"
        )

    values = np.empty(
        (contract.expected_rows, contract.expected_samples),
        dtype=np.dtype(dtype),
    )
    feature_ids: list[str] = []
    observed_quarantine: set[int] = set()
    retained = 0

    with source.open("rb") as handle:
        sample_ids = _validated_header(handle.readline(), expected_sample_ids, contract)
        for physical_line, record in enumerate(handle, start=2):
            record_hash = hashlib.sha256(record).hexdigest()
            if physical_line in contract.quarantine:
                if record_hash != contract.quarantine[physical_line]:
                    raise ValueError(
                        f"Authorized quarantine record changed at physical line {physical_line}"
                    )
                observed_quarantine.add(physical_line)
                continue

            payload = record.rstrip(b"\r\n")
            if b"\x00" in payload or payload.count(b"\t") + 1 != contract.expected_samples:
                raise ValueError(
                    f"Unexpected structural defect at physical line {physical_line}"
                )
            try:
                numeric = np.fromstring(payload.decode("ascii"), sep="\t", dtype=np.float64)
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Non-ASCII count data at physical line {physical_line}"
                ) from exc
            if numeric.size != contract.expected_samples:
                raise ValueError(f"Invalid numeric record at physical line {physical_line}")
            if (
                not np.isfinite(numeric).all()
                or (numeric < 0).any()
                or (numeric != np.floor(numeric)).any()
            ):
                raise ValueError(f"Invalid count value at physical line {physical_line}")
            if retained >= contract.expected_rows:
                raise ValueError("OMIX007580 contains more regular records than expected")

            values[retained, :] = numeric
            feature_ids.append(
                anonymous_feature_id(contract.accession, contract.source_sha256, physical_line)
            )
            retained += 1

    if observed_quarantine != set(contract.quarantine):
        raise ValueError("The complete authorized quarantine set was not observed")
    if retained != contract.expected_rows:
        raise ValueError(
            f"OMIX007580 retained {retained} regular rows; expected {contract.expected_rows}"
        )
    if sha256_file(source) != source_hash_before:
        raise ValueError("OMIX007580 source changed while it was being read")

    matrix = pd.DataFrame(values, index=feature_ids, columns=sample_ids, copy=False)
    matrix.index.name = "anonymous_feature_id"
    audit = AnonymousCountMatrixAudit(
        source_accession=contract.accession,
        source_sha256=contract.source_sha256,
        metadata_sha256=contract.metadata_sha256,
        retained_rows=retained,
        retained_samples=len(sample_ids),
        quarantined_records=len(observed_quarantine),
        quarantined_physical_lines=";".join(map(str, sorted(observed_quarantine))),
        feature_identity="unknown",
        row_identifier_scheme="source_sha256_and_physical_line",
        source_validation="exact_release_and_quarantine_checksums_verified",
    )
    matrix.attrs["input_audit"] = audit.to_record()
    return matrix, audit
