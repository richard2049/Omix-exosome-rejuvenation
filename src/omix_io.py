"""
# omix_io.py

# I/O helpers for OMIX datasets: safe loading of matrices and metadata,
# plus simple guardrails on shapes and sample IDs.
# -----------------------------------------------------------------------------
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Guardrails (conservative defaults for laptop-scale analysis)
# -----------------------------------------------------------------------------
MAX_METADATA_ROWS = 5000
MAX_ALLOWED_SAMPLES = 5000
MAX_EXPECTED_SAMPLE_COLS = 20000   # hard ceiling for "wide" processed matrices
MAX_HEADER_COLS = 200_000
MAX_OUTPUT_COLS = 200_000
DEFAULT_MAX_PARTS = 5


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def find_first_present_column(meta: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(meta.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {col.lower(): col for col in meta.columns}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def assert_metadata_reasonable(meta: pd.DataFrame, max_rows: int = MAX_METADATA_ROWS) -> None:
    if meta.shape[0] > max_rows:
        raise ValueError(
            f"Metadata looks too large for a sample sheet "
            f"({meta.shape[0]} rows > {max_rows}). Likely wrong file."
        )


def _looks_like_xlsx(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except Exception:
        return False


def _read_header_columns_fast(file_path: Path, sep: str = "\t") -> List[str]:
    """Fast header read + early hard-stops to prevent column explosions."""
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        line = f.readline().rstrip("\n")

    cols = [c.strip() for c in line.split(sep)]
    n_samples = max(len(cols) - 1, 0)  # first column is feature ID

    if n_samples > MAX_EXPECTED_SAMPLE_COLS:
        raise ValueError(
            f"{file_path.name} appears to have {n_samples} sample columns "
            f"(> {MAX_EXPECTED_SAMPLE_COLS}). This is almost certainly not the "
            "correct processed matrix file."
        )

    if len(cols) > MAX_HEADER_COLS:
        raise ValueError(
            f"{file_path.name} header has {len(cols)} columns (> {MAX_HEADER_COLS}). "
            "Likely not a proper wide matrix."
        )

    return cols


def detect_sample_id_column_by_overlap(
    meta: pd.DataFrame,
    header_cols: List[str],
    candidates: List[str],
    min_overlap: int = 3,
) -> Optional[str]:
    """Optional utility: choose sample-id col by overlap with matrix header."""
    if meta.empty or not header_cols:
        return None

    header_set = set(map(str, header_cols))
    scored = []

    for c in candidates:
        if c in meta.columns:
            vals = meta[c].astype(str).values
            overlap = sum(v in header_set for v in vals)
            if overlap:
                scored.append((overlap, c))

    if not scored:
        for c in meta.columns:
            vals = meta[c].astype(str).values
            overlap = sum(v in header_set for v in vals)
            if overlap:
                scored.append((overlap, c))

    if not scored:
        return None

    scored.sort(reverse=True)
    best_overlap, best_col = scored[0]
    return best_col if best_overlap >= min_overlap else None


# -----------------------------------------------------------------------------
# Metadata loader (robust to fake/mislabeled Excel)
# -----------------------------------------------------------------------------
def load_omix_metadata(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Loads OMIX metadata safely.
    Handles real Excel and 'fake xlsx' that is actually CSV/TSV/HTML.
    """
    logger.info("Loading OMIX metadata: %s", path)
    suffix = path.suffix.lower()

    # "xlsx" extension but not ZIP -> treat as text-like
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"} and not _looks_like_xlsx(path):
        logger.warning("Excel-like extension but not real XLSX ZIP. Trying CSV/TSV/HTML.")

        for try_fn in (
            lambda: pd.read_csv(path),
            lambda: pd.read_csv(path, sep="\t"),
        ):
            try:
                meta = try_fn()
                meta.columns = [str(c).strip() for c in meta.columns]
                assert_metadata_reasonable(meta)
                return meta
            except Exception:
                pass

        # HTML last, non-fatal if no tables
        try:
            tables = pd.read_html(str(path), flavor="lxml")
            meta = tables[0]
            meta.columns = [str(c).strip() for c in meta.columns]
            assert_metadata_reasonable(meta)
            return meta
        except ValueError:
            logger.warning("No HTML tables found in %s.", path)
        except Exception as e:
            logger.warning("HTML parsing failed for %s: %s", path, str(e))

        raise ValueError(f"Metadata could not be parsed as CSV/TSV/HTML: {path}")

    # Real modern Excel
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        meta = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        meta.columns = [str(c).strip() for c in meta.columns]
        assert_metadata_reasonable(meta)
        return meta

    # Old Excel
    if suffix == ".xls":
        meta = pd.read_excel(path, sheet_name=sheet_name, engine="xlrd")
        meta.columns = [str(c).strip() for c in meta.columns]
        assert_metadata_reasonable(meta)
        return meta

    # CSV/TSV/TXT
    if suffix in {".csv", ".tsv", ".txt"}:
        try:
            meta = pd.read_csv(path)
        except Exception:
            meta = pd.read_csv(path, sep="\t")
        meta.columns = [str(c).strip() for c in meta.columns]
        assert_metadata_reasonable(meta)
        return meta

    # Unknown extension: try Excel if signature matches
    if _looks_like_xlsx(path):
        meta = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        meta.columns = [str(c).strip() for c in meta.columns]
        assert_metadata_reasonable(meta)
        return meta

    # Final fallback: CSV then TSV
    try:
        meta = pd.read_csv(path)
    except Exception:
        meta = pd.read_csv(path, sep="\t")

    meta.columns = [str(c).strip() for c in meta.columns]
    assert_metadata_reasonable(meta)
    return meta


# -----------------------------------------------------------------------------
# Matrix loader (memory-safe, directory-aware)
# -----------------------------------------------------------------------------
def _list_matrix_parts(path: Path, max_parts: int = DEFAULT_MAX_PARTS) -> List[Path]:
    candidates: List[Path] = []
    for pat in ("*.txt", "*.tsv", "*.csv"):
        candidates.extend(path.glob(pat))

    keep = ("count", "counts", "matrix", "expr", "expression", "rna", "transcript")
    drop = ("meta", "metadata", "readme", "annotation", "sample", "info", "download")

    filtered = []
    for f in candidates:
        name = f.name.lower()
        if any(k in name for k in drop):
            continue
        if any(k in name for k in keep):
            filtered.append(f)

    if not filtered:
        candidates = [f for f in candidates if f.is_file()]
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
        filtered = candidates[:max_parts]

    return sorted(filtered)


def _read_one_matrix_file(
    file_path: Path,
    index_col: int = 0,
    allowed_samples: Optional[Set[str]] = None,
    dtype: str = "float32",
    sep: str = "\t",
) -> pd.DataFrame:
    header = _read_header_columns_fast(file_path, sep=sep)
    first_col = header[0] if header else None
    if not first_col:
        raise ValueError(f"Empty header in matrix file: {file_path}")

    usecols = None
    
    if allowed_samples is not None:
        allowed = set(map(str, allowed_samples))
        kept = [first_col] + [c for c in header[1:] if c in allowed]

        if len(kept) <= 1:
            raise ValueError(
                f"No allowed sample columns found in {file_path.name}. "
                "Check sample IDs vs metadata."
            )

        if len(kept) > MAX_ALLOWED_SAMPLES + 1:
            raise ValueError(
                f"Too many columns would be loaded from {file_path.name} "
                f"({len(kept)-1} samples). Likely wrong metadata."
            )

        usecols = kept

    try:
        df = pd.read_csv(
            file_path,
            sep=sep,
            index_col=index_col,
            usecols=usecols,
            engine="c",
        )
    except Exception as e:
        logger.warning("C-engine read failed for %s (%s). Using python engine.", file_path.name, str(e))
        df = pd.read_csv(
            file_path,
            sep=sep,
            index_col=index_col,
            usecols=usecols,
            engine="python",
            on_bad_lines="warn",
        )

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    if dtype is not None:
        numeric_cols = df.columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="raise").astype(dtype)

    if df.shape[1] > MAX_OUTPUT_COLS:
        raise ValueError(
            f"{file_path.name} produced {df.shape[1]} columns (> {MAX_OUTPUT_COLS}). "
            "Likely not a proper wide matrix."
        )

    return df


def load_omix_matrix(
    path: Path,
    index_col: int = 0,
    allowed_samples: Optional[Set[str]] = None,
    dtype: str = "float32",
) -> pd.DataFrame:
    """
    Load OMIX processed matrix from a file or a directory of parts.
    Strong preflight checks to prevent catastrophic RAM allocations.
    """
    if path.is_dir():
        if allowed_samples is None:
            raise ValueError(
                f"Refusing to load directory without allowed_samples: {path}. "
                "Provide metadata-derived sample IDs."
            )

        parts = _list_matrix_parts(path)
        if not parts:
            raise FileNotFoundError(f"No plausible matrix files found in: {path}")

        # Preflight: header-only validation for every part
        for p in parts:
            _read_header_columns_fast(p, sep="\t")

        dfs = [
            _read_one_matrix_file(
                p, index_col=index_col, allowed_samples=allowed_samples, dtype=dtype
            )
            for p in parts
        ]

        out = pd.concat(dfs, axis=1, copy=False)
        out = out.loc[:, ~out.columns.duplicated()]

        if out.shape[1] > MAX_OUTPUT_COLS:
            raise ValueError(
                f"Directory load produced {out.shape[1]} columns (> {MAX_OUTPUT_COLS}). "
                "Likely wrong files were selected."
            )
        return out

    # Single-file case
    return _read_one_matrix_file(
        path, index_col=index_col, allowed_samples=allowed_samples, dtype=dtype
    )


# -----------------------------------------------------------------------------
# Alignment helper (final hard stops)
# -----------------------------------------------------------------------------
def align_matrix_and_metadata(
    matrix: pd.DataFrame,
    meta: pd.DataFrame,
    sample_id_candidates: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align matrix columns to metadata sample IDs with absolute safety checks."""
    assert_metadata_reasonable(meta)

    # Absolute stop: if this triggers, your matrix selection is wrong
    if matrix.shape[1] > MAX_EXPECTED_SAMPLE_COLS:
        raise ValueError(
            f"Matrix has {matrix.shape[1]} columns (> {MAX_EXPECTED_SAMPLE_COLS}). "
            "This is not a normal processed OMIX sample count; likely wrong file."
        )

    sample_col = find_first_present_column(meta, sample_id_candidates)
    if sample_col is None:
        raise ValueError(
            "No sample ID column found in metadata. Tighten sample_id_col_candidates."
        )

    meta = meta.copy()
    meta[sample_col] = meta[sample_col].astype(str)

    mcols = set(matrix.columns)
    common = [s for s in meta[sample_col].values if s in mcols]

    if not common:
        raise ValueError("No overlapping sample IDs between matrix and metadata.")

    if len(common) > MAX_ALLOWED_SAMPLES:
        raise ValueError(
            f"Too many overlapping samples ({len(common)} > {MAX_ALLOWED_SAMPLES}). "
            "Wrong metadata or wrong sample ID column."
        )

    matrix = matrix.loc[:, common]
    meta = meta.set_index(sample_col).loc[common].reset_index()
    meta = meta.rename(columns={sample_col: "sample_id"})

    logger.info("Aligned %d samples.", len(common))
    return matrix, meta