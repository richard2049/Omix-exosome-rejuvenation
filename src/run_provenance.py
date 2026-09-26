"""Machine-readable provenance for pipeline executions."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable


RUN_MANIFEST_NAME = "run_manifest.json"
_SOURCE_DIRS = ("src", "config", "tests", "docs", ".github")
_ROOT_SOURCE_FILES = (
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "RESULTS.md",
    "environment.yml",
    "environment-omix007582-r.yml",
)
_PACKAGE_NAMES = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.as_posix()}",
            *arguments,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _source_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for dirname in _SOURCE_DIRS:
        root = repo_root / dirname
        if root.is_dir():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    paths.extend(
        path for name in _ROOT_SOURCE_FILES if (path := repo_root / name).is_file()
    )
    return sorted(set(paths), key=lambda path: path.relative_to(repo_root).as_posix())


def _source_snapshot(repo_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in _source_paths(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        digest = _sha256_file(path)
        records.append({"path": relative, "sha256": digest, "size_bytes": path.stat().st_size})
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")

    try:
        head = _git(repo_root, "rev-parse", "HEAD")
        status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        head = ""
        status = f"git inspection failed: {exc}"

    return {
        "git_head": head,
        "git_dirty": bool(status and not status.startswith("git inspection failed:")),
        "git_status_porcelain": status.splitlines() if status else [],
        "source_tree_sha256": combined.hexdigest(),
        "source_file_count": len(records),
        "source_files": records,
    }


def _configured_input_paths(cfg: Any) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for prefix in (
        "primate_bulk",
        "primate_plasma",
        "primate_methylation",
        "mouse_exosome_bulk",
    ):
        value = getattr(cfg, prefix, None)
        if value is None:
            continue
        for field in ("matrix", "metadata"):
            path = getattr(value, field, None)
            if path is not None:
                candidates.append((f"{prefix}.{field}", Path(path)))

    for field in (
        "clock_fold_assignments",
        "mouse_tissue_mapping_path",
        "plasma_linkage_manifest",
        "gmt_path",
    ):
        path = getattr(cfg, field, None)
        if path is not None:
            candidates.append((field, Path(path)))

    data_root = getattr(cfg, "data_root", None)
    if data_root is not None and bool(getattr(cfg, "enable_subset_validation_block", False)):
        for name in ("OMIX007583-01.zip", "OMIX007586-02.zip"):
            candidates.append((f"subset_validation.{name}", Path(data_root) / name))

    unique: dict[str, tuple[str, Path]] = {}
    for label, path in candidates:
        unique[str(path.resolve()).casefold()] = (label, path.resolve())
    return sorted(unique.values(), key=lambda item: item[0])


def _input_inventory(cfg: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, path in _configured_input_paths(cfg):
        record: dict[str, Any] = {
            "label": label,
            "path": str(path),
            "exists": path.is_file(),
        }
        if path.is_file():
            record.update(
                size_bytes=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        records.append(record)
    return records


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _PACKAGE_NAMES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _output_inventory(paths: Iterable[Path], *, exclude: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in paths:
        if not root.is_dir():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.resolve() == exclude.resolve():
                continue
            records.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return records


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def begin_run_manifest(
    cfg: Any,
    *,
    repo_root: Path,
    argv: Iterable[str] | None = None,
) -> Path:
    """Write the immutable-input and dirty-source record before computation."""
    manifest_path = Path(cfg.results_dir) / RUN_MANIFEST_NAME
    manifest = {
        "schema_version": 1,
        "status": "started",
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "command": list(argv if argv is not None else sys.argv),
        "working_directory": str(Path.cwd().resolve()),
        "repository_root": str(repo_root.resolve()),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "platform": platform.platform(),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "packages": _package_versions(),
        },
        "source": _source_snapshot(repo_root.resolve()),
        "configuration": _jsonable(cfg),
        "inputs": _input_inventory(cfg),
        "outputs": [],
        "error": None,
    }
    _write_manifest(manifest_path, manifest)
    return manifest_path


def finalize_run_manifest(
    manifest_path: Path,
    cfg: Any,
    *,
    status: str,
    error: BaseException | None = None,
) -> None:
    """Finalize a started manifest without concealing failed executions."""
    if status not in {"completed", "failed"}:
        raise ValueError(f"Unsupported run-manifest status: {status}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["completed_at_utc"] = _utc_now()
    manifest["outputs"] = _output_inventory(
        (Path(cfg.results_dir), Path(cfg.figures_dir)),
        exclude=manifest_path,
    )
    if error is not None:
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    _write_manifest(manifest_path, manifest)
