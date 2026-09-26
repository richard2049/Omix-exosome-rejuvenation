"""Small, restartable audit of public deposits; never changes pipeline inputs.

Fetches three processed matrices and a focused BioSample panel. The project
index is complete; detailed metadata coverage is explicitly recorded. No naming
match is promoted to cross-modal animal identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

PROJECT = "PRJCA035748"
BASE = "https://ngdc.cncb.ac.cn"
MATRICES = {
    "OMIX007581": ("csv", ",", "plasma_proteomics"),
    "OMIX009654": ("txt", "\t", "candidate_exosome_proteomics"),
    "OMIX009655": ("txt", "\t", "candidate_exosome_metabolomics"),
}


class TableRows(HTMLParser):
    """Read HTML table cells, including nested BioSample attribute tables."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row:
            self.rows.append(self.row)
            self.row = []


def table_rows(payload: bytes) -> list[list[str]]:
    parser = TableRows()
    parser.feed(payload.decode("utf-8-sig"))
    return parser.rows


def project_index(payload: bytes) -> pd.DataFrame:
    rows = [{"biosample_id": row[0], "source_sample_name": row[1]}
            for row in table_rows(payload)
            if len(row) == 2 and re.fullmatch(r"SAMC\d+", row[0])]
    frame = pd.DataFrame(rows)
    if frame.empty or frame.biosample_id.duplicated().any():
        raise ValueError("Project index is empty or contains duplicate BioSample IDs.")
    if frame.source_sample_name.eq("").any():
        raise ValueError("Project index contains unnamed specimens.")
    return frame.sort_values("biosample_id").reset_index(drop=True)


def parse_biosample(payload: bytes, accession: str) -> dict:
    fields = {row[0].lower(): row[1] for row in table_rows(payload) if len(row) == 2}

    def pick(*keys):
        return next((fields[k.lower()] for k in keys if fields.get(k.lower())), "")

    found = pick("Accession", "Sample accession", "样本编号")
    if found != accession:
        raise ValueError(f"BioSample accession mismatch: expected {accession}, got {found!r}")
    name = pick("Sample name", "样品名称")
    organism = pick("Organism", "Organism name", "物种名称", "来源生物")
    age_raw = pick("Age", "年龄")
    age_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s+(Year|Month|Week|Day)\(s\)", age_raw, re.I)
    alias = re.match(r"^([FM]-(?:GES|WT|V)-\d+)(?:_|-|$)", name)
    gsa = sorted(set(re.findall(r"\bCRA\d+\b", payload.decode("utf-8-sig"))))
    if not name or not organism:
        raise ValueError(f"Missing sample name or organism in {accession}")
    return {
        "biosample_id": accession, "source_sample_name": name,
        "organism": organism, "sex": pick("Sex", "性别"),
        "age_raw": age_raw, "age": float(age_match[1]) if age_match else None,
        "age_unit": age_match[2].lower() if age_match else "",
        "tissue": pick("Tissue", "组织器官"),
        "treatment_description": pick("Description", "描述信息"),
        "animal_alias": alias[1] if alias else "",
        "alias_source": "sample_name" if alias else "unresolved",
        "animal_id_validated": "", "cross_modal_identity": "not_confirmed",
        "gsa_accession": ";".join(gsa), "bioproject": PROJECT,
    }


class Snapshots:
    """Cache exact responses with immutable payloads and retrieval metadata."""

    def __init__(self, root: Path, offline: bool):
        self.root = root
        self.offline = offline
        root.mkdir(parents=True, exist_ok=True)

    def get(self, name: str, url: str) -> tuple[bytes, dict]:
        path = self.root / name
        record_path = self.root / (name + ".json")
        if path.exists():
            payload = path.read_bytes()
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["source_url"] != url or record["sha256"] != hashlib.sha256(payload).hexdigest():
                raise ValueError(f"Snapshot provenance mismatch: {path}")
            return payload, record
        if self.offline:
            raise FileNotFoundError(f"Offline snapshot missing: {path}")
        request = Request(url, headers={"User-Agent": "SRSC-public-data-audit/1.0"})
        with urlopen(request, timeout=30) as response:
            payload = response.read(5_000_001)
        if len(payload) > 5_000_000:
            raise ValueError(f"Response exceeds small-audit size limit: {url}")
        record = {"source_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                  "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                  "snapshot": name}
        with path.open("xb") as handle:
            handle.write(payload)
        with record_path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        return payload, record


def read_matrix(payload: bytes, accession: str) -> pd.DataFrame:
    sep = MATRICES[accession][1]
    header = next(csv.reader(io.StringIO(payload.decode("utf-8-sig")), delimiter=sep))
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate column names in {accession}")
    frame = pd.read_csv(io.BytesIO(payload), sep=sep)
    required = {"COMP ID", "COMPOUND Name"} if accession == "OMIX009655" else {"Protein accession", "Gene name"}
    if not required.issubset(frame):
        raise ValueError(f"Unexpected matrix schema for {accession}")
    return frame


def sample_columns(frame: pd.DataFrame, accession: str) -> list[str]:
    pattern = {"OMIX007581": r"[FM](?:Y|V|WT|GES)_\d+",
               "OMIX009654": r"(?:F3|WT)_\d+", "OMIX009655": r"sample-\d+"}[accession]
    columns = [c for c in frame if re.fullmatch(pattern, c)]
    if not columns:
        raise ValueError(f"No recognized sample columns in {accession}")
    return columns


def candidate_links(plasma_columns: list[str], manifest: pd.DataFrame,
                    bulk: pd.DataFrame | None) -> pd.DataFrame:
    """One row per plasma sample, never one row per tissue or BioSample."""
    if len(plasma_columns) != len(set(plasma_columns)):
        raise ValueError("Duplicate plasma sample identifiers")
    if bulk is not None and "orig.ident" not in bulk:
        raise ValueError("Bulk metadata requires orig.ident for candidate existence checks")
    bulk_ids = set(bulk["orig.ident"].dropna().astype(str)) if bulk is not None else set()
    rows = []
    for sample in plasma_columns:
        match = re.fullmatch(r"([FM])(GES|WT|V)_(\d+)", sample)
        alias = "-".join(match.groups()) if match else ""
        candidates = manifest.loc[manifest.animal_alias.eq(alias)] if alias else manifest.iloc[:0]
        rows.append({"plasma_sample_id": sample, "candidate_animal_alias": alias,
                     "bulk_alias_present": alias in bulk_ids if bulk is not None else None,
                     "biosample_ids": ";".join(sorted(candidates.biosample_id.unique())),
                     "n_biosamples_in_selected_panel": len(candidates),
                     "mapping_status": "candidate" if alias else "unresolved",
                     "identity_confirmed": False,
                     "missing_author_key": "plasma_sample_id_to_animal_id_and_sampling_timepoint"})
    return pd.DataFrame(rows)


def run_refresh(output_dir: Path, data_root: Path | None = None, offline: bool = False) -> dict:
    output_dir = output_dir.resolve()
    if data_root is not None and output_dir.is_relative_to(data_root.resolve()):
        raise ValueError("Audit outputs must be outside the source data root")
    if any(part.lower() == "raw" for part in output_dir.parts):
        raise ValueError("Audit outputs must not be written under raw data")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = Snapshots(output_dir / "sources", offline)
    provenance, matrices, inventory = [], {}, []
    for accession, (suffix, _, role) in MATRICES.items():
        filename = f"{accession}-01.{suffix}"
        payload, record = snapshots.get(filename, f"https://download.cncb.ac.cn/OMIX/{accession}/{filename}")
        provenance.append(record)
        frame = read_matrix(payload, accession)
        columns = sample_columns(frame, accession)
        matrices[accession] = frame
        id_col = "COMP ID" if accession == "OMIX009655" else "Protein accession"
        numeric = frame[columns].apply(pd.to_numeric, errors="raise")
        inventory.append({"accession": accession, "role": role, "feature_rows": len(frame),
                          "sample_identifiers": len(columns), "total_columns": len(frame.columns),
                          "sample_names": ";".join(columns), "sha256": record["sha256"],
                          "duplicate_feature_ids": int(frame[id_col].duplicated().sum()),
                          "missing_feature_ids": int(frame[id_col].isna().sum()),
                          "missing_value_fraction": float(numeric.isna().mean().mean()),
                          "source_url": record["source_url"]})
        print(f"Audited {accession}: {len(frame)} rows, {len(columns)} sample identifiers", flush=True)

    payload, record = snapshots.get(f"{PROJECT}.html", f"{BASE}/bioproject/browse/{PROJECT}")
    provenance.append(record)
    index = project_index(payload)
    # One tissue for each intervention alias, all PBMC samples, and two mouse
    # examples. This is a provenance panel, not an analytical sample selection.
    selected = index.loc[index.source_sample_name.str.fullmatch(r"[FM]-(?:GES|WT|V)-\d+_Hippocampus|[GWVY][1-8]")
                         | index.biosample_id.isin(["SAMC4815734", "SAMC4844342"])].copy()
    if selected.empty:
        raise ValueError("No records matched the documented BioSample panel")

    def fetch_sample(row):
        accession, name = row
        payload, record = snapshots.get(f"{accession}.html", f"{BASE}/biosample/browse/{accession}")
        parsed = parse_biosample(payload, accession)
        if parsed["source_sample_name"] != name:
            raise ValueError(f"Project/BioSample name mismatch for {accession}")
        parsed.update({k: record[k] for k in ["source_url", "retrieved_at", "sha256"]})
        return parsed, record

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for parsed, record in pool.map(fetch_sample, selected.itertuples(index=False, name=None)):
            records.append(parsed)
            provenance.append(record)
            if len(records) % 10 == 0:
                print(f"BioSample panel: {len(records)}/{len(selected)}", flush=True)
    manifest = pd.DataFrame(records).sort_values("biosample_id")
    if manifest.biosample_id.duplicated().any():
        raise ValueError("Duplicate BioSample rows in manifest")
    index["details_retrieved"] = index.biosample_id.isin(manifest.biosample_id)
    bulk = pd.read_csv(data_root / "OMIX007580-02.csv") if data_root is not None else None
    plasma = matrices["OMIX007581"]
    exo = matrices["OMIX009654"]
    links = candidate_links(sample_columns(plasma, "OMIX007581"), manifest, bulk)
    # Exact gene symbols are a descriptive feature overlap, not orthology or
    # evidence of sample identity. Do not split ambiguous multi-gene annotations.
    genes = sorted(set(plasma["Gene name"].dropna()) & set(exo["Gene name"].dropna()))
    overlap = {"exact_gene_annotation_overlap": len(genes),
               "exact_protein_accession_overlap": len(set(plasma["Protein accession"].dropna()) & set(exo["Protein accession"].dropna())),
               "exact_sample_id_overlap": len(set(sample_columns(plasma, "OMIX007581")) & set(sample_columns(exo, "OMIX009654")))}
    summary = {"project": PROJECT, "project_biosamples": len(index),
               "detailed_biosamples": len(manifest), "panel_scope": "intervention hippocampus; all PBMC; two mouse examples",
               "plasma_candidates": int(links.mapping_status.eq("candidate").sum()),
               "plasma_identity_confirmed": 0, **overlap}
    if data_root is not None:
        local = data_root / "OMIX007581-01.csv"
        summary["local_plasma_matches_download"] = hashlib.sha256(local.read_bytes()).hexdigest() == inventory[0]["sha256"]
        summary["bulk_metadata_sha256"] = hashlib.sha256((data_root / "OMIX007580-02.csv").read_bytes()).hexdigest()
    for filename, frame in [("dataset_inventory.csv", pd.DataFrame(inventory)),
                            ("project_index.csv", index), ("sample_manifest.csv", manifest),
                            ("plasma_linkage_candidates.csv", links),
                            ("shared_gene_annotations.csv", pd.DataFrame({"gene_annotation": genes})),
                            ("source_inventory.csv", pd.DataFrame(provenance))]:
        frame.to_csv(output_dir / filename, index=False)
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/data_refresh"))
    parser.add_argument("--data-root", type=Path, help="Existing full OMIX inputs; read-only")
    parser.add_argument("--offline", action="store_true", help="Rebuild tables only from verified cached snapshots")
    args = parser.parse_args()
    run_refresh(args.output_dir, args.data_root, args.offline)


if __name__ == "__main__":
    main()
