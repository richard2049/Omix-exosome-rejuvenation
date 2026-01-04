from pathlib import Path
import pandas as pd

matrix_path = Path(r"D:\DATA\SRSC\data\OMIX009283-01.txt")
out_meta = Path(r"D:\DATA\SRSC\data\OMIX009283_metadata.csv")

# Read only header
cols = pd.read_csv(matrix_path, sep="\t", nrows=0).columns.tolist()
sample_ids = [c for c in cols if c.lower() not in {"gene", "genes", "gene_id", "symbol"}]

# Minimal metadata
meta = pd.DataFrame({"sample_id": sample_ids})

# Best-effort inference from sample names (optional)
# If your columns contain tissue/group info like "Liver_Exosome_1"
def infer_group(s):
    s_low = s.lower()
    if "exo" in s_low or "exosome" in s_low or "hmpc" in s_low:
        return "Exosome"
    if "control" in s_low or "saline" in s_low or "wtc" in s_low:
        return "Control"
    return pd.NA

meta["group"] = meta["sample_id"].map(infer_group)

out_meta.write_text(meta.to_csv(index=False), encoding="utf-8")
print("Wrote:", out_meta)