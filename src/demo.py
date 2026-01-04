import pandas as pd
from pathlib import Path

data_dir = Path("D:/DATA/SRSC/data")

meth_matrix = pd.read_csv(data_dir / "OMIX007582_beta_matrix.csv", index_col=0)
meth_meta   = pd.read_csv(data_dir / "OMIX007582-02.csv")

print("Methylation columns (first 5):", meth_matrix.columns[:5])
print("Metadata columns:", meth_meta.columns.tolist())
print(meth_meta.head())

import pandas as pd
from pathlib import Path

data_dir = Path("D:/DATA/SRSC/data")

meth_matrix = pd.read_csv(data_dir / "OMIX007582_beta_matrix.csv", index_col=0)
meth_meta   = pd.read_csv(data_dir / "OMIX007582-02.csv")

print("First 5 methylation columns:", meth_matrix.columns[:5].tolist())
print("Metadata columns:", meth_meta.columns.tolist())

# Check overlap of column names with the obvious metadata fields
for col in ["OriginalSampleName", "sample", "OriginalSampleName.1"]:
    if col in meth_meta.columns:
        overlap = set(meth_matrix.columns) & set(meth_meta[col].astype(str))
        print(f"{col}: {len(overlap)} overlaps")
        if overlap:
            print("Example overlaps:", list(overlap)[:5])

import pandas as pd
from pathlib import Path

data_dir = Path("D:/DATA/SRSC/data")

meth_matrix = pd.read_csv(data_dir / "OMIX007582_beta_matrix.csv", index_col=0)
meth_meta   = pd.read_csv(data_dir / "OMIX007582-02.csv")

print("Matrix shape:", meth_matrix.shape)
print("Matrix columns (first 5):", meth_matrix.columns[:5].tolist())
print("Metadata columns:", meth_meta.columns.tolist())
print(meth_meta.head())