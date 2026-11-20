"""Generate the repository's synthetic demo inputs without reading source data."""

import argparse
import csv
from pathlib import Path
from random import Random


def write_table(path, header, rows, delimiter=","):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def generate(output_dir: Path) -> None:
    rng = Random(20261120)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for group, base_age in (("Y_C", 5), ("M_C", 12), ("O_V", 20), ("O_WT", 20), ("O_GES", 20)):
        for replicate, sex in enumerate(("F", "M", "F"), 1):
            animal = f"DEMO_{group}_{replicate}"
            for tissue in ("Heart", "Liver"):
                metadata.append((f"{animal}_{tissue}", animal, tissue, base_age + replicate - 2, group, sex))
    write_table(
        output_dir / "OMIX007580-02_example.csv",
        ["sample", "animal_id", "tissue", "age", "group", "sex"],
        metadata,
    )
    write_table(
        output_dir / "OMIX007580_01_example.txt",
        ["gene_id", *(sample[0] for sample in metadata)],
        (
            [f"DEMO_GENE_{feature:03d}", *(
                30 + feature * 3 + age * (1 + feature % 4) + rng.randrange(30)
                for _, _, _, age, _, _ in metadata
            )]
            for feature in range(1, 25)
        ),
        delimiter="\t",
    )

    plasma_samples = [
        (f"{sex}{group}_{replicate}", stage)
        for stage, group in enumerate(("Y", "V", "WT", "GES"))
        for sex in ("F", "M")
        for replicate in range(91, 95)
    ]
    write_table(
        output_dir / "OMIX007581-01_example.csv",
        ["Protein accession", "Gene name", *(sample for sample, _ in plasma_samples)],
        (
            [f"DEMO_PROTEIN_{feature:03d}", f"DEMO_SYMBOL_{feature:03d}", *(
                100 + feature * 5 + stage * (feature % 11) + rng.randrange(40)
                for _, stage in plasma_samples
            )]
            for feature in range(1, 65)
        ),
    )

    # Deliberately unmatched identifiers exercise the missing sample-map path.
    write_table(
        output_dir / "OMIX007582-02_example.csv",
        ["OriginalSampleName", "sample", "tissue", "age", "group", "OriginalSampleName.1"],
        (
            (f"DEMO_METH_{i:03d}", sample, tissue, age, group, f"DEMO_METH_{i:03d}")
            for i, (sample, _, tissue, age, group, _) in enumerate(metadata[:6], 1)
        ),
    )
    write_table(
        output_dir / "OMIX007582_beta_matrix_example.csv",
        ["CpG", *(f"DEMO_ARRAY_R{i:02d}C01" for i in range(1, 7))],
        (
            [f"cgDEMO{site:05d}", *(f"{rng.randint(1, 999) / 1000:.3f}" for _ in range(6))]
            for site in range(1, 25)
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    generate(parser.parse_args().output_dir)
