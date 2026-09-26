# Group Label Crosswalk

This note aligns article cohort labels with repository and OMIX labels as far
as the currently observable evidence supports.

## Evidence basis

- Primary publication:
  - `https://doi.org/10.1016/j.cell.2025.05.021`
- Repository metadata:
  - `data/RAW/data/OMIX007580-02.csv`
  - `data/RAW/data/OMIX007581-01.csv`
  - `data/RAW/data/OMIX007582-02.csv`
  - `data/RAW/data/OMIX007583-01.zip`
  - `data/RAW/data/OMIX007586-02.zip`
- Public source records:
  - BioProject `PRJCA030875`: `https://ngdc.cncb.ac.cn/bioproject/browse/PRJCA030875`
  - Article preview: `https://www.sciencedirect.com/science/article/abs/pii/S0092867425005719`
  - DOI landing: `https://doi.org/10.1016/j.cell.2025.05.021`

## Article cohort definitions

From the paper text:

- `A1`: `3-5 years`
- `A2`: `10-12 years`
- `A3`: `16-19 years`
- `A4`: `19-23 years`

The paper also states that the `A4` group was randomly divided into:

- `A4-Ctrl`
- `A4-WTC`
- `A4-SRC`

## Repo / OMIX cohort definitions

From `data/RAW/data/OMIX007580-02.csv` unique-animal metadata:

- `Y_C`: ages `4-5`
- `M_C`: ages `10-12`
- `O_C`: ages `16-18`
- `O_V`: ages `19-23`
- `O_WT`: ages `19-22`
- `O_GES`: ages `19-22`

## Crosswalk

| Article label | Repo / OMIX label | Confidence | Rationale |
|---|---|---|---|
| `A1-Ctrl` | `Y_C` | `high` | Age range matches directly: article `3-5 years`, repo `4-5 years`. |
| `A2-Ctrl` | `M_C` | `high` | Age range matches directly: article `10-12 years`, repo `10-12 years`. |
| `A3-Ctrl` | `O_C` | `high` | Age range is strongly consistent: article `16-19 years`, repo `16-18 years`. |
| `A4-WTC` | `O_WT` | `high` | The paper uses `WTC`; OMIX uses `WT`; BioProject `PRJCA030875` includes biosamples such as `WT-MSC-F-1`. Ages also match the article `A4` range. |
| `A4-SRC` | `O_GES` | `high-confidence inferred` | The paper uses `SRC`; BioProject `PRJCA030875` includes biosamples such as `GESMSC-F-1`, `GESMSC-F2`, and `GESMSC-M1`. This strongly supports `GES` as the OMIX/BioProject encoding of the genetically enhanced senescence-resistant treatment arm, even though the article text does not use the literal string `GES`. |
| `A4-Ctrl` | `O_V` | `high-confidence inferred` | The paper states `A4-Ctrl` received saline. BioProject `PRJCA030875` includes biosamples such as `O-V-F-1` and `O-V-M-1`, which strongly supports `V` as the old vehicle control arm. |

## Plasma label crosswalk

Group-label interpretation and individual identity are distinct. The table
below describes likely treatment groups; it does not validate shared animal
numbering across assays. Since the September 2026 refresh, the pipeline marks
name-derived animal links `inferred`, including V, WT, and GES candidates.
The earlier high-confidence *group* crosswalk does not override that status.

From `data/RAW/data/OMIX007581-01.csv`, the plasma columns encode sex and treatment/group:

| Plasma pattern | Likely repo group | Confidence | Notes |
|---|---|---|---|
| `FY_*` | young female control subset | `medium` | `F` and `Y` are consistent with female / young; exact article cohort label is not stated in plasma headers. |
| `MY_*` | young male control subset | `medium` | Same limitation as above. |
| `FV_*` / `MV_*` | `O_V` | `high` | Matches repo-side deterministic mapping already used in the pipeline. |
| `FWT_*` / `MWT_*` | `O_WT` | `high` | Matches repo-side deterministic mapping already used in the pipeline. |
| `FGES_*` / `MGES_*` | `O_GES` | `high` | Matches repo-side deterministic mapping already used in the pipeline. |

## Subset-specific naming clues

### `OMIX007586-02.zip` hippocampus subset

`sample.info.csv` in `OMIX007586-02.zip` is the strongest naming bridge currently available inside the repo because it directly pairs article-style cohort labels with file-level sample names:

| Sample naming in hippocampus subset | Treat column | Group column | Crosswalk implication |
|---|---|---|---|
| `WT-MSC-F-1`, `WT-MSC-M-4` | `WTC` | `A4_WTC` | Directly supports `A4-WTC <-> O_WT`. |
| `GESMSC-F-1`, `GESMSC-M4` | `SRC` | `A4_SRC` | Directly supports `A4-SRC <-> O_GES`. |
| `O-V-F-1`, `O-V-M-4` | `Ctrl` | `A4_Ctrl` | Directly supports `A4-Ctrl <-> O_V`. |
| `14-O-F-1`, `46-O-M-HIP` | `Ctrl` | `A3_Ctrl` | Supports the old-control-stage mapping `A3-Ctrl <-> O_C`. |
| `53-M-M-HIP`, `58-MF` | `Ctrl` | `A2_Ctrl` | Supports `A2-Ctrl <-> M_C`. |
| `01-Y-M-HIP`, `11-YF` | `Ctrl` | `A1_Ctrl` | Supports `A1-Ctrl <-> Y_C`. |

This file upgrades the current crosswalk from age-based plausibility to direct within-dataset naming evidence for the core arms.

### `OMIX007583-01.zip` ovary subset

`sample.info.csv` in `OMIX007583-01.zip` is less explicit at the article-label level, but still useful:

| Ovary subset label | Treat column | Group column | Crosswalk implication |
|---|---|---|---|
| `F-V-2-O`, `F-V-4-O` | `V` | `O_V` | Confirms use of `V` inside the ovary subset for the old intervention-era control arm. |
| `F-WT-1-O`, `F-WT-4-O` | `WT` | `O_WT` | Confirms use of `WT` inside the ovary subset. |
| `F-GES-1-O`, `F-GES-4-O` | `GES` | `O_GES` | Confirms use of `GES` inside the ovary subset. |
| `12-OF-ovary`, `31-OF-ovary` | `C` | `O` | Supports female-only old control samples corresponding to the broader `O_C` stage. |
| `Ovary 56 M`, `Ovary 58 M` | `C` | `M` | Supports middle-aged female ovary controls corresponding to `M_C`. |
| `Ovary 11 Y`, `Ovary 37 Y` | `C` | `Y` | Supports young female ovary controls corresponding to `Y_C`. |

This ovary subset does not use the article `A1/A2/A3/A4` labels directly, so it is supportive rather than canonical.

## Methylation label crosswalk

From `data/RAW/data/OMIX007582-02.csv`:

- `O_V`, `O_WT`, and `O_GES` are present directly.
- `Y_C` and `M_C` are present directly.
- Additional labels such as `Y_WT`, `O_CR`, `Y_WS`, `O_Met`, and `O_VC` appear in methylation metadata and are not part of the current core SRSC intervention crosswalk.

## Practical use in this repo

Use this crosswalk as the working label legend for repository interpretation, with the following caution:

1. `A1-Ctrl <-> Y_C`, `A2-Ctrl <-> M_C`, `A3-Ctrl <-> O_C`, and `A4-WTC <-> O_WT` are strong enough to use operationally.
2. `A4-SRC <-> O_GES` and `A4-Ctrl <-> O_V` are now supported both by the public `PRJCA030875` BioProject sample names and by `OMIX007586-02.zip/sample.info.csv`, but they should still be described as inferred cross-source mappings rather than verbatim article labels.
3. Do not collapse `O_C` and `O_V` into a single "old control" bucket without stating the distinction. The paper uses age-staged controls plus an A4 intervention-era control arm.
