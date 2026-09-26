# Study Design

This file summarizes the study design from the primary publication, public
database records, and released metadata. It keeps explicit status labels and
does not fill gaps by guesswork.

## Sources consulted

- Primary study publication:
  - `https://doi.org/10.1016/j.cell.2025.05.021`
- Working label legend:
  - `docs/group_label_crosswalk.md`
- Public database records:
  - `https://ngdc.cncb.ac.cn/bioproject/browse/PRJCA030875`
  - `https://www.sciencedirect.com/science/article/abs/pii/S0092867425005719`
  - `https://doi.org/10.1016/j.cell.2025.05.021`
- Repository context:
  - `README.md`
  - `src/run_pipeline.py`
  - `src/config.py`
- Repository data files:
  - `data/RAW/data/OMIX007580-02.csv`
  - `data/RAW/data/OMIX007581-01.csv`
  - `data/RAW/data/OMIX007582-02.csv`
  - `data/RAW/data/OMIX007582_beta_matrix.csv`
  - `data/RAW/data/OMIX007583-01.zip`
  - `data/RAW/data/OMIX007586-02.zip`

## Source reliability note

- `reported`: Directly stated in the primary publication or released metadata.
- `conflicting`: The article uses `WTC` and `SRC` terminology, while OMIX/repo metadata often use `WT` and `GES`. Public BioProject naming improves this mapping substantially, but the labels are still not text-identical across sources.
- `unknown`: If a detail is not in the extracted paper text or metadata files, it remains unknown here.

## Article identity

| Item | Status | Evidence |
|---|---|---|
| Title | `reported` | "Senescence-resistant human mesenchymal progenitor cells counter aging in primates" |
| Journal | `reported` | `Cell` |
| Publication year | `reported` | `2025` |
| DOI | `reported` | `10.1016/j.cell.2025.05.021` |
| Main species in the primate study | `reported` | `cynomolgus monkeys` |

## Study objective

| Item | Status | Evidence |
|---|---|---|
| Core objective | `reported` | The paper states that the authors hypothesized FOXO3-activated human SRCs could provide enhanced geroprotective effects in primates when delivered intravenously. |
| High-level claim tested | `reported` | The paper reports whether genetically engineered senescence-resistant human mesenchymal progenitor cells can decelerate aging-related decline across multiple organs in primates. |
| Exosome mechanism relevance | `reported` | The paper explicitly frames exosomes as couriers of SRC geroprotective effects and includes dedicated exosome experiments. |

## Experimental model

| Item | Status | Evidence |
|---|---|---|
| Primate model | `reported` | Aged cynomolgus monkeys were used for the main in vivo intervention study. |
| Rodent model | `reported` | C57BL/6J mice and BALB/c nude mice were used for exosome and cell biodistribution / tumorigenicity assays. |
| Human cell models | `reported` | WTCs are wild-type human mesenchymal progenitor cells derived from wild-type hESCs; SRCs are senescence-resistant human mesenchymal progenitor cells derived from FOXO3-engineered hESCs. |
| Origin of monkeys | `reported` | The paper states that all monkeys originated from Southeast Asia. |

## Groups and intervention arms

### Article-level intervention groups

| Item | Status | Evidence |
|---|---|---|
| Aged monkey treatment groups | `reported` | The paper names `A4-Ctrl`, `A4-WTC`, and `A4-SRC` as the core aged intervention cohorts. |
| Young control cohort | `reported` | The paper names `A1-Ctrl` in figure legends and methods for some comparisons. |
| Additional age-control cohorts | `reported` | The methods mention `A2-Ctrl`, `A3-Ctrl`, and later age cohorts in modeling sections. |
| Control treatment | `reported` | `A4-Ctrl` received saline. |
| WTC treatment | `reported` | `A4-WTC` received wild-type human mesenchymal progenitor cells. |
| SRC treatment | `reported` | `A4-SRC` received FOXO3-activated senescence-resistant human mesenchymal progenitor cells. |

### Intervention details

| Item | Status | Evidence |
|---|---|---|
| Route | `reported` | Intravenous injection |
| Frequency | `reported` | Biweekly |
| Dose | `reported` | `2 x 10^6 cells/kg` body weight suspended in saline |
| Study duration | `reported` | `44-week study` in aged cynomolgus monkeys |

### Repo / OMIX group labels

| Repo / OMIX label | Status | Evidence |
|---|---|---|
| `Y_C`, `M_C`, `O_C` | `reported` | Present in `OMIX007580-02.csv` as control cohorts stratified by age. |
| `O_WT` | `high-confidence inferred` | OMIX uses `WT`; BioProject `PRJCA030875` lists biosamples such as `WT-MSC-F-1`; `OMIX007586-02.zip/sample.info.csv` maps `WT-MSC-*` to `A4_WTC`; the article uses `WTC` for the wild-type treatment arm. |
| `O_GES` | `high-confidence inferred` | OMIX uses `GES`; BioProject `PRJCA030875` lists biosamples such as `GESMSC-F-1`; `OMIX007586-02.zip/sample.info.csv` maps `GESMSC-*` to `A4_SRC`; the article uses `SRC` for the genetically enhanced senescence-resistant arm. |
| `O_V` | `high-confidence inferred` | OMIX uses `V`; BioProject `PRJCA030875` lists biosamples such as `O-V-F-1`; `OMIX007586-02.zip/sample.info.csv` maps `O-V-*` to `A4_Ctrl`; the paper states `A4-Ctrl` received saline. |

### Working article-to-repo crosswalk

Use `docs/group_label_crosswalk.md` as the operational label legend.

Current best-supported mapping:

| Article label | Repo / OMIX label | Status |
|---|---|---|
| `A1-Ctrl` | `Y_C` | `high-confidence inferred` |
| `A2-Ctrl` | `M_C` | `high-confidence inferred` |
| `A3-Ctrl` | `O_C` | `high-confidence inferred` |
| `A4-WTC` | `O_WT` | `high-confidence inferred` |
| `A4-SRC` | `O_GES` | `high-confidence inferred` |
| `A4-Ctrl` | `O_V` | `high-confidence inferred` |

## Outcomes and endpoints

| Endpoint class | Status | Evidence |
|---|---|---|
| Multi-organ transcriptomic aging / biological age | `reported` | The paper reports deceleration of multi-organ aging clocks and transcriptAge changes across tissues. |
| Blood and plasma rejuvenation signals | `reported` | Figure 2 and related methods analyze PBMCs and plasma in monkeys after WTC or SRC treatment. |
| Histological aging biomarkers across organs | `reported` | Figure 4 and related text report multi-organ histological aging-marker changes. |
| Brain function / cognition | `reported` | WGTA delay-task testing and structural MRI were performed in monkeys. |
| Bone health | `reported` | The paper states improved bone density. |
| Reproductive system / ovarian aging | `reported` | Figure 6 and ovary-related analyses assess reproductive rejuvenation. |
| Hippocampal cell-state rejuvenation | `reported` | Figure 5 and snRNA-seq analyses assess hippocampal cell populations. |
| Exosome-mediated geroprotection | `reported` | Figure 7 and supplementary exosome analyses test SRC-derived exosome effects. |
| Repository aging clock / rejuvenation score | `reported` | These are repo-derived computational summaries inspired by the study, not direct article terminology. |

## Sample size

The paper does not present one single `n` for every assay. Sample size varies by endpoint and figure.

### Reported article-level examples

| Assay / comparison | Status | Evidence |
|---|---|---|
| WGTA delay task | `reported` | Figure 1 legend reports `7` monkeys in `A4-Ctrl` and `A4-SRC`, and `8` monkeys in `A4-WTC` for panel B. |
| Several physiological and imaging analyses | `reported` | Figure 1 legend reports `16` monkeys in `A1-Ctrl`, `7` in `A4-Ctrl`, and `8` in `A4-SRC` for some panels; `6` in `A1-Ctrl` and `8` in aged groups for others. |
| PBMC / plasma and multi-omics analyses | `reported` | Figure 2 legend reports `n = 6-8 monkeys per group`. |
| Histology panels across organs | `reported` | Figure 4 legend reports `n = 6-8 monkeys per group`. |
| Reproductive tissue assays | `reported` | Figure 6 legend reports `n = 3-4 monkeys per group` for some panels. |
| Exosome mouse study | `reported` | Figure 7 legend reports `n = 10 mice per group` for some endpoints and `n = 4-5 biological samples per group` for others. |

### Observable repo data sizes

| Dataset | Observable size | Status | Source |
|---|---:|---|---|
| Bulk metadata rows | 2059 | `reported` | `data/RAW/data/OMIX007580-02.csv` |
| Bulk unique `orig.ident` values | 61 | `reported` | `data/RAW/data/OMIX007580-02.csv` |
| Bulk tissues | 39 | `reported` | `data/RAW/data/OMIX007580-02.csv` |
| Plasma samples | 32 | `reported` | `data/RAW/data/OMIX007581-01.csv` |
| Plasma protein rows | 3608 | `reported` | `data/RAW/data/OMIX007581-01.csv` |
| Methylation metadata rows | 620 | `reported` | `data/RAW/data/OMIX007582-02.csv` |
| Methylation sample columns | 643 | `reported` | `data/RAW/data/OMIX007582_beta_matrix.csv` |

## Randomization and blinding

| Item | Status | Evidence |
|---|---|---|
| Randomization | `reported` | The methods state that cynomolgus monkeys were randomly assigned into groups. |
| Blinding | `unknown` | No explicit blinding statement was found in the extracted paper text. |

## Statistics

| Item | Status | Evidence |
|---|---|---|
| Global statistical threshold | `reported` | `p < 0.05` was regarded as statistically significant. |
| Software for cognition / MRI | `reported` | `SPSS` |
| Software for sequencing / methylation processing | `reported` | `R` packages |
| Software for other experimental data | `reported` | `GraphPad Prism` |
| Statistical tests reported | `reported` | One-way ANOVA, two-way ANOVA, two-sided Student's t test, two-sided Wilcoxon rank-sum test, GLMM analysis |
| Figure-specific corrections | `reported` | Bonferroni, Dunnett, Tukey, and Kruskal-Wallis variants are used in figure-specific analyses |

## How the repository maps to the paper

| Repo component | Status | Evidence |
|---|---|---|
| `OMIX007580` bulk transcriptomics | `reported` | Matches the repo bulk RNA-seq block used for tissue-level aging and rejuvenation analyses. |
| `OMIX007581` plasma proteomics | `reported` | Used for exploratory within-plasma `GES - V`, `WT - V`, and `GES - WT` protein contrasts and a reference-trained age-state axis; cross-modality linkage remains a separate unresolved question. |
| `OMIX007582` DNA methylation | `reported` | Matches the repo optional methylation block and aligns with paper methylation-age analyses. |
| `OMIX007583-01.zip` ovary expression subset | `reported` | Aligns with paper ovarian / reproductive analyses and confirms subset-internal use of `V`, `WT`, and `GES` labels for female old intervention-era samples. |
| `OMIX007586-02.zip` hippocampus expression subset | `reported` | Aligns with paper hippocampal analyses and directly maps `WT-MSC-*`, `GESMSC-*`, and `O-V-*` samples to `A4_WTC`, `A4_SRC`, and `A4_Ctrl` in `sample.info.csv`. |
| Repo `rejuvenation_score`, causal gate, response-alignment profile, and exosome-fraction status | `reported` | These are repository-level analytic abstractions, not article-native endpoint names. The profile quantifies non-causal cross-species resemblance; the fraction remains causally unidentified. |

## Known gaps and ambiguities

September 2026 update: [PRJCA035748](https://ngdc.cncb.ac.cn/bioproject/browse/PRJCA035748)
provides additional specimen provenance for the article-linked GSA deposits.
The [refresh audit](data_refresh.md) records two additional cargo resources,
OMIX009654 and OMIX009655, and their conflict with the plasma wording in the
deposit titles. These are separate from recipient plasma OMIX007581. Bulk
BioSample aliases do not establish cross-modal individual identity; the 24
plasma naming matches are now candidates rather than confirmed links.

1. `SRC` in the paper and `GES` in OMIX/repo metadata are strongly supported across paper, OMIX, BioProject, and `OMIX007586` subset metadata, but the label translation is still cross-source and not written as a single canonical legend in the article PDF.
2. `WTC` in the paper and `WT` in OMIX/repo metadata are strongly supported across paper, OMIX, BioProject, and `OMIX007586` subset metadata, but the names are still not text-identical.
3. `O_V` is now strongly supported as the old vehicle/saline control arm by paper, BioProject, and `OMIX007586` subset metadata, but this conclusion is still assembled across sources rather than stated as a single glossary item in the paper text.
4. `data/RAW/data/OMIX007582-02.csv` and `data/RAW/data/OMIX007582_beta_matrix.csv` still have a `620` vs `643` sample-accounting mismatch. See `docs/OMIX007582_sample_map_audit.md`.
5. `data/RAW/data/OMIX007580-02.csv` contains one anomalous `orig.ident` value, `58-MF-C-Trachea`, that should be audited before treating the file as perfectly clean metadata.
6. Some page-1 / page-2 text in the extracted article still contains publisher-layout artifacts; section-level extraction is therefore preferable to naive whole-file reading.

## Practical implications for this repo

1. The repo objective is scientifically aligned with the paper's mechanism question, especially around whether broad geroprotection may be mediated in part by SRC-derived exosomes.
2. The repo should continue to separate reproduced study-aligned analyses,
   exploratory cross-species response alignment, and causal decomposition. The
   first two can be quantified under their documented assumptions; the third
   is not identified by the available design.
3. Use `docs/group_label_crosswalk.md` when translating article cohort names into repo group names.
4. Any claim equating repo `O_GES` directly with article `A4-SRC`, or repo `O_V` with article `A4-Ctrl`, should still be phrased as a cross-source mapping rather than a verbatim article label substitution.

## Inference Structure

The study combines modalities whose samples are not uniformly linked at the
animal level. Consequently, the availability of transcriptomic, plasma,
methylation, and exosome-related data does not imply that all cross-modal
quantities are estimable.

Analyses involving cross-modality linkage, mediation, exosome attribution, or
cross-species mechanistic support are governed by:

[`docs/inference_framework.md`](inference_framework.md)

In particular:

- linkage must be demonstrated rather than inferred;
- estimability is evaluated before model execution or interpretation;
- mediation requires compatible animal-level observations;
- exosome association is distinct from exosome causality;
- mouse evidence is mechanism-supportive rather than direct macaque or human
  validation.
