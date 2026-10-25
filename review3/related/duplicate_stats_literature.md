# Published duplicate-record statistics

Compiled 2026-10-20. "Exact" means rows identical across all columns unless stated otherwise. Percentages I computed myself are marked "(computed)".

## IoTID20

| Figure | Source | Exact location |
|---|---|---|
| 164,087 duplicated records removed from 625,783 rows (86 columns), leaving 461,696. That is 26.2% (computed). | Alsulami, Abu Al-Haija, Tayeb, Alqahtani, "An Intrusion Detection and Classification System for IoT Traffic with Improved Data Engineering," *Applied Sciences* 12(23):12336, 2022. DOI 10.3390/app122312336 | Sec. 4.1 (dataset size) and Sec. 4.2 ("there were 164,087 duplications of records"). Read at https://www.mdpi.com/2076-3417/12/23/12336 |

Not found: any 2024 to 2026 paper that reports IoTID20 duplicate rates. Note that the draft paper's figure (58.2% exact duplicates in *cleaned* IoTID20) refers to a different population, after cleaning and column dropping, and probably a different feature set. The two figures are not contradictory, but the paper should state which columns were compared.

## Edge-IIoTset

| Figure | Source | Exact location |
|---|---|---|
| ML-EdgeIIoT-dataset.csv: 814 exact duplicate rows (0.52%), all with consistent labels | Galal, arXiv:2608.15761 (2026, preprint) | HTML full text, https://arxiv.org/html/2608.15761 |
| DNN-EdgeIIoT-dataset.csv: the corrected protocol removed 309,530 duplicates (2,219,201 rows in total; the cleaned subset is 1,909,671 after other cleaning steps as well) | same | same |
| Independent check: I measured exactly 814 duplicate rows over all 63 columns of the ML file (157,800 rows) downloaded for this project | this project, `/home/claude/r3/data/DATA_SOURCES.md` | |

IMPORTANT for the paper: the draft says Edge-IIoTset has "157,800 rows containing only 1,661 distinct feature vectors" (98.95% duplicates). Over all 63 columns there are 156,986 distinct rows. After dropping 15 identifier and payload columns there are still 152,074. The 1,661 figure can only come from a much smaller selected feature set. The paper must name that feature set, or a reviewer who checks it against Galal's 0.52% will treat it as an error.

## CICIDS2017

| Figure | Source | Exact location |
|---|---|---|
| No duplicate-row figure, but 25.9% of all flows are TCP "appendix" artefacts (up to 50% in some attack classes); more than 20% of traces were reconstructed or relabelled | Engelen, Rimmer, Joosen, IEEE SPW 2021, DOI 10.1109/SPW53761.2021.00009 | Full text, https://lirias.kuleuven.be/retrieve/713070 |
| Duplicated packets reported as one of the capture errors (no count in the abstract) | Lanvin et al., CRiSIS 2022 / LNCS 13857, DOI 10.1007/978-3-031-31108-6_2 | Abstract, https://researchportal.ip-paris.fr/en/publications/errors-inthecicids2017-dataset-andthesignificant-differences-inde/ |
| NOT PEER REVIEWED: 256,479 full-row duplicates (9.06%) and 309,079 feature-level duplicates (10.92%) in the merged MachineLearningCVE files; 698 duplicate groups with conflicting labels (7,020 rows) | GitHub audit table, Essa-Almutawa/network-intrusion-detection-cicids2017, commit 0b45ba7ff5d869a37dba93a4f3493aedb95101b6, `results/eda/tables/table04_duplicates_summary.csv` | Fetched with git |
| NOT PEER REVIEWED: per-file redundant copies that sum to 256,479 (for example Wednesday 81,909; PortScan 72,353) | GitHub, tejas-sangale04/Intrusion-Detection-System-using-Random-Forest, commit fe2430abe39152b92299c7a1e061d18aeef3fd10, `reports/metrics/inspection/duplicates_by_file.csv` | Fetched with git |

The two independent repository audits agree exactly (256,479), which supports the figure, but neither is a publication. Cite them as supplementary material only, or reproduce the count once the full CICIDS2017 files are available.

## WUSTL-EHMS-2020

- No published duplicate statistic found.
- Measured here: 0 exact duplicate rows in the 16,318-row, 45-column file (see DATA_SOURCES.md).

## CICIoMT2024

- No published duplicate statistic found. The dataset page (unb.ca) and the dataset paper's abstract give no row or duplicate counts.

## Related (not a requested dataset)

- CICIoT2023: 25,641,309 of 46,775,660 valid rows are exact duplicates (54.82%) over 39 features. Tolay, arXiv:2608.10349 (2026, preprint), https://arxiv.org/html/2608.10349. This could be cited as further evidence that heavy duplication is common in CIC IoT benchmarks.
