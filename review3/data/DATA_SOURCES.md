# Dataset mirrors (Phase 2.2)

Checked on 2026-10-20. This machine can reach github.com (git protocol), raw.githubusercontent.com and media.githubusercontent.com. The GitHub search API (`gh search`) and `api.github.com/repos/...` are blocked for unattached repositories, so candidates were found with WebSearch and checked with a blobless shallow `git clone --filter=blob:none --no-checkout --depth 1` plus `git ls-tree -l`. About 75 repositories were checked. Downloads were only fetched and parsed with pandas. Nothing in them was executed.

Summary:

| Dataset | Status | Local path |
|---|---|---|
| IoTID20 | NOT OBTAINED | none |
| Edge-IIoTset | OBTAINED (ML-EdgeIIoT-dataset.csv only; the DNN file was not found) | `/home/claude/r3/data/edge_iiotset/` |
| CICIDS2017 | PARTIAL ONLY (1 of 8 days, reduced 32-column variant); the full set was not obtained | `/home/claude/r3/data/cicids2017_partial/` |
| WUSTL-EHMS-2020 | OBTAINED | `/home/claude/r3/data/wustl_ehms_2020/` |
| CICIoMT2024 | NOT OBTAINED (not needed, because WUSTL was obtained) | none |

---

## 1. WUSTL-EHMS-2020: OBTAINED

- Source repo: https://github.com/tushargahiwal/AI-Based-Cyber-Security-IOMT
- Commit: `19df88524aaeaa516310035e5b4424c8b0503bbe` (2026-10-03)
- File URL: https://raw.githubusercontent.com/tushargahiwal/AI-Based-Cyber-Security-IOMT/19df88524aaeaa516310035e5b4424c8b0503bbe/data/raw/wustl-ehms-2020.csv
- Local file: `/home/claude/r3/data/wustl_ehms_2020/wustl-ehms-2020.csv`
- Size: 3,930,570 bytes
- sha256: `2d5e5fb2248d935d0e30fd2f93ae26c8d9391dc036bd0552a8911223e6b65f17`
- Rows: 16,318 data rows (header excluded). Columns: 45.
- Label columns: `Label` (0/1) and `Attack Category`
  - Label: 0 = 14,272; 1 = 2,046
  - Attack Category: normal 14,272; Spoofing 1,124; Data Alteration 922
- Exact duplicate rows: 0
- Official description (https://www.cse.wustl.edu/~jain/ehms/index.html, fetched): 16,318 samples, 14,272 normal (87.5%) and 2,046 attack (12.5%), 44 features (35 network flow metrics, 8 biometric features and 1 label), download size 3,946,888 bytes.
- Match: YES for row count and class counts. The 45 columns are the 44 described plus the `Attack Category` column, which the page does not list separately. The file is 16,318 bytes smaller than the official download size (3,930,570 vs 3,946,888). 16,318 equals the row count, so this is consistent with CRLF line endings having been converted to LF. That is an inference I have not verified.
- Doubts: this is a third-party copy inside an application repo, not the authors' release. The labels and counts match exactly. Byte-level identity with the official file cannot be confirmed without the official checksum.

## 2. Edge-IIoTset: OBTAINED (ML subset only)

- Source repo: https://github.com/Rayan-Ali1083/Layer7Defend
- Commit: `5b7f0fa5abe085230d0ce2450fec2ab75c6fb891` (2023-12-01)
- File URL: https://raw.githubusercontent.com/Rayan-Ali1083/Layer7Defend/5b7f0fa5abe085230d0ce2450fec2ab75c6fb891/ML-EdgeIIoT-dataset.csv
- Local file: `/home/claude/r3/data/edge_iiotset/ML-EdgeIIoT-dataset.csv`
- Size: 82,184,390 bytes (a plain git blob, not LFS; the local git blob hash `859dc0f76f37714a6be851136f0a0905e73f2cd2` matches the tree entry at that commit)
- sha256: `53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a`
- Rows: 157,800 data rows (`wc -l` gives 157,801 including the header). Columns: 63 (61 features plus `Attack_label` and `Attack_type`).
- Label columns:
  - Attack_label: 1 = 133,499; 0 = 24,301
  - Attack_type: Normal 24,301; DDoS_UDP 14,498; DDoS_ICMP 14,090; Ransomware 10,925; DDoS_HTTP 10,561; SQL_injection 10,311; Uploading 10,269; DDoS_TCP 10,247; Backdoor 10,195; Vulnerability_scanner 10,076; Port_Scanning 10,071; XSS 10,052; Password 9,989; MITM 1,214; Fingerprinting 1,001 (15 classes)
- Duplicates measured here: 814 exact duplicate rows over all 63 columns. After dropping 15 high-cardinality or identifier columns (frame.time, IPs, payloads, ports and similar), 5,726 duplicate rows remain (152,074 distinct of 157,800). NOTE: the draft paper says this file has 157,800 rows but only 1,661 distinct feature vectors. That figure must come from the project's own feature selection, which I did not reproduce. Check that the paper's sentence names the feature set it used.
- Official description: the 2022 IEEE Access paper and the README notes of its Kaggle and DataPort release name this file as part of "Selected dataset for ML and DL". The authors' pages could not be fetched here (Kaggle and IEEE DataPort are blocked). The header matches the 63 columns the repo README lists, and the 15 classes match the Edge-IIoTset taxonomy (14 attacks plus Normal). The figure of about 157,800 rows given in the task brief matches exactly.
- Match: YES on rows, columns and class set, judged against the brief and the repo README. I could not check it against an official checksum.
- Doubts: this is a third-party copy, and the repo README points to the authors' Kaggle page as its source. DNN-EdgeIIoT-dataset.csv (2,219,201 rows, 63 columns; a third-party note at https://github.com/nickjeffrey/cisis23igpl/blob/main/Edge-IIoTset2023_dataset_preprocessing.md reports `wc -l` = 2,219,202 with 1,615,643 Normal rows) was NOT found on any reachable GitHub repo.

## 3. CICIDS2017: PARTIAL ONLY

Full MachineLearningCVE set: NOT OBTAINED.
- The closest candidate, https://github.com/roshanyadav7/Machine-Learning-CVE (commit `c891e2d87d1a0b4cab685ffb35cb94753dfffa4e`), has all 8 `*.pcap_ISCX.csv` names under `datasets/` as Git LFS pointers. The pointer sizes match the known MachineLearningCVE sizes (for example Wednesday 225,166,395 bytes; Monday 176,927,918 bytes). However, media.githubusercontent.com returns 404 for all 8 files, and the repo's `datasets/README.md` says the CSVs were not uploaded because of their size. The LFS objects do not exist on the server.
- Other candidates checked had code only, results CSVs only, or LFS pointers with no objects: ScriptedLines404/Network-Intrusion-Detection-System-using-Deep-Learning (`cicids2017_cleaned.csv`, LFS 404), anacletu/ml-intrusion-detection-cicids2017, rokibulroni/CIC-IDS-2017-Dataset (readme only), elifnurkarakoc/CICIDS2017 (no data files) and r-dube/CICIDS (only small processed or sampled files of 0.3 to 46 MB, not the originals).

Partial file obtained (use only as a sanity check, not for experiments):
- Source repo: https://github.com/StarterArcher/CICIDS2017
- Commit: `e33819772fe09911f5defb0a9c12db19301b6458` (2019-04-23)
- File URL: https://raw.githubusercontent.com/StarterArcher/CICIDS2017/e33819772fe09911f5defb0a9c12db19301b6458/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Local file: `/home/claude/r3/data/cicids2017_partial/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- Size: 36,010,816 bytes. sha256: `306294008927756094b069d24764bfa6519fe267a8104903c056fc9c3cf38636`
- Rows: 225,745. Columns: 32 (Flow ID and many statistics columns are absent; it includes Source IP, Timestamp and other GeneratedLabelledFlows-style identifiers).
- Label column: `' Label'` (leading space): DDoS 128,027; BENIGN 97,718. Exact duplicate rows: 2.
- Match: the row and class counts match the Friday DDoS day (225,745 rows; 128,027 DDoS), but the file is a column-reduced variant, not the 79-column MachineLearningCVE file (MachineLearningCVE Friday DDoS is about 77 MB). The repo also has two unexplained binary blobs (`3E3E9300`, `AC7B8300`, about 36.7 MB each) that were not downloaded.
- Doubts: unknown provenance and a reduced column set. Not suitable as the CICIDS2017 benchmark.

## 4. IoTID20: NOT OBTAINED

No reachable GitHub repository holding `IoT Network Intrusion Dataset.csv` (about 625,783 rows, 86 columns) was found. Repos checked that turned out to hold code only or other datasets: istec-iuc/AI-IDS-IoT, nadiab14/IDS_IOTDataset, rifatnoor-92/IoT-Network-Intrusion-Detection, cfoh/IoT-Network-Intrusion-Detection-with-Ensemble-Learners (UNSW Bot-IoT zips), ValerianRey/IDS_dataset_exploration, ocblvck/novel-quantum-ensembles-iot-nid (README says the dataset is not stored in the repo), phungthai2002/MachineLearning_IotDataset, comsyssec/datasets (mirai, NSL-KDD and UNSW-NB15 only), mnazhoq/IoT-Related-Dataset-and-Resources (links only), and about 20 more.
- The only IoTID20-derived file found was https://github.com/YafengWu/AEWAE `Data/IoT_2020_b_n_0.01.csv` (3,930,145 bytes). Its `Data/README.md` describes it as "sampled and normalized IoTID20". It is a 1% normalised sample and is unusable for the audit, so it was not downloaded.
- The official sources (IEEE DataPort, the Google Sites page and Kaggle mirrors) are blocked here.

## 5. CICIoMT2024: NOT OBTAINED

https://github.com/duaabn/CICIoMT2024 holds only evaluation outputs and a 65 MB `data/evidence/flow_metadata.csv` (not the dataset's train or test CSVs). https://github.com/L9ng807/WiFI_and_MQTT_attack_Dataset is empty (no HEAD). The official source (unb.ca) is blocked. Not pursued further because WUSTL-EHMS-2020 was obtained.

## Network notes

- `media.githubusercontent.com` IS reachable and serves LFS objects when they exist (tested: a 1,966-byte LFS file from ScriptedLines404/... returned 200).
- The git LFS batch API through github.com is refused by the session proxy for unattached repos. `add_repo` read access does not cover LFS.
- `github.com/<owner>/<repo>/raw/...` returns 403 through the proxy. Use raw.githubusercontent.com instead.
