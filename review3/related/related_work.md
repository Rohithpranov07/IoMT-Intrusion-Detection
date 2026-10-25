# Related work (Phase 8.4)

Compiled 2026-10-20. Each entry below was checked against at least one of these: Crossref, OpenAlex, arXiv, the publisher page or an institutional record. The "Verified at" URL is the record actually fetched. Summaries are my own wording, based on the abstract text returned by that record. Empty fields mean the value was not found; none were guessed.

22 papers: Group 1 (9), Group 2 (7), Group 3 (4), Group 4 (2) plus one older anchor (1). BibTeX is in `references.bib`.

Access notes: ScienceDirect, ResearchGate, Semantic Scholar, dblp, PMC and IEEE Xplore pages were blocked or empty from this machine. Metadata therefore comes mostly from Crossref and OpenAlex. The PLOS ONE printable PDF and some OpenAlex records were refused with HTTP 429 (rate limit) and could not be read.

---

## Group 1. IoMT and healthcare-IoT intrusion detection with ML or DL (2024 to 2026)

1. **Berguiga, A., Harchay, A., Massaoudi, A.** "HIDS-IoMT: A Deep Learning-Based Intelligent Intrusion Detection System for the Internet of Medical Things." *IEEE Access*, vol. 13, pp. 32863-32882, 2025. DOI: 10.1109/ACCESS.2025.3543127. Verified at: https://api.openalex.org/works/doi:10.1109/ACCESS.2025.3543127
   Summary: A CNN feature extractor feeds an LSTM in a fog-deployed detector on a Raspberry Pi, evaluated on IoTID20 and Edge-IIoTset, with reported accuracy of 99.92% and recall of 99.99%. (This is the system this project audits.)

2. **Dadkhah, S., Pinto Neto, E. C., Ferreira, R., Molokwu, R. C., Sadeghi, S., Ghorbani, A. A.** "CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT." *Internet of Things*, vol. 28, art. 101351, 2024. DOI: 10.1016/j.iot.2024.101351. Verified at: https://api.openalex.org/works/doi:10.1016/j.iot.2024.101351 (also cited on https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)
   Summary: Releases an IoMT benchmark built by running 18 attacks against 40 real and simulated medical devices over Wi-Fi, MQTT and Bluetooth, with captures that span each device's full network lifecycle.

3. **Hosain, Y., Çakmak, M.** "XAI-XGBoost: an innovative explainable intrusion detection approach for securing internet of medical things systems." *Scientific Reports*, vol. 15, no. 1, art. 22278, 2025. DOI: 10.1038/s41598-025-07790-0. Verified at: https://api.openalex.org/works/doi:10.1038/s41598-025-07790-0
   Summary: Combines hybrid resampling, recursive feature elimination and a tuned XGBoost model with SHAP and LIME explanations on WUSTL-EHMS-2020, reporting about 99.2% accuracy and a ROC-AUC of 1.0.

4. **Shaikh, J. A., Wang, C., Us Sima, M. W., Arshad, M., Owais, M., Alnashwan, R. O., Chelloug, S. A., Muthanna, M. S. A.** "RCLNet: an effective anomaly-based intrusion detection for securing the IoMT system." *Frontiers in Digital Health*, vol. 6, art. 1467241, 2024. DOI: 10.3389/fdgth.2024.1467241. Verified at: https://api.openalex.org/works/doi:10.3389/fdgth.2024.1467241
   Summary: Uses random-forest feature selection, a CNN-LSTM with a self-adaptive attention layer and focal loss for class imbalance, and reports 99.78% accuracy on WUSTL-EHMS-2020.

5. **Ghourabi, A., Alkhalil, A.** "A Federated Learning Model for Detecting Cyberattacks in Internet of Medical Things Networks." *IEEE Access*, vol. 13, pp. 123018-123030, 2025. DOI: 10.1109/ACCESS.2025.3588808. Verified at: https://api.openalex.org/works/doi:10.1109/access.2025.3588808
   Summary: Trains Bayesian-tuned XGBoost models locally on IoMT clients and aggregates them federatively, evaluated on CICIoMT2024, ECU-IoHT and WUSTL-EHMS.

6. **Alalhareth, M., Hong, S.-C.** "Enhancing the Internet of Medical Things (IoMT) Security with Meta-Learning: A Performance-Driven Approach for Ensemble Intrusion Detection Systems." *Sensors*, vol. 24, no. 11, art. 3519, 2024. DOI: 10.3390/s24113519. Verified at: https://api.openalex.org/works/doi:10.3390/s24113519
   Summary: Proposes an ensemble IDS whose voting weights are set from each base model's accuracy, loss and confidence, and reports gains in detection rate, F1 and false-positive rate over the comparison models.

7. **Lazrek, G., Chetioui, K., Balboul, Y., Mazer, S., El Bekkali, M.** "An RFE/Ridge-ML/DL based anomaly intrusion detection approach for securing IoMT system." *Results in Engineering*, vol. 23, art. 102659, 2024. DOI: 10.1016/j.rineng.2024.102659. Verified at: https://api.openalex.org/works/doi:10.1016/j.rineng.2024.102659
   Summary: Pairs recursive feature elimination and ridge-based selection with ML and DL classifiers on WUSTL-EHMS; the best model, a decision tree, reaches 97.85% test accuracy.

8. **Söğüt, E., Kayaoğlu, M., Polat, O.** "MSCA-Net: A Multi-Scale Depthwise Attention Network for Multi-Class Intrusion Detection in Internet of Medical Things." *Sensors*, vol. 26, no. 13, art. 4036, 2026. DOI: 10.3390/s26134036. Verified at: https://api.openalex.org/works/doi:10.3390/s26134036
   Summary: A lightweight model that combines multi-scale depthwise convolutions, channel attention and a small LSTM, evaluated on WUSTL-EHMS-2020 and CICIoMT2024 with 99.75% six-class accuracy and about 2x faster inference than nine baselines.

9. **Mohammadi, A., Aminian, M., Ghahramani, H., Asghari, S. A.** "Securing Healthcare with Deep Learning: A CNN-Based Model for medical IoT Threat Detection." In *Proc. 2024 19th Iranian Conference on Intelligent Systems (ICIS)*, 2024. DOI: 10.1109/ICIS64839.2024.10887510. arXiv: 2410.23306. Verified at: https://arxiv.org/abs/2410.23306 (DOI listed on https://arxiv.org/abs/2410.23306v2)
   Summary: A CNN classifier for CICIoMT2024 that reports about 99% accuracy in binary, category and multiclass settings, outperforming logistic regression, AdaBoost, a DNN and random forest. Pages were not found.

## Group 2. Evaluation methodology problems in ML-based intrusion detection

10. **Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger, C., Cavallaro, L., Rieck, K.** "Dos and Don'ts of Machine Learning in Computer Security." In *Proc. 31st USENIX Security Symposium (USENIX Security 22)*, pp. 3971-3988, 2022. ISBN 978-1-939133-31-1. DOI: none found. arXiv: 2010.09470. Verified at: https://www.usenix.org/conference/usenixsecurity22/presentation/arp and https://arxiv.org/abs/2010.09470
    Summary: Catalogues recurring pitfalls in the design and evaluation of learning-based security systems, shows they are common across 30 top-venue papers, and demonstrates experimentally how they inflate reported results.

11. **Engelen, G., Rimmer, V., Joosen, W.** "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study." In *2021 IEEE Security and Privacy Workshops (SPW)*, pp. 7-12, 2021. DOI: 10.1109/SPW53761.2021.00009. Verified at: https://api.crossref.org/works/10.1109/SPW53761.2021.00009 and full text at https://lirias.kuleuven.be/retrieve/713070
    Summary: Re-examines the CICIDS2017 generation pipeline, finds faults in traffic generation, flow construction, feature extraction and labelling, and reconstructs or relabels more than 20% of the traces, which changes ML benchmark results.

12. **Lanvin, M., Gimenez, P.-F., Han, Y., Majorczyk, F., Mé, L., Totel, É.** "Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes." In *Risks and Security of Internet and Systems (CRiSIS 2022)*, LNCS vol. 13857, pp. 18-33, Springer, 2023. DOI: 10.1007/978-3-031-31108-6_2. Verified at: https://researchportal.ip-paris.fr/en/publications/errors-inthecicids2017-dataset-andthesignificant-differences-inde/
    Summary: Documents further CICIDS2017 flaws, including out-of-order packets, duplicated packets and attacks that were never labelled, and measures how correcting them changes supervised IDS evaluation.

13. **Liu, L., Engelen, G., Lynar, T., Essam, D., Joosen, W.** "Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018." In *2022 IEEE Conference on Communications and Network Security (CNS)*, 2022. DOI: 10.1109/CNS56114.2022.9947235. Verified at: https://archive.cps-vo.org/node/97449
    Summary: Identifies previously undocumented attack-orchestration, feature-generation, documentation and labelling errors in two CIC benchmarks and releases a relabelled version with its labelling logic. Pages were not found.

14. **Flood, R., Engelen, G., Aspinall, D., Desmet, L.** "Bad Design Smells in Benchmark NIDS Datasets." In *2024 IEEE 9th European Symposium on Security and Privacy (EuroS&P)*, pp. 658-675, 2024. DOI: 10.1109/EuroSP60621.2024.00042. Verified at: https://api.crossref.org/works/10.1109/EuroSP60621.2024.00042 and https://www.research.ed.ac.uk/en/publications/bad-design-smells-in-benchmark-nids-datasets/
    Summary: Manually audits seven heavily cited synthetic NIDS benchmarks, defines six "data design smells" with heuristics to measure them, and shows through case studies that these smells bias experimental conclusions.

15. **Bouke, M. A., Abdullah, A.** "An empirical study of pattern leakage impact during data preprocessing on machine learning-based intrusion detection models reliability." *Expert Systems with Applications*, vol. 230, art. 120715, 2023. DOI: 10.1016/j.eswa.2023.120715. Verified at: https://api.crossref.org/works/10.1016/j.eswa.2023.120715 and https://api.openalex.org/works/doi:10.1016/j.eswa.2023.120715
    Summary: LIMITED VERIFICATION. The paper's existence and metadata are confirmed, but **the abstract could not be read**: Crossref and OpenAlex have no abstract field, and ScienceDirect, ResearchGate, Semantic Scholar and the UPM repository were all blocked or failed. From the title and OpenAlex keywords (data leakage, data preprocessing, overfitting, NSL-KDD, UNSW-NB15, KDD Cup 1999), it is an empirical study of how leakage introduced at the preprocessing stage affects the reliability of ML IDS models on those benchmark datasets. Check this summary against the abstract before submission.

16. **Galal, M. M.** "Provenance, Not Behaviour: A Serialisation Artifact in Edge-IIoTset and a Leakage-Free Benchmark for Precision-Agriculture Intrusion Detection." arXiv:2608.15761, 2026 (preprint). Verified at: https://arxiv.org/abs/2608.15761v1 and https://arxiv.org/html/2608.15761
    Summary: Shows that Edge-IIoTset's recommended one-hot encoding lets four columns reveal the label perfectly, because a missing-field placeholder is serialised differently ("0" against "0.0") in normal and attack files; correcting this cuts reported scores, and the paper also reports duplicate counts (see `duplicate_stats_literature.md`). This is directly relevant to the Edge-IIoTset half of this project and is not yet peer reviewed.

## Group 3. Network simulation (ns-3, OMNeT++) used to evaluate intrusion detection or IoT security (2023 to 2026)

17. **Meyer, P., Häckel, T., Lübeck, T., Korf, F., Schmidt, T. C.** "A Framework for the Systematic Assessment of Anomaly Detectors in Time-Sensitive Automotive Networks." In *Proc. 2024 IEEE Vehicular Networking Conference (VNC)*, pp. 57-64, 2024. DOI: 10.1109/VNC61989.2024.10576017. arXiv: 2405.01324. Verified at: https://api.openalex.org/works/doi:10.1109/VNC61989.2024.10576017 and https://arxiv.org/html/2405.01324v1 (simulator stated in Sec. III-B as OMNeT++ with INET)
    Summary: Builds an OMNeT++ toolchain that generates topologies, traffic, anomalies and attacks so anomaly detectors can be assessed repeatably inside a simulated in-vehicle network, and shows that detection quality depends on the traffic mix and anomaly type.

18. **Rustam, F., Obaidat, I., Jurcut, A. D.** "MULTI-LF: A Continuous Learning Framework for Real-Time Malicious Traffic Detection in Multi-Environment Networks." arXiv:2504.11575, 2025 (v2 retitled; preprint). Verified at: https://arxiv.org/abs/2504.11575
    Summary: Builds a Docker and ns-3 testbed that emulates mixed IoT and conventional traffic to produce labelled live data, then evaluates a two-stage continual-learning detector on that traffic in real time.

19. **Chaudhari, S. S., Manjula, L., Muttur, C. S., Shadaksharaiah, A. M., Preethi, D., Sparshika, T. N.** "Intelligent DDoS Attack Detection for IoT Network Scenario using NS3." In *2023 International Conference on Integrated Intelligence and Communication Systems (ICIICS)*, pp. 1-6, 2023. DOI: 10.1109/ICIICS59993.2023.10421286. Verified at: https://api.openalex.org/works/doi:10.1109/ICIICS59993.2023.10421286 (abstract read on the ResearchGate record)
    Summary: Simulates an IoT botnet in ns-3 to build its own normal and DDoS traffic, then detects attacks with a random forest using jellyfish-swarm feature selection. Note that the simulator generates data here; the detector is not placed in the loop.

20. **Rajab, R. M., Ward, R. P., Wilson, I. D., Abuhmida, M.** "Assessing IoMT Network Resilience: A Comparative Performance Analysis of Simulated Attacks using NS-3 [Appendix]." Zenodo report linked to iCETiC '25, 2025. DOI: 10.5281/zenodo.16810649. Verified at: https://zenodo.org/records/16810649
    Summary: Simulates DoS, DDoS, man-in-the-middle, blackhole and MQTT flooding against an ns-3 model of a wireless IoMT network (infusion pump, smart shirt) and measures loss, latency and energy, finding MITM most disruptive and energy alone a poor attack signal. This is a Zenodo appendix record; the proceedings chapter itself was not found.

## Group 4. Adaptive or dynamic thresholds for anomaly or intrusion detection

21. **Hundman, K., Constantinou, V., Laporte, C., Colwell, I., Soderstrom, T.** "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding." In *Proc. 24th ACM SIGKDD Int. Conf. on Knowledge Discovery & Data Mining (KDD '18)*, pp. 387-395, 2018. DOI: 10.1145/3219819.3219845. arXiv: 1802.04431. Verified at: https://arxiv.org/abs/1802.04431 and https://api.openalex.org/works/doi:10.1145/3219819.3219845
    Summary: Flags anomalies in NASA telemetry from LSTM prediction errors using an unsupervised, nonparametric threshold that adapts to recent error statistics, plus pruning steps to cut false positives. (Older anchor reference for dynamic thresholding.)

22. **Yuan, Y., Huang, Y., Wu, Y.** "Adaptive NAD: Online and Self-adaptive Unsupervised Network Anomaly Detector." arXiv:2410.22967, 2024 (v6, 2026; preprint). Verified at: https://arxiv.org/abs/2410.22967
    Summary: An online unsupervised network anomaly detector that generates pseudo-labels with a two-layer scheme and retrains with a new threshold calculation as traffic evolves, evaluated on CIC-Darknet2020, NSL-KDD and Edge-IIoTset.

Reserve for Group 4 (verified, not counted in the 22): **Ding, F., Zhao, Y., Li, Z., Tang, H., Liu, Y., Guo, D.** "Adaptive Dynamic Thresholds for Unsupervised Joint Anomaly Detection and Trend Prediction." *Sensors*, vol. 26, no. 1, art. 257 (published online 31 Dec 2025). DOI: 10.3390/s26010257. Verified at: https://api.openalex.org/works/doi:10.3390/s26010257. Summary: derives a self-adjusting threshold from historical data and refines it with trend forecasts in a feedback loop.

---

## Verified but not included (available if needed)

- Bouke, M. A., El Atigh, H., Abdullah, A. "Towards robust and efficient intrusion detection in IoMT: a deep learning approach addressing data leakage and enhancing model generalizability." *Multimedia Tools and Applications*, vol. 84, no. 30, pp. 36523-36542, 2024. DOI: 10.1007/s11042-024-19916-z (OpenAlex). WUSTL-EHMS-2020 leakage-aware model, about 99% accuracy (abstract read on ResearchGate).
- Apicella, A., Isgrò, F., Prevete, R. "Don't push the button! Exploring data leakage risks in machine learning and transfer learning." *Artificial Intelligence Review*, vol. 58, no. 11, 2025. DOI: 10.1007/s10462-025-11326-3 (OpenAlex). A general leakage taxonomy, not specific to IDS.
- Rosay, A., Carlier, F., Cheval, E., Leroux, P. "From CIC-IDS2017 to LYCOS-IDS2017: A corrected dataset for better performance." *IEEE/WIC/ACM Int. Conf. on Web Intelligence*, pp. 570-575, 2021. DOI: 10.1145/3486622.3493973 (Crossref). Abstract not read.
- Naghib, A., Gharehchopogh, F. S., Zamanifar, A. "A comprehensive and systematic literature review on intrusion detection systems in the internet of medical things." *Artificial Intelligence Review*, vol. 58, no. 4, 2025. DOI: 10.1007/s10462-024-11101-w (OpenAlex). A review of 28 IoMT IDS studies.
- Meyer, P., Häckel, T., Reider, S., Korf, F., Schmidt, T. C. "Network Anomaly Detection in Cars: A Case for Time-Sensitive Stream Filtering and Policing." arXiv:2112.11109. Uses OMNeT++ for all benchmarks. Its year (2021) is outside the Group 3 window, and I did not check where it was formally published.

Rejected after checking (they did not use a network simulator, or no abstract was available): Tariq et al. 2024 (J. Cloud Computing, fog FL IDS, no simulator); Verma and Gautam 2025 (PPNA, ns-3 in the title, but no abstract available); Bokka and Sadasivam 2024 (J. Electronic Testing, simulator not named and no abstract); Hara et al. 2024 (ICNCC, ns-3 IoT malware, no abstract); Xue et al. 2024 (Neural Computing and Applications, RL dynamic thresholding, no abstract).

---

## Positioning against Bouke and Abdullah (2023)

Caveat: I could not read Bouke and Abdullah's abstract from this machine (see entry 15). The sentence below describing their work rests on the title, the OpenAlex keyword record and the draft paper's existing characterisation. Check it against the abstract before submission.

> Bouke and Abdullah [bouke2023empirical] show empirically, on benchmark intrusion detection datasets, that pattern leakage introduced during data preprocessing inflates the apparent reliability of machine-learning intrusion detection models. Our work starts from that finding and extends it in three directions. First, we identify exact duplicate records as a second leakage vector that survives correct split ordering: even when resampling and scaling are fitted on the training fold alone, identical rows still fall on both sides of a random partition. Second, we audit published confusion matrices arithmetically, which exposes partitions drawn from the oversampled population and transposed precision and recall without any access to code. Third, we place the detector's threshold rule inside an ns-3 network simulation, revealing failure modes such as false quarantine of legitimate medical devices and a latch-off failure that no dataset evaluation can show.

(Word count of the quoted paragraph: about 130.)
