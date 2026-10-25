# Review 3 multi-partition summary (Attack = positive)

## edge_table7  (n = 20 partitions)

- all cnn_accuracy: 0.8840 ± 0.0615 (95% CI ± 0.0288)
- all cnn_precision: 0.9465 ± 0.0325 (95% CI ± 0.0152)
- all cnn_recall: 0.9257 ± 0.0995 (95% CI ± 0.0465)
- all cnn_f1: 0.9315 ± 0.0422 (95% CI ± 0.0198)
- all bilstm_accuracy: 0.7769 ± 0.0421 (95% CI ± 0.0197)
- all bilstm_precision: 0.9920 ± 0.0053 (95% CI ± 0.0025)
- all bilstm_recall: 0.7549 ± 0.0511 (95% CI ± 0.0239)
- all bilstm_f1: 0.8563 ± 0.0312 (95% CI ± 0.0146)
- all transformer_accuracy: 0.7955 ± 0.0318 (95% CI ± 0.0149)
- all transformer_precision: 0.9921 ± 0.0069 (95% CI ± 0.0032)
- all transformer_recall: 0.7759 ± 0.0402 (95% CI ± 0.0188)
- all transformer_f1: 0.8702 ± 0.0231 (95% CI ± 0.0108)
- all ensemble_accuracy: 0.7946 ± 0.0187 (95% CI ± 0.0088)
- all ensemble_precision: 0.9928 ± 0.0042 (95% CI ± 0.0020)
- all ensemble_recall: 0.7741 ± 0.0223 (95% CI ± 0.0104)
- all ensemble_f1: 0.8697 ± 0.0137 (95% CI ± 0.0064)
- all oracle_accuracy: 0.9478 ± 0.0728 (95% CI ± 0.0341)
- all oracle_ceiling_over_best: 0.0581 ± 0.0265 (95% CI ± 0.0124)

## edge_table8  (n = 5 partitions)

- all shap_jaccard: 0.8986 ± 0.0408 (95% CI ± 0.0506)
- all lime_jaccard: 0.2209 ± 0.0162 (95% CI ± 0.0201)

## iotid20_table3  (n = 20 partitions)

- HONEST accuracy: 0.9970 ± 0.0003 (95% CI ± 0.0001)
- HONEST precision: 0.9976 ± 0.0003 (95% CI ± 0.0001)
- HONEST recall: 0.9990 ± 0.0002 (95% CI ± 0.0001)
- HONEST f1: 0.9983 ± 0.0001 (95% CI ± 0.0001)
- HONEST test_synthetic_pct: 0.0000 ± 0.0000 (95% CI ± 0.0000)
- LEAKY accuracy: 0.9993 ± 0.0001 (95% CI ± 0.0000)
- LEAKY precision: 0.9988 ± 0.0001 (95% CI ± 0.0000)
- LEAKY recall: 0.9997 ± 0.0001 (95% CI ± 0.0000)
- LEAKY f1: 0.9993 ± 0.0001 (95% CI ± 0.0000)
- LEAKY test_synthetic_pct: 46.5817 ± 0.0490 (95% CI ± 0.0229)
- ORDER accuracy: 0.9990 ± 0.0001 (95% CI ± 0.0000)
- ORDER precision: 0.9992 ± 0.0001 (95% CI ± 0.0000)
- ORDER recall: 0.9997 ± 0.0001 (95% CI ± 0.0000)
- ORDER f1: 0.9994 ± 0.0000 (95% CI ± 0.0000)
- ORDER test_synthetic_pct: 0.0000 ± 0.0000 (95% CI ± 0.0000)

## iotid20_table7  (n = 20 partitions)

- all cnn_accuracy: 0.8818 ± 0.0347 (95% CI ± 0.0162)
- all cnn_precision: 0.9091 ± 0.0229 (95% CI ± 0.0107)
- all cnn_recall: 0.9610 ± 0.0308 (95% CI ± 0.0144)
- all cnn_f1: 0.9340 ± 0.0200 (95% CI ± 0.0094)
- all bilstm_accuracy: 0.9244 ± 0.0309 (95% CI ± 0.0145)
- all bilstm_precision: 0.9422 ± 0.0290 (95% CI ± 0.0136)
- all bilstm_recall: 0.9739 ± 0.0115 (95% CI ± 0.0054)
- all bilstm_f1: 0.9576 ± 0.0167 (95% CI ± 0.0078)
- all transformer_accuracy: 0.8898 ± 0.0368 (95% CI ± 0.0172)
- all transformer_precision: 0.9100 ± 0.0313 (95% CI ± 0.0147)
- all transformer_recall: 0.9705 ± 0.0207 (95% CI ± 0.0097)
- all transformer_f1: 0.9390 ± 0.0200 (95% CI ± 0.0094)
- all ensemble_accuracy: 0.9074 ± 0.0222 (95% CI ± 0.0104)
- all ensemble_precision: 0.9193 ± 0.0195 (95% CI ± 0.0091)
- all ensemble_recall: 0.9803 ± 0.0094 (95% CI ± 0.0044)
- all ensemble_f1: 0.9487 ± 0.0120 (95% CI ± 0.0056)
- all oracle_accuracy: 0.9572 ± 0.0188 (95% CI ± 0.0088)
- all oracle_ceiling_over_best: 0.0253 ± 0.0096 (95% CI ± 0.0045)

## iotid20_table8  (n = 5 partitions)

- all shap_jaccard: 0.8455 ± 0.0689 (95% CI ± 0.0855)
- all lime_jaccard: 0.2545 ± 0.0825 (95% CI ± 0.1024)

## Paired Wilcoxon tests, Bonferroni over 7

- edge_table7 ensemble F1 vs best branch F1: p = 1.91e-06, Bonferroni p = 1.34e-05
- edge_table8 SHAP vs LIME Jaccard: p = 0.0625, Bonferroni p = 0.438
- iotid20_table3 F1 LEAKY vs ORDER: p = 1.91e-06, Bonferroni p = 1.34e-05
- iotid20_table3 F1 ORDER vs HONEST: p = 1.91e-06, Bonferroni p = 1.34e-05
- iotid20_table3 F1 LEAKY vs HONEST: p = 1.91e-06, Bonferroni p = 1.34e-05
- iotid20_table7 ensemble F1 vs best branch F1: p = 3.81e-06, Bonferroni p = 2.67e-05
- iotid20_table8 SHAP vs LIME Jaccard: p = 0.0625, Bonferroni p = 0.438