# Human evaluation of explanation stability

Participants: 13. Responses: 260.

Roles: UG student 11, PG student 2
Network experience: Less than 1 year 7, None 3, 1 to 3 years 3
ML familiarity median 3/7, security familiarity median 3/7

| category      |   alerts_per_reader |   reason_correct_% |    sd |   self_confidence_1to7 |   trust_1to7 |   median_time_s |   act_% |   escalate_% |   ignore_% |
|:--------------|--------------------:|-------------------:|------:|-----------------------:|-------------:|----------------:|--------:|-------------:|-----------:|
| CERTIFIED     |                   8 |              45.19 | 43.46 |                   5.44 |         4.95 |           29.86 |   56.73 |        41.35 |       1.92 |
| UNCERTIFIED   |                   6 |              43.59 | 44.9  |                   5.28 |         4.51 |           31.09 |   41.03 |        58.97 |       0    |
| CONTRADICTORY |                   6 |              42.31 | 43.89 |                   5.55 |         4.6  |           36.96 |   41.03 |        58.97 |       0    |

## Paired Wilcoxon tests (participant means), Bonferroni over 12 tests

| comparison                   | measure   |   n |   median_diff |   p_raw |   p_bonferroni |
|:-----------------------------|:----------|----:|--------------:|--------:|---------------:|
| CERTIFIED vs UNCERTIFIED     | acc       |  13 |        0      |  0.375  |         1      |
| CERTIFIED vs UNCERTIFIED     | trust     |  13 |        0.0417 |  0.5049 |         1      |
| CERTIFIED vs UNCERTIFIED     | selfconf  |  13 |        0.125  |  0.4121 |         1      |
| CERTIFIED vs UNCERTIFIED     | time_s    |  13 |       -4.603  |  0.2439 |         1      |
| CERTIFIED vs CONTRADICTORY   | acc       |  13 |        0      |  0.4688 |         1      |
| CERTIFIED vs CONTRADICTORY   | trust     |  13 |       -0.125  |  0.6611 |         1      |
| CERTIFIED vs CONTRADICTORY   | selfconf  |  13 |        0      |  0.6069 |         1      |
| CERTIFIED vs CONTRADICTORY   | time_s    |  13 |       -2.964  |  0.7354 |         1      |
| UNCERTIFIED vs CONTRADICTORY | acc       |  13 |        0      |  0.75   |         1      |
| UNCERTIFIED vs CONTRADICTORY | trust     |  13 |       -0.1667 |  0.7422 |         1      |
| UNCERTIFIED vs CONTRADICTORY | selfconf  |  13 |       -0.1667 |  0.0547 |         0.6562 |
| UNCERTIFIED vs CONTRADICTORY | time_s    |  13 |       -0.2015 |  0.7354 |         1      |

## Action by category

| category      |   act |   escalate |   ignore |
|:--------------|------:|-----------:|---------:|
| CERTIFIED     |    59 |         43 |        2 |
| CONTRADICTORY |    32 |         46 |        0 |
| UNCERTIFIED   |    32 |         46 |        0 |

chi-square = 9.99, dof = 4, p = 0.04063

Post-study agreement (1-7, mean): notes helped 4.85; could tell when unsure 4.85; would rely on them 4.69

Free-text answers to 'which alerts looked less reliable' must be coded by two team members independently (spotted the UNVERIFIED banner: yes / no / partly) and agreement reported as Cohen's kappa before quoting a rate.
