# Verified Results

## Provenance

This is a reconstructed multi-output benchmark on synthetic data. It is not the
original paper implementation and does not contain participant records.

## Local reproduction

`make reproduce-results` completed locally on June 11, 2026.

| Model | Micro F1 | Macro F1 | Hamming loss | Subgroup gap |
|---|---:|---:|---:|---:|
| Logistic | 0.741 | 0.653 | 0.187 | 0.011 |
| Random Forest | 0.763 | 0.645 | 0.167 | 0.001 |

Random forest improves micro F1 and hamming loss, while logistic regression has
slightly higher macro F1. The generated sensitive-group attribute is included
only to exercise subgroup evaluation; it is not evidence of real-world
fairness.
