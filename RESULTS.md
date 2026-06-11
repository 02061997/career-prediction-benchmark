# Results

## Published paper results

| Model | Domain accuracy | Position accuracy |
|---|---:|---:|
| Random Forest | 91.21% | 95.97% |
| Decision Tree | 90.60% | 93.23% |
| Extra Tree | 89.65% | 90.50% |
| Extra Trees | 85.87% | 90.70% |
| K-Neighbors | 90.28% | 92.62% |
| Radius Neighbors | 60.65% | 45.67% |

These values are transcribed from the paper and were **not reproduced** because
the original 420-person LinkedIn-derived dataset is unavailable and contains
personal information.

## Local reference results

The June 11, 2026 synthetic run validates the complete dimensions, feature
selection, train/test split, multioutput behavior, and six-model benchmark.
Accuracies are intentionally not compared numerically with the private dataset:
Random Forest obtained 41.67% domain, 75.00% position, and 31.67% strict joint
accuracy on the synthetic fixture.

Machine-readable evidence is under `reports/latest/`.
