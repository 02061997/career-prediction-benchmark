# Paper Specification

Source: *Multi-Output Career Prediction: Dataset, Method, and Benchmark
Suite* (2023).

## Implemented specification

- 420 records with 26 binary historical features.
- Entropy and joint-entropy-informed reduction to 11 features.
- Two categorical outputs: domain (6 classes) and position (8 classes).
- Random 300/120 train/test partition.
- Random Forest, Decision Tree, Extra Tree, Extra Trees, 1-NN, and Radius
  Neighbors classifiers.
- Per-output accuracy and strict joint accuracy.

The original LinkedIn-derived dataset contains personal information and is not
distributed. The executable benchmark uses a deterministic synthetic fixture
with the same dimensions and target cardinalities. Published paper results are
stored separately and are not presented as reproduced.
