# Multi-Output Career Prediction Benchmark

Paper-faithful public research companion for *Multi-Output Career Prediction:
Dataset, Method, and Benchmark Suite*.

The implementation follows the paper's 26 binary inputs, entropy-based
selection to 11 features, two categorical outputs, 300/120 split, and six
classifiers. The original LinkedIn-derived records are private and are not
redistributed; executable experiments use a deterministic privacy-safe fixture.

```bash
uv sync
make test
make reproduce-smoke
make reproduce-results
```

See `PAPER_SPEC.md` and `RESULTS.md` for the distinction between published and
local reference results.

See `PRIVACY_FAIRNESS.md` for the public boundary: the runnable fixture has no
direct identifiers or demographic sensitive attributes, and the repository does
not claim group-fairness conclusions.
