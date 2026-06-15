# Data Documentation

## Original Paper Data

The paper used LinkedIn-derived career records with 26 binary inputs and two
categorical outputs. The original records contain personal information and are
not redistributed in this public companion.

## Included Fixture

Executable experiments use a deterministic privacy-safe synthetic fixture with
the same high-level interface: 26 binary input features, entropy selection to
11 features, six domain classes, eight position classes, and a 300/120 split.

## Redistribution Boundary

No private LinkedIn-derived records, direct identifiers, demographic sensitive
attributes, scraped profiles, or raw personal data are included. Published
paper results are separated from local synthetic-fixture results.

## Generated Artifacts

`make reproduce-results` writes `reports/latest/`, including:

- `metrics.json`
- `predictions.parquet`
- `statistical_tests.json`
- `privacy_fairness.json`
- `feature_selection.csv`
- `benchmark.csv`

## NOT_RUN Limitations

The original 420-person private benchmark remains `NOT_RUN`. This repository
does not claim group-fairness conclusions because the synthetic fixture does
not model demographic sensitive attributes.
