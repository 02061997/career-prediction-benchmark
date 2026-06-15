# Privacy and Fairness Boundary

This repository reconstructs the paper's benchmark shape without publishing
the private LinkedIn-derived records.

## What Is Public

- A deterministic synthetic fixture with 420 rows, 26 binary features, six
  domain classes, and eight position classes.
- Entropy-based feature selection to 11 features.
- Six multiclass-multioutput classifiers.
- Machine-readable `NOT_RUN` records for the private dataset experiment.

## What Is Not Public

- Names, LinkedIn profile URLs, employers, emails, phone numbers, resumes, or
  other direct identifiers.
- The original 420-person dataset.
- Any demographic sensitive attributes such as age, gender, race, ethnicity, or
  nationality.

## Fairness Limitations

Because the executable fixture excludes demographic sensitive attributes, this
repository cannot estimate group fairness, disparate impact, or subgroup
calibration. It should not be used for automated hiring, screening, ranking, or
career counseling decisions. Its purpose is methodological inspection of the
paper's dimensionality, multioutput formulation, and benchmark mechanics.
