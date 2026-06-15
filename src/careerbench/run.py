import argparse

import pandas as pd

from .artifacts import environment, output_dir, publish_latest, save
from .core import PUBLISHED_RESULTS, generate_reference_dataset, privacy_audit, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = output_dir(args.smoke)
    audit = privacy_audit(generate_reference_dataset())
    records, predictions, selected = evaluate()
    predictions.to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(records).to_csv(out / "benchmark.csv", index=False)
    save(out / "metrics.json", records)
    save(out / "privacy_fairness.json", audit)
    save(
        out / "statistical_tests.json",
        {
            "published_results": PUBLISHED_RESULTS,
            "published_results_reproduced": False,
            "local_reference_results": records,
            "privacy_fairness_audit": audit,
            "not_run": [
                {
                    "experiment": "Original LinkedIn-derived 420-person benchmark",
                    "reason": "The private dataset contains personal information and is unavailable.",
                }
            ],
        },
    )
    save(out / "environment.json", environment())
    save(
        out / "data_manifest.json",
        {
            "source": "deterministic privacy-safe synthetic reference fixture",
            "original_private_dataset_included": False,
            "samples": 420,
            "input_features": 26,
            "selected_features": selected,
            "outputs": {"domain_classes": 6, "position_classes": 8},
            "privacy_controls": audit,
        },
    )
    save(
        out / "config.yaml",
        {"train_samples": 300, "test_samples": 120, "selected_feature_count": 11},
    )
    (out / "run.log").write_text("completed\n")
    if not args.smoke:
        publish_latest(out)
    print(out)


if __name__ == "__main__":
    main()
