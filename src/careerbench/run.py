import argparse
import pandas as pd
from .artifacts import environment, output_dir, save
from .core import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = output_dir(args.smoke)
    records = evaluate(500 if args.smoke else 5000)
    frame = pd.DataFrame(records)
    frame.to_parquet(out / "predictions.parquet", index=False)
    save(out / "metrics.json", records)
    save(out / "statistical_tests.json", {"models": len(records)})
    save(out / "environment.json", environment())
    save(
        out / "data_manifest.json",
        {"source": "sklearn deterministic synthetic multilabel generator"},
    )
    save(out / "config.yaml", {"labels": 6, "train_fraction": 0.7})
    (out / "run.log").write_text("completed\n")
    print(out)


if __name__ == "__main__":
    main()
