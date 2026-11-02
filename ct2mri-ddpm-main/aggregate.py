"""
Average the test results of the five cross-validation folds of one experiment.

    python aggregate.py

The experiment is selected by EQUALIZATION, MASK_TYPE and EXP_TAG in config.py.
For each fold it reads the newest runs/<exp_name(fold)>/results/metrics_summary_*.csv
written by test.py, and writes the mean and std across folds (plus every fold's
value) to runs/diffusion_<dataset>[_<tag>]_cv_summary.csv. Folds without results
are reported and skipped.
"""

import csv

import numpy as np

import config


def latest_summary(test_fold):
    results_dir = config.OUTPUT_ROOT / config.exp_name(test_fold) / "results"
    files = sorted(results_dir.glob("metrics_summary_*.csv"))
    return files[-1] if files else None


def read_means(path):
    with open(path, newline="") as f:
        return {row["metric"]: float(row["mean"]) for row in csv.DictReader(f)}


def main():
    per_fold = {}
    for fold in range(1, config.NUM_FOLDS + 1):
        path = latest_summary(fold)
        if path is None:
            print(f"fold {fold}: no metrics_summary_*.csv under {config.exp_name(fold)}/results, skipped")
            continue
        per_fold[fold] = read_means(path)
        print(f"fold {fold}: {path.relative_to(config.OUTPUT_ROOT)}")
    if not per_fold:
        raise SystemExit("No fold results found; run test.py first.")

    folds = sorted(per_fold)
    metrics = list(per_fold[folds[0]])
    rows = []
    for metric in metrics:
        values = [per_fold[f].get(metric, float("nan")) for f in folds]
        row = {"metric": metric, "mean": float(np.mean(values)), "std": float(np.std(values)),
               "num_folds": len(folds)}
        row.update({f"fold{f}": v for f, v in zip(folds, values)})
        rows.append(row)

    tag = f"_{config.EXP_TAG}" if config.EXP_TAG else ""
    out = config.OUTPUT_ROOT / f"diffusion_{config.DATASET_NAME}{tag}_cv_summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, ["metric", "mean", "std", "num_folds"] + [f"fold{f}" for f in folds])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{config.DATASET_NAME}{tag}: {len(folds)} fold(s) {folds}")
    for row in rows:
        print(f"  {row['metric']:<18}{row['mean']:.4f} +/- {row['std']:.4f}")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
