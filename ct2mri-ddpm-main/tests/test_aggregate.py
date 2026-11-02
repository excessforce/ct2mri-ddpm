"""aggregate.py averages the newest metrics_summary CSV of each fold."""

import csv

import pytest

import aggregate
import config


def write_summary(results_dir, stamp, mae):
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"metrics_summary_{stamp}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, ["metric", "mean", "std"])
        writer.writeheader()
        writer.writerows([{"metric": "fid", "mean": 10.0, "std": ""},
                          {"metric": "mae", "mean": mae, "std": 0.01}])


def test_aggregate_uses_latest_summary_per_fold(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    for fold, mae in [(1, 0.1), (2, 0.3)]:
        results = tmp_path / config.exp_name(fold) / "results"
        write_summary(results, "2000-01-01-00-00-00", 99.0)   # older run, must be ignored
        write_summary(results, "2026-01-01-00-00-00", mae)

    aggregate.main()

    with open(tmp_path / f"diffusion_{config.DATASET_NAME}_cv_summary.csv", newline="") as f:
        rows = {r["metric"]: r for r in csv.DictReader(f)}
    assert float(rows["mae"]["mean"]) == pytest.approx(0.2)
    assert float(rows["mae"]["std"]) == pytest.approx(0.1)
    assert rows["mae"]["num_folds"] == "2"


def test_aggregate_without_results_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(SystemExit):
        aggregate.main()
