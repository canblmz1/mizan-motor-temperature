"""Verify frozen artifact digests and recorded predictions; never fit a model."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from mizan_motor import MotorTemperatureEstimator


ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ("ambient", "coolant", "u_d", "u_q", "i_d", "i_q", "motor_speed")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_digests() -> int:
    provenance = json.loads((ROOT / "docs/provenance.json").read_text(encoding="utf-8"))
    for entry in provenance["frozen_files"]:
        path = ROOT / entry["path"]
        require(path.stat().st_size == entry["bytes"], f"Size differs: {entry['path']}")
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        require(digest.hexdigest() == entry["SHA256"], f"SHA-256 differs: {entry['path']}")
    return len(provenance["frozen_files"])


def check_startup_predictions() -> dict:
    with (ROOT / "examples/telemetry.csv").open(encoding="utf-8", newline="") as stream:
        samples = list(csv.DictReader(stream))
    with (ROOT / "examples/expected.csv").open(encoding="utf-8", newline="") as stream:
        expected = list(csv.DictReader(stream))
    require(len(samples) == len(expected) == 36, "The complete 36-row fixture is required")
    differences = {}
    for model_name in ("p17", "p06"):
        estimator = MotorTemperatureEstimator(ROOT / "models" / f"{model_name}.json")
        previous = None
        errors = []
        for sample, reference in zip(samples, expected):
            require(sample["session_id"] == reference["session_id"], "Fixture order differs")
            if sample["session_id"] != previous:
                estimator.reset()
                previous = sample["session_id"]
            value = estimator.step({name: float(sample[name]) for name in CHANNELS})
            errors.append(abs(value - float(reference[f"expected_{model_name}_c"])))
        difference = max(errors)
        require(difference <= 1e-6, f"{model_name}: frozen prediction agreement failed")
        differences[model_name] = difference
    return {"rows_per_model": len(samples), "maximum_difference_c": differences}


def check_recorded_scores() -> dict:
    """Replay published aggregates from old predictions, not a new accuracy test."""
    benchmark = json.loads((ROOT / "results/benchmark.json").read_text(encoding="utf-8"))
    errors = defaultdict(lambda: defaultdict(list))
    with gzip.open(ROOT / "results/predictions.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            reference = float(row["reference_pm_C"])
            require(math.isfinite(reference), "Nonfinite recorded reference")
            for family, column in (("network__anchor", "selected_C"), ("ewma_ridge", "classical_C")):
                value = float(row[column])
                require(math.isfinite(value), "Nonfinite recorded prediction")
                errors[family][row["profile"]].append(abs(value - reference))
    replayed = {}
    for family, groups in errors.items():
        require(set(groups) == {"60", "62", "74"}, "Recorded session identity differs")
        require(sum(map(len, groups.values())) == 63904, "Recorded row coverage differs")
        macro = math.fsum(math.fsum(values) / len(values) for values in groups.values()) / 3
        recorded = benchmark["families"][family]["test"]["equal_profile_MAE_C"]
        require(abs(macro - recorded) <= 2e-6, f"Recorded macro MAE differs: {family}")
        replayed[family] = macro
    return replayed


def main() -> None:
    result = {
        "status": "PASS",
        "frozen_file_digests_checked": check_digests(),
        "startup_regression": check_startup_predictions(),
        "recorded_session_macro_mae_c": check_recorded_scores(),
        "new_training": False,
        "new_independent_accuracy_experiment": False,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
