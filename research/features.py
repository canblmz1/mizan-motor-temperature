"""Causal 2 Hz telemetry histories. The hidden rotor channel never enters features."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "PROTOCOL.json").read_text(encoding="utf-8"))
INVENTORY = json.loads((HERE / "PROFILE_INVENTORY.json").read_text(encoding="utf-8"))
PRIVATE = None  # Configured by the explicit research CLI.
DATA_ZIP = None
RESULTS = None
ARCHIVE_SHA256 = None
INPUTS = ("ambient", "coolant", "u_d", "u_q", "i_d", "i_q", "motor_speed")
TAUS = np.array([10, 60, 300, 900, 1800], dtype=float)
ALPHA = (1 - np.exp(-0.5 / TAUS))[:, None]
VAR_INPUTS = [2, 3, 4, 5, 6]
BASE_NAMES = list(INPUTS) + [
    "i_squared",
    "u_squared",
    "electrical_dot",
    "abs_speed",
    "speed_squared",
    "q_flux_proxy",
    "d_flux_proxy",
    "rotating",
]
NAMES = (
    BASE_NAMES
    + [f"ewma_{tau}s_{name}" for tau in (10, 60, 300, 900, 1800) for name in BASE_NAMES]
    + [f"ewstd_{tau}s_{BASE_NAMES[i]}" for tau in (60, 300, 900) for i in VAR_INPUTS]
)
DIMENSION = len(NAMES)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def base_features(values):
    ambient, coolant, ud, uq, i_d, i_q, speed = values
    rotating = abs(speed) >= 100
    return np.array(
        [
            ambient / 100,
            coolant / 100,
            ud / 500,
            uq / 500,
            i_d / 200,
            i_q / 200,
            speed / 6000,
            (i_d * i_d + i_q * i_q) / 40000,
            (ud * ud + uq * uq) / 250000,
            (ud * i_d + uq * i_q) / 100000,
            abs(speed) / 6000,
            (speed / 6000) ** 2,
            10 * uq / speed if rotating else 0,
            10 * ud / speed if rotating else 0,
            float(rotating),
        ],
        dtype=float,
    )


class History:
    def __init__(self):
        self.mean = None
        self.second = None

    def step(self, online_values):
        base = base_features(online_values)
        if not np.isfinite(base).all():
            raise ValueError("Nonfinite available telemetry")
        observed = base[VAR_INPUTS]
        if self.mean is None:
            self.mean = np.tile(base, (5, 1))
            self.second = np.tile(observed * observed, (5, 1))
        else:
            self.mean += ALPHA * (base - self.mean)
            self.second += ALPHA * (observed * observed - self.second)
        variance = np.maximum(0, self.second[1:4] - self.mean[1:4, VAR_INPUTS] ** 2)
        return np.concatenate((base, self.mean.ravel(), np.sqrt(variance).ravel())).astype(
            np.float32
        )


def create_arrays(folder, count):
    folder.mkdir(parents=True, exist_ok=True)
    return {
        "x": np.lib.format.open_memmap(
            folder / "x.npy", mode="w+", dtype=np.float32, shape=(count, DIMENSION)
        ),
        "y": np.lib.format.open_memmap(
            folder / "y.npy", mode="w+", dtype=np.float32, shape=(count,)
        ),
        "anchor": np.lib.format.open_memmap(
            folder / "anchor.npy", mode="w+", dtype=np.float32, shape=(count,)
        ),
        "profile": np.lib.format.open_memmap(
            folder / "profile.npy", mode="w+", dtype=np.int16, shape=(count,)
        ),
        "ordinal": np.lib.format.open_memmap(
            folder / "ordinal.npy", mode="w+", dtype=np.int32, shape=(count,)
        ),
    }


def load_arrays(folder):
    return {
        name: np.load(folder / (name + ".npy"), mmap_mode="r")
        for name in ("x", "y", "anchor", "profile", "ordinal")
    }


def cache(stage, guard=lambda: None):
    """Only the source stage may parse calibration labels; only frozen test parses tests."""
    if PRIVATE is None or DATA_ZIP is None or RESULTS is None:
        raise RuntimeError("Use run.py with explicit --data-zip and --work-dir")
    if stage not in {"source", "test"}:
        raise ValueError(stage)
    if stage == "test" and not (RESULTS / "FROZEN_MANIFEST.json").exists():
        raise RuntimeError("Test measurements cannot be parsed before model freeze")
    root = PRIVATE / "cache" / stage
    signature = {
        "features_code_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(HERE / "PROTOCOL.json"),
        "archive_sha256": ARCHIVE_SHA256,
    }
    if (root / "COMPLETE.json").exists():
        existing = json.loads((root / "COMPLETE.json").read_text())
        if existing["signature"] != signature:
            raise RuntimeError("Cached data belong to different locked feature code")
        for row in existing["arrays"]:
            if sha(root / row["path"]) != row["sha256"]:
                raise RuntimeError("Feature cache changed")
        if stage == "source":
            return load_arrays(root / "fit"), load_arrays(root / "validation"), existing
        return load_arrays(root / "test"), existing
    if sha(DATA_ZIP) != signature["archive_sha256"]:
        raise RuntimeError("Measurement archive changed")
    count = {int(k): v for k, v in INVENTORY["profile_rows"].items()}
    train_ids = INVENTORY["training_profiles"]
    val_ids = PROTOCOL["splits"]["validation_profiles"]
    test_ids = PROTOCOL["splits"]["test_profiles"]
    selected = {
        pid: set(np.linspace(0, count[pid] - 1, min(256, count[pid]), dtype=int))
        for pid in train_ids
    }
    fit_count = sum(len(v) for v in selected.values())
    if stage == "source":
        fit = create_arrays(root / "fit", fit_count)
        validation = create_arrays(root / "validation", sum(count[p] for p in val_ids))
        allowed = set(train_ids + val_ids)
    else:
        test = create_arrays(root / "test", sum(count[p] for p in test_ids))
        allowed = set(test_ids)
    states, ordinals = {}, {}
    fit_position = val_position = test_position = 0
    parsed_profiles = set()
    # Constant RC identification sufficient statistics; only source training rows.
    rc_xtx, rc_xty = np.zeros((5, 5)), np.zeros(5)
    rc_n = 0
    previous_rc = {}
    numeric_rows = 0
    with (
        zipfile.ZipFile(DATA_ZIP) as z,
        z.open(PROTOCOL["dataset"]["csv"]) as stream,
    ):
        reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
        for row in reader:
            pid = int(float(row["profile_id"]))
            if pid not in allowed:
                continue  # The withheld numeric strings are never converted to measurements.
            ordinal = ordinals.get(pid, 0)
            ordinals[pid] = ordinal + 1
            raw = [float(row[name]) for name in INPUTS]
            target = float(row["pm"])
            if not math.isfinite(target):
                raise ValueError("Missing rotor reference; requires explicit review")
            if pid not in states:
                states[pid] = History()
            x = states[pid].step(raw)
            parsed_profiles.add(pid)
            numeric_rows += 1
            destination = None
            if stage == "test":
                destination, position = test, test_position
                test_position += 1
            elif pid in val_ids:
                destination, position = validation, val_position
                val_position += 1
            else:
                old = previous_rc.get(pid)
                if old is not None:
                    # Positive heat loss/rate products, in scaled coordinates.
                    old_target, old_raw = old
                    design = np.array(
                        [
                            (old_raw[1] - old_target) / 100,
                            (old_raw[0] - old_target) / 100,
                            (old_raw[4] ** 2 + old_raw[5] ** 2) / 40000,
                            abs(old_raw[6]) / 6000,
                            (old_raw[6] / 6000) ** 2,
                        ]
                    )
                    increment = (target - old_target) / 0.5
                    weight = 1 / count[pid]
                    rc_xtx += weight * np.outer(design, design)
                    rc_xty += weight * design * increment
                    rc_n += 1
                previous_rc[pid] = (target, raw)
                if ordinal in selected[pid]:
                    destination, position = fit, fit_position
                    fit_position += 1
            if destination is not None:
                destination["x"][position] = x
                destination["y"][position] = target
                destination["anchor"][position] = raw[1]
                destination["profile"][position] = pid
                destination["ordinal"][position] = ordinal
            if numeric_rows % 100000 == 0:
                guard()
                print(
                    json.dumps({"feature_stage": stage, "numeric_rows": numeric_rows}), flush=True
                )
    for pid in allowed:
        if ordinals.get(pid) != count[pid]:
            raise RuntimeError("Parsed profile count differs from index-only inventory")
    if stage == "source":
        if fit_position != fit_count or val_position != len(validation["y"]):
            raise RuntimeError("Source cache row count mismatch")
        groups = {"fit": fit, "validation": validation}
        save(
            root / "RC_STATISTICS.json",
            {
                "xtx": rc_xtx.tolist(),
                "xty": rc_xty.tolist(),
                "increments": rc_n,
                "source_profiles": sorted(train_ids),
                "full_training_labels_used_only_for_RC": True,
            },
        )
    else:
        if test_position != len(test["y"]):
            raise RuntimeError("Test cache row count mismatch")
        groups = {"test": test}
    for arrays in groups.values():
        for array in arrays.values():
            array.flush()
    receipt = {
        "signature": signature,
        "stage": stage,
        "numeric_rows_parsed": numeric_rows,
        "numeric_profiles_parsed": sorted(parsed_profiles),
        "feature_dimension": DIMENSION,
        "feature_names": NAMES,
        "hidden_temperature_in_features_or_initial_state": False,
        "arrays": [
            {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(root.rglob("*.npy"))
        ],
    }
    save(root / "COMPLETE.json", receipt)
    return (
        (load_arrays(root / "fit"), load_arrays(root / "validation"), receipt)
        if stage == "source"
        else (load_arrays(root / "test"), receipt)
    )
