"""Bounded local source fitting and a separate, fit-disabled final test stage."""

from __future__ import annotations

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"

import argparse
import json
import hashlib
import importlib.metadata
import inspect
import sys
import zipfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = MODELS = PROCESS = WORK_DIR = DATA_ZIP = None
DATA_LOCK = None
RUNTIME = None
START = time.monotonic()
LIMIT = 2400
PEAK = 0
MIN_FREE = 1 << 63
RESOURCE_ERROR = None
DONE = threading.Event()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def observe():
    global PEAK, MIN_FREE, RESOURCE_ERROR
    PEAK = max(PEAK, PROCESS.memory_info().rss)
    MIN_FREE = min(MIN_FREE, psutil.virtual_memory().available)
    if PEAK > 2048 * 1024**2:
        RESOURCE_ERROR = "Process RSS exceeded portable 2048 MiB cap"
    if MIN_FREE < 1024 * 1024**2:
        RESOURCE_ERROR = "Available RAM below 1024 MiB"
    if time.monotonic() - START > LIMIT:
        RESOURCE_ERROR = "Registered stage wall-time budget exhausted"


def guard():
    observe()
    if RESOURCE_ERROR:
        raise RuntimeError(RESOURCE_ERROR)


def watch():
    while not DONE.wait(0.2):
        observe()


def code_lock(create=False):
    names = ("PROTOCOL.json", "PROFILE_INVENTORY.json", "features.py", "learn.py",
             "run.py", "fast_svr.py", "export_models.py", "requirements.txt",
             "REFERENCE_ENVIRONMENT.json")
    hashes = {name: features.sha(HERE / name) for name in names}
    payload = {"files": hashes, "data": DATA_LOCK,
               "configurations": learn.candidates(), "runtime": RUNTIME}
    path = RESULTS / "CODE_LOCK.json"
    if path.exists():
        previous = read(path)
        if any(previous[k] != v for k, v in payload.items()):
            raise RuntimeError("Frozen source, dataset, split, environment or candidate grid changed")
    elif create:
        save(path, {"created_utc": datetime.now(UTC).isoformat(), **payload})
    else:
        raise RuntimeError("A completed source-stage code lock is required")


def verify_frozen():
    code_lock()
    manifest = read(RESULTS / "FROZEN_MANIFEST.json")
    for name, field in (("CODE_LOCK.json", "code_lock_sha256"),
                        ("SOURCE_SELECTION.json", "selection_sha256"),
                        ("SOURCE_CACHE_RECEIPT.json", "source_cache_receipt_sha256")):
        if features.sha(RESULTS / name) != manifest[field]:
            raise RuntimeError("Frozen source artifact changed: " + name)
    for record in manifest["models"].values():
        path = Path(record["model_path"]).resolve()
        if path.parent != MODELS or not path.is_file() or features.sha(path) != record["model_sha256"]:
            raise RuntimeError("Frozen model path or content changed")
    return manifest


def key(row):
    return (
        row["validation"]["equal_profile_MAE_C"],
        row["validation"]["pooled"]["maximum_underestimation_C"],
        row["config"]["id"],
    )


def family_key(config):
    return config["family"] + ("__" + config["view"] if "view" in config else "")


def persist_candidate(config, model, fit_log, validation):
    if config["family"] == "svr":
        import fast_svr
        indices = []
        for pid in np.unique(validation["profile"]):
            group = np.flatnonzero(validation["profile"] == pid)
            indices.extend(group[np.linspace(0, len(group) - 1, 12, dtype=int)])
        subset = {k: v[indices] for k, v in validation.items()}
        original = fast_svr.ORIGINAL_PREDICT(model, subset)
        accelerated = fast_svr.predict(model, subset)
        difference = float(np.max(np.abs(original - accelerated)))
        if difference > 1e-6:
            raise RuntimeError("Frozen RBF inference differs from native SVR")
        fit_log = {**fit_log, "actual_SVR_parameters": model["estimator"].get_params(),
                   "source_equivalence_points": len(indices),
                   "maximum_inference_difference_C": difference}
    prediction = learn.predict(model, validation)
    score = learn.evaluate(validation, prediction)
    model_path = MODELS / (config["id"] + ".joblib")
    if model_path.exists():
        raise RuntimeError("Refusing to overwrite a final candidate model")
    joblib.dump(model, model_path, compress=3)
    record = {
        "config": config,
        "model_path": str(model_path),
        "model_sha256": features.sha(model_path),
        "validation": score,
        "fit_log": fit_log,
    }
    save(RESULTS / "candidates" / (config["id"] + ".json"), record)
    print(
        json.dumps(
            {
                "completed_candidate": config["id"],
                "family": family_key(config),
                "validation_MAE_C": score["equal_profile_MAE_C"],
            }
        ),
        flush=True,
    )
    return record


def source():
    if (RESULTS / "FROZEN_MANIFEST.json").exists() or (
        RESULTS / "TEST_PARSE_STARTED.json"
    ).exists():
        raise RuntimeError("Completed source models cannot be refitted")
    fit, validation, receipt = features.cache("source", guard)
    save(RESULTS / "SOURCE_CACHE_RECEIPT.json", receipt)
    normalization = learn.scaler(fit)
    save(RESULTS / "SOURCE_SCALER.json", normalization)
    MODELS.mkdir(parents=True, exist_ok=True)
    records = []
    base = initial = None
    for config in learn.candidates():
        if config["id"] == "p08":
            ewma = min((r for r in records if r["config"]["family"] == "ewma_ridge"), key=key)
            instant = min((r for r in records if r["config"]["family"] == "instant_ridge"), key=key)
            base = joblib.load(ewma["model_path"])
            initial = joblib.load(instant["model_path"])
            save(
                RESULTS / "NONLINEAR_SHARED_BASE.json",
                {
                    "ewma": ewma["config"],
                    "instantaneous_RC_initialization": instant["config"],
                    "same_base_for_neural_and_nonlinear_classical_residual_views": True,
                },
            )
        completed = RESULTS / "candidates" / (config["id"] + ".json")
        guard()
        if completed.exists():
            record = read(completed)
            if (
                record["config"] != config
                or features.sha(record["model_path"]) != record["model_sha256"]
            ):
                raise RuntimeError("Completed candidate receipt mismatch")
        else:
            rc = (
                read(features.PRIVATE / "cache/source/RC_STATISTICS.json")
                if config["family"] == "constant_RC"
                else None
            )
            model, log = learn.train(
                config, fit, normalization, validation, base=base, initial=initial, rc_statistics=rc
            )
            record = persist_candidate(config, model, log, validation)
            del model
        records.append(record)
    by_family = {}
    for record in records:
        name = family_key(record["config"])
        if name not in by_family or key(record) < key(by_family[name]):
            by_family[name] = record
    selected = min(records, key=key)
    classical = min((r for r in records if r["config"]["family"] != "network"), key=key)
    neural = min((r for r in records if r["config"]["family"] == "network"), key=key)
    selection = {
        "selected": selected["config"],
        "classical": classical["config"],
        "neural": neural["config"],
        "by_family_view": by_family,
        "all_candidates": records,
        "test_numeric_values_opened": False,
    }
    save(RESULTS / "SOURCE_SELECTION.json", selection)
    # Freeze every family/view, rather than only a convenient weak comparison.
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "selected": selected["config"],
        "classical": classical["config"],
        "neural": neural["config"],
        "models": by_family,
        "selection_sha256": features.sha(RESULTS / "SOURCE_SELECTION.json"),
        "source_cache_receipt_sha256": features.sha(RESULTS / "SOURCE_CACHE_RECEIPT.json"),
        "code_lock_sha256": features.sha(RESULTS / "CODE_LOCK.json"),
        "test_numeric_parse_started": False,
    }
    save(RESULTS / "FROZEN_MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "SOURCE_FROZEN": True,
                "selected": selected["config"],
                "validation_MAE_C": selected["validation"]["equal_profile_MAE_C"],
                "strongest_classical": classical["config"],
                "best_neural": neural["config"],
            }
        ),
        flush=True,
    )


def test():
    if (RESULTS / "TEST_RESULTS.json").exists() or (RESULTS / "TEST_PARSE_STARTED.json").exists():
        raise RuntimeError("Final evaluation already started; refusing to overwrite or rerun it")
    manifest = verify_frozen()
    if features.sha(RESULTS / "SOURCE_SELECTION.json") != manifest["selection_sha256"]:
        raise RuntimeError("Model selection changed after freeze")
    for record in manifest["models"].values():
        if features.sha(record["model_path"]) != record["model_sha256"]:
            raise RuntimeError("A frozen model changed")
    marker = RESULTS / "TEST_PARSE_STARTED.json"
    if not marker.exists():
        save(
            marker,
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "frozen_manifest_sha256": features.sha(RESULTS / "FROZEN_MANIFEST.json"),
                "fit_calls": learn.FIT_CALLS,
            },
        )
    data, receipt = features.cache("test", guard)
    save(RESULTS / "TEST_CACHE_RECEIPT.json", receipt)
    metrics = {}
    output = {}
    for name, record in manifest["models"].items():
        model = joblib.load(record["model_path"])
        prediction = learn.predict(model, data)
        score = learn.evaluate(data, prediction)
        metrics[name] = {"config": record["config"], "test": score}
        output[name] = prediction
        print(
            json.dumps(
                {
                    "test_family": name,
                    "test_MAE_C": score["equal_profile_MAE_C"],
                    "max_abs_C": score["pooled"]["max_absolute_C"],
                }
            ),
            flush=True,
        )
        del model
        guard()
    destination = features.PRIVATE / "test_predictions.npz"
    if destination.exists():
        raise RuntimeError("Refusing to overwrite completed test prediction arrays")
    np.savez_compressed(destination, **output)
    if learn.FIT_CALLS:
        raise RuntimeError("Unexpected fit during final test")
    save(
        RESULTS / "TEST_RESULTS.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "selected": manifest["selected"],
            "classical": manifest["classical"],
            "neural": manifest["neural"],
            "families": metrics,
            "prediction_file": str(destination),
            "prediction_sha256": features.sha(destination),
            "fit_calls_during_test": learn.FIT_CALLS,
            "no_initial_or_lagged_hidden_target": True,
            "complete_native_rows_no_warmup_exclusion": True,
        },
    )


def parser_for_stage(export=False):
    parser = argparse.ArgumentParser(description=(
        "Export locally trained frozen models without fitting." if export else
        "Optional CPU research reference. Source fits the fixed grid; test evaluates frozen models."))
    if not export:
        parser.add_argument("stage", choices=("source", "test"))
    parser.add_argument("--data-zip", type=Path, required=True,
                        help="Existing ZIP containing the exact public measures_v2.csv")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="New, nonexistent directory for source; same directory for test/export")
    return parser


def dataset_identity(path):
    """Byte verification only; held-out numeric fields are not parsed here."""
    protocol = read(HERE / "PROTOCOL.json")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with zipfile.ZipFile(path) as archive:
        matches = [m for m in archive.infolist() if m.filename == protocol["dataset"]["csv"]]
        if len(matches) != 1 or matches[0].file_size != protocol["dataset"]["csv_bytes"]:
            raise RuntimeError("Exactly one pinned CSV member with the recorded size is required")
        csv_digest = hashlib.sha256()
        with archive.open(matches[0]) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                csv_digest.update(chunk)
    if csv_digest.hexdigest() != protocol["dataset"]["csv_sha256"]:
        raise RuntimeError("Dataset CSV is not the recorded public version-3 content")
    return {"archive_sha256": digest.hexdigest(), "csv_sha256": csv_digest.hexdigest(),
            "csv_bytes": matches[0].file_size}


def configure(data_zip, work_dir, *, create=False):
    """Load optional ML dependencies only after explicit CLI arguments are parsed."""
    global RESULTS, MODELS, PROCESS, WORK_DIR, DATA_ZIP, DATA_LOCK, RUNTIME
    global joblib, np, psutil, features, learn
    WORK_DIR, DATA_ZIP = work_dir.resolve(), data_zip.resolve(strict=True)
    if not DATA_ZIP.is_file():
        raise ValueError("--data-zip must name an existing file")
    if WORK_DIR == HERE or WORK_DIR.is_relative_to(HERE):
        raise ValueError("Keep generated work outside the research source directory")
    if create:
        if WORK_DIR.exists():
            raise FileExistsError("Source requires a new directory; partial runs are not overwritten")
    elif not WORK_DIR.is_dir():
        raise FileNotFoundError("Use the same completed source --work-dir")
    DATA_LOCK = dataset_identity(DATA_ZIP)
    import joblib
    import numpy as np
    import psutil
    import features
    import learn
    import fast_svr
    from threadpoolctl import threadpool_limits
    # The recorded cloud run used the same two-thread numerical limit.
    threadpool_limits(limits=2)
    PROCESS = psutil.Process()
    affinity = None
    if hasattr(PROCESS, "cpu_affinity"):
        affinity = PROCESS.cpu_affinity()[:2]
        PROCESS.cpu_affinity(affinity)
    RESULTS, MODELS = WORK_DIR / "results", WORK_DIR / "models"
    features.PRIVATE, features.RESULTS = WORK_DIR, RESULTS
    features.DATA_ZIP, features.ARCHIVE_SHA256 = DATA_ZIP, DATA_LOCK["archive_sha256"]
    native = "sample_weight" in inspect.signature(learn.MLPRegressor.partial_fit).parameters
    if not native:
        if not all(features.INVENTORY["profile_rows"][str(p)] >= 256
                   for p in features.INVENTORY["training_profiles"]):
            raise RuntimeError("Old MLP API cannot represent nonuniform profile weights")
        original_partial_fit = learn.MLPRegressor.partial_fit

        def uniform_partial_fit(self, x, y, sample_weight=None):
            if sample_weight is not None and not np.allclose(sample_weight, 1.0, rtol=0, atol=1e-12):
                raise RuntimeError("Refusing to discard nonuniform training weights")
            return original_partial_fit(self, x, y)

        learn.MLPRegressor.partial_fit = uniform_partial_fit
    original_svr = learn.SVR

    def cached_svr(**kwargs):
        kwargs["cache_size"] = 1024
        return original_svr(**kwargs)

    learn.SVR = cached_svr
    learn.predict = fast_svr.predict
    learn.GUARD = guard
    RUNTIME = {"python": sys.version.split()[0],
               "versions": {name: importlib.metadata.version(name)
                            for name in ("numpy", "scipy", "scikit-learn", "joblib", "psutil", "threadpoolctl")},
               "MLP_uniform_weight_compatibility": not native,
               "CPU_threads": 2, "affinity_enforced": affinity is not None}
    if create:
        WORK_DIR.mkdir(parents=True, exist_ok=False)
        RESULTS.mkdir()
    return RUNTIME


def main(argv=None):
    global LIMIT, START
    parser = parser_for_stage()
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    LIMIT = 2400 if args.stage == "source" else 300
    START = time.monotonic()
    configure(args.data_zip, args.work_dir, create=args.stage == "source")
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    completed = False
    try:
        guard()
        code_lock(create=args.stage == "source")
        source() if args.stage == "source" else test()
        code_lock()
        completed = True
    except Exception as error:
        save(RESULTS / (args.stage + f"_INTERRUPTION_{time.time_ns()}.json"),
             {"error": repr(error), "fit_calls": learn.FIT_CALLS})
        raise
    finally:
        DONE.set()
        thread.join(timeout=1)
        save(RESULTS / (args.stage + f"_RESOURCES_{time.time_ns()}.json"),
             {"completed": completed, "elapsed_seconds": time.monotonic() - START,
              "max_process_RSS_bytes": PEAK, "min_available_ram_bytes": MIN_FREE,
              "effective_RSS_cap_mib": 2048, "stage_seconds_limit": LIMIT,
              "guard_scope": "sampled/cooperative, not a hard process sandbox",
              "runtime": RUNTIME, "fit_calls": learn.FIT_CALLS,
              "cloud_or_network_operations": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
