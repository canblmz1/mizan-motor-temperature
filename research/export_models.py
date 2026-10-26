"""Export actual frozen coefficients, not replayed labels or a new fitted model."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"

from pathlib import Path


HERE = Path(__file__).resolve().parent


def ridge_payload(record):
    return {
        "coefficient": record["coefficient"],
        "intercept": record["intercept"],
        "mean": record["normalization"]["mean"],
        "scale": record["normalization"]["scale"],
    }


def export(record):
    config = record["config"]
    family = config["family"]
    out = {
        "kind": family,
        "config": config,
        "inputs": list(features.INPUTS),
        "feature_names": features.NAMES,
        "sample_period_seconds": 0.5,
        "hidden_target_or_initial_temperature_required": False,
        "safety_certified": False,
    }
    if family in {"coolant", "constant_gap"}:
        return {**out, "gap": record["gap"]}
    if family == "constant_RC":
        return {
            **out,
            "coefficient": record["coefficient"],
            "initial": ridge_payload(record["initial"]),
        }
    out.update(
        {
            "mean": record["normalization"]["mean"],
            "scale": record["normalization"]["scale"],
            "dimension": record["dimension"],
        }
    )
    if family.endswith("ridge"):
        return {**out, **ridge_payload(record)}
    out.update(
        {
            "target_mean": record["target_mean"],
            "target_scale": record["target_scale"],
            "base": ridge_payload(record["base"]) if record["base"] is not None else None,
        }
    )
    if family == "network":
        out["members"] = [
            {"weights": [matrix.T for matrix in estimator.coefs_], "biases": estimator.intercepts_}
            for estimator in record["estimators"]
        ]
    elif family == "hgb":
        estimator = record["estimator"]
        out["initial_prediction"] = float(estimator._baseline_prediction[0, 0])
        out["trees"] = []
        for predictors in estimator._predictors:
            nodes = predictors[0].nodes
            out["trees"].append(
                {
                    "leaf": nodes["is_leaf"],
                    "feature": nodes["feature_idx"],
                    "threshold": nodes["num_threshold"],
                    "left": nodes["left"],
                    "right": nodes["right"],
                    "value": nodes["value"],
                }
            )
    elif family == "extra_trees":
        out["trees"] = []
        for estimator in record["estimator"].estimators_:
            tree = estimator.tree_
            out["trees"].append(
                {
                    "leaf": tree.children_left == -1,
                    "feature": tree.feature,
                    "threshold": tree.threshold,
                    "left": tree.children_left,
                    "right": tree.children_right,
                    "value": tree.value[:, 0, 0],
                }
            )
    elif family == "svr":
        estimator = record["estimator"]
        out.update(
            {
                "gamma": float(estimator._gamma),
                "support": estimator.support_vectors_,
                "dual": estimator.dual_coef_[0],
                "svm_intercept": float(estimator.intercept_[0]),
            }
        )
    else:
        raise ValueError(family)
    return out


def export_frozen():
    manifest = run.verify_frozen()
    destination = run.WORK_DIR / "exported_models"
    destination.mkdir(exist_ok=False)
    selected_ids = {manifest[name]["id"] for name in ("selected", "classical", "neural")}
    selected_ids |= {
        row["config"]["id"]
        for name, row in manifest["models"].items()
        if name in {"ewma_ridge", "instant_ridge", "constant_RC"}
    }
    records = {row["config"]["id"]: row for row in manifest["models"].values()}
    for candidate_id in sorted(selected_ids):
        row = records[candidate_id]
        if features.sha(row["model_path"]) != row["model_sha256"]:
            raise RuntimeError("Frozen native model differs")
        payload = export(joblib.load(row["model_path"]))
        payload["source_only_selection"] = True
        payload["native_model_sha256"] = row["model_sha256"]
        run.save(destination / (candidate_id + ".json"), payload)
    run.save(
        destination / "MODEL_INDEX.json",
        {
            "selected": manifest["selected"],
            "classical": manifest["classical"],
            "neural": manifest["neural"],
            "exported_ids": sorted(selected_ids),
            "no_fitting": True,
        },
    )
    print("EXPORTED", sorted(selected_ids))


def main(argv=None):
    import sys
    global run, features, joblib
    import run
    parser = run.parser_for_stage(export=True)
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    run.configure(args.data_zip, args.work_dir)
    import features
    import joblib
    export_frozen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
