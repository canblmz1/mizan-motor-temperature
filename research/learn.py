"""Same-information classical estimators and validation-stopped neural ensembles."""

from __future__ import annotations

import copy
import json
import time
import warnings

import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def noop():
    pass


GUARD = noop
FIT_CALLS = 0


def weights(profiles):
    names, counts = np.unique(profiles, return_counts=True)
    mapping = dict(zip(names, counts, strict=True))
    w = np.array([1 / mapping[p] for p in profiles], dtype=float)
    return w / w.sum()


def scaler(data):
    w = weights(data["profile"])
    x = np.asarray(data["x"], dtype=float)
    mean = np.average(x, axis=0, weights=w)
    std = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    return {"mean": mean, "scale": np.where(std > 1e-8, std, 1.0)}


def candidates():
    result = []

    def add(family, **kwargs):
        result.append({"id": f"p{len(result):02d}", "family": family, **kwargs})

    for family in ("coolant", "constant_gap"):
        add(family)
    for family in ("instant_ridge", "ewma_ridge"):
        for alpha in (0.1, 10.0, 1000.0):
            add(family, alpha=alpha)
    for view in ("anchor", "ridge_residual"):
        for leaves in (15, 31):
            add("hgb", view=view, leaves=leaves)
    for family in ("extra_trees", "svr"):
        for view in ("anchor", "ridge_residual"):
            add(family, view=view)
    for view in ("anchor", "ridge_residual"):
        for architecture in ((64, 32), (128, 64)):
            add("network", view=view, architecture=list(architecture))
    add("constant_RC")
    return result


def score(truth, prediction):
    if len(truth) != len(prediction) or not np.isfinite(prediction).all():
        raise FloatingPointError("Incomplete/nonfinite prediction; no silent exclusions")
    e = np.asarray(prediction, dtype=float) - truth
    absolute = np.abs(e)
    return {
        "n": len(truth),
        "coverage": 1.0,
        "MAE_C": float(absolute.mean()),
        "RMSE_C": float(np.sqrt(np.mean(e * e))),
        "p95_abs_C": float(np.quantile(absolute, 0.95)),
        "maximum_underestimation_C": float(max(0, -np.min(e))),
        "max_absolute_C": float(np.max(absolute)),
        "bias_C": float(e.mean()),
    }


def evaluate(data, prediction):
    by_profile = {}
    for pid in np.unique(data["profile"]):
        select = data["profile"] == pid
        by_profile[str(pid)] = score(data["y"][select], prediction[select])
    return {
        "equal_profile_MAE_C": float(np.mean([r["MAE_C"] for r in by_profile.values()])),
        "pooled": score(data["y"], prediction),
        "profiles": by_profile,
    }


def train(config, data, normalization, validation, *, base=None, initial=None, rc_statistics=None):
    global FIT_CALLS
    FIT_CALLS += 1
    GUARD()
    family = config["family"]
    record = {"config": config, "normalization": normalization, "base": base}
    w = weights(data["profile"])
    target = np.asarray(data["y"], dtype=float) - data["anchor"]
    if family in {"coolant", "constant_gap"}:
        record["gap"] = 0.0 if family == "coolant" else float(np.average(target, weights=w))
        return record, {"trained": family == "constant_gap"}
    if family == "constant_RC":
        xtx, xty = np.asarray(rc_statistics["xtx"]), np.asarray(rc_statistics["xty"])
        upper = np.linalg.cholesky(xtx + np.eye(5) * 1e-8).T
        response = np.linalg.solve(upper.T, xty)
        coefficient, residual = nnls(upper, response)
        record.update({"coefficient": coefficient, "initial": initial, "base": None})
        return record, {
            "source_derivative_increments": rc_statistics["increments"],
            "normal_equation_residual": residual,
            "rates": coefficient.tolist(),
            "hidden_target_initialization": False,
        }
    dimension = 15 if family == "instant_ridge" else len(normalization["mean"])
    z = (
        np.asarray(data["x"][:, :dimension], dtype=float) - normalization["mean"][:dimension]
    ) / normalization["scale"][:dimension]
    record["dimension"] = dimension
    if family.endswith("ridge"):
        estimator = Ridge(alpha=config["alpha"]).fit(z, target, sample_weight=w * len(w))
        record.update(
            {"coefficient": estimator.coef_, "intercept": float(estimator.intercept_), "base": None}
        )
        return record, {"training_rows": len(z)}
    if config["view"] == "ridge_residual":
        target -= gap_predict(base, z)
    else:
        record["base"] = None
    mean = float(np.average(target, weights=w))
    std = max(1e-8, float(np.sqrt(np.average((target - mean) ** 2, weights=w))))
    scaled_target = (target - mean) / std
    record.update({"target_mean": mean, "target_scale": std})
    if family == "network":
        estimators = [
            MLPRegressor(
                hidden_layer_sizes=tuple(config["architecture"]),
                activation="relu",
                solver="adam",
                alpha=0.01,
                batch_size=256,
                learning_rate_init=0.001,
                max_iter=1,
                shuffle=True,
                early_stopping=False,
                random_state=seed,
            )
            for seed in (1729, 2718)
        ]
        record["estimators"] = estimators
        best, best_key, last_improvement = None, (float("inf"), float("inf")), 0
        log = []
        for epoch in range(1, 101):
            for estimator in estimators:
                estimator.partial_fit(z, scaled_target, sample_weight=w * len(w))
                GUARD()
            prediction = predict(record, validation)
            metrics = evaluate(validation, prediction)
            key = (metrics["equal_profile_MAE_C"], metrics["pooled"]["maximum_underestimation_C"])
            if key < best_key:
                best, best_key, last_improvement = copy.deepcopy(record), key, epoch
            log.append(
                {
                    "epoch": epoch,
                    "seed_training_loss": [float(e.loss_) for e in estimators],
                    "validation": metrics,
                }
            )
            if epoch == 1 or epoch % 10 == 0:
                print(
                    json.dumps(
                        {
                            "candidate": config["id"],
                            "epoch": epoch,
                            "validation_MAE_C": metrics["equal_profile_MAE_C"],
                            "best_epoch": last_improvement,
                        }
                    ),
                    flush=True,
                )
            if epoch >= 10 and epoch - last_improvement >= 15:
                break
        return best, {
            "epochs_run": epoch,
            "selected_common_epoch": last_improvement,
            "fixed_seeds": [1729, 2718],
            "best_validation": log[last_improvement - 1]["validation"],
            "curve": log,
            "stopping": "patience" if epoch < 100 else "registered epoch limit",
            "optimizer_convergence_claim": False,
        }
    if family == "hgb":
        estimator = HistGradientBoostingRegressor(
            max_leaf_nodes=config["leaves"],
            max_iter=250,
            learning_rate=0.07,
            max_bins=64,
            l2_regularization=10,
            early_stopping=False,
            random_state=1729,
        )
    elif family == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=120, max_leaf_nodes=512, min_samples_leaf=5, n_jobs=1, random_state=1729
        )
    elif family == "svr":
        estimator = SVR(C=10, gamma="scale", epsilon=0.2 / std, cache_size=24)
    else:
        raise ValueError(family)
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(z, scaled_target, sample_weight=w * len(w))
    record["estimator"] = estimator
    GUARD()
    return record, {
        "seconds": time.monotonic() - start,
        "warnings": [str(r.message) for r in caught],
        "fit_status": int(estimator.fit_status_) if family == "svr" else None,
    }


def gap_predict(ridge, standardized):
    return standardized[:, : ridge["dimension"]] @ ridge["coefficient"] + ridge["intercept"]


def predict(model, data):
    family = model["config"]["family"]
    if family in {"coolant", "constant_gap"}:
        return np.asarray(data["anchor"], dtype=float) + model["gap"]
    if family == "constant_RC":
        coefficient = model["coefficient"]
        output = np.empty(len(data["x"]), dtype=float)
        states = {}
        for i in range(len(output)):
            pid = data["profile"][i]
            x = data["x"][i]
            if pid not in states:
                initial = model["initial"]
                norm = initial["normalization"]
                standardized = (np.asarray(x[:15], dtype=float) - norm["mean"][:15]) / norm[
                    "scale"
                ][:15]
                state = float(
                    data["anchor"][i] + standardized @ initial["coefficient"] + initial["intercept"]
                )
                states[pid] = state
            state = states[pid]
            output[i] = state
            ambient, coolant = float(x[0]) * 100, float(x[1]) * 100
            heating = coefficient[2] * x[7] + coefficient[3] * x[10] + coefficient[4] * x[11]
            conductance = (coefficient[0] + coefficient[1]) / 100
            forcing = (coefficient[0] * coolant + coefficient[1] * ambient) / 100 + heating
            if conductance > 1e-12:
                decay = np.exp(-0.5 * conductance)
                state = decay * state + (1 - decay) * forcing / conductance
            else:
                state += 0.5 * forcing
            states[pid] = state
            if i % 4096 == 0:
                GUARD()
        return output
    prediction = np.empty(len(data["x"]), dtype=float)
    norm, dimension = model["normalization"], model["dimension"]
    for begin in range(0, len(prediction), 2048):
        end = min(begin + 2048, len(prediction))
        z = (
            np.asarray(data["x"][begin:end, :dimension], dtype=float) - norm["mean"][:dimension]
        ) / norm["scale"][:dimension]
        if family.endswith("ridge"):
            gap = gap_predict(model, z)
        else:
            if family == "network":
                value = np.mean([e.predict(z) for e in model["estimators"]], axis=0)
            else:
                value = model["estimator"].predict(z)
            gap = model["target_mean"] + model["target_scale"] * value
            if model["base"] is not None:
                gap += gap_predict(model["base"], z)
        prediction[begin:end] = data["anchor"][begin:end] + gap
        GUARD()
    return prediction
