"""Exact frozen RBF-SVR formula in bounded BLAS batches; no parameter fitting."""

import numpy as np

import learn

ORIGINAL_PREDICT = learn.predict


def predict(model, data):
    if model["config"]["family"] != "svr":
        return ORIGINAL_PREDICT(model, data)
    estimator = model["estimator"]
    if estimator.kernel != "rbf":
        raise ValueError("Only the registered RBF kernel is supported")
    norm = model["normalization"]
    support = estimator.support_vectors_
    support_norm = np.einsum("ij,ij->i", support, support)
    dual = estimator.dual_coef_[0]
    prediction = np.empty(len(data["x"]), dtype=float)
    for begin in range(0, len(prediction), 128):
        end = min(begin + 128, len(prediction))
        z = (np.asarray(data["x"][begin:end], dtype=float) - norm["mean"]) / norm["scale"]
        # At most 128 * n_support doubles; no full test-by-support Gram matrix.
        kernel = z @ support.T
        kernel *= -2
        kernel += np.einsum("ij,ij->i", z, z)[:, None]
        kernel += support_norm[None, :]
        np.maximum(kernel, 0, out=kernel)
        kernel *= -estimator._gamma
        np.exp(kernel, out=kernel)
        value = kernel @ dual + estimator.intercept_[0]
        gap = model["target_mean"] + model["target_scale"] * value
        if model["base"] is not None:
            gap += learn.gap_predict(model["base"], z)
        prediction[begin:end] = data["anchor"][begin:end] + gap
        learn.GUARD()
    return prediction
