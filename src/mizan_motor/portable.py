# Provenance: mizan_motor_virtual_sensor_v1/portable.py, research commit 17b7c7a.
# Numerical algorithm preserved verbatim; public validation lives in api.py.

"""Standard-library inference from seven online channels; no hidden temperature."""

from __future__ import annotations

import math
import struct

INPUT_NAMES = ("ambient", "coolant", "u_d", "u_q", "i_d", "i_q", "motor_speed")
TAUS = (10, 60, 300, 900, 1800)
VAR_INPUTS = (2, 3, 4, 5, 6)


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class OnlineHistory:
    def __init__(self):
        self.mean = None
        self.square = None

    def step(self, values):
        if len(values) != 7 or not all(math.isfinite(v) for v in values):
            raise ValueError("Seven finite online measurements are required")
        ambient, coolant, ud, uq, i_d, i_q, speed = values
        rotating = abs(speed) >= 100
        base = [
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
            10 * uq / speed if rotating else 0.0,
            10 * ud / speed if rotating else 0.0,
            float(rotating),
        ]
        observed_square = [base[i] * base[i] for i in VAR_INPUTS]
        if self.mean is None:
            self.mean = [base.copy() for _ in TAUS]
            self.square = [observed_square.copy() for _ in TAUS]
        else:
            for j, tau in enumerate(TAUS):
                alpha = 1 - math.exp(-0.5 / tau)
                for i in range(15):
                    self.mean[j][i] += alpha * (base[i] - self.mean[j][i])
                for i in range(5):
                    self.square[j][i] += alpha * (observed_square[i] - self.square[j][i])
        feature = base + [v for row in self.mean for v in row]
        feature += [
            math.sqrt(max(0.0, self.square[j][k] - self.mean[j][i] * self.mean[j][i]))
            for j in (1, 2, 3)
            for k, i in enumerate(VAR_INPUTS)
        ]
        return [float32(v) for v in feature], float32(coolant)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def tree_value(tree, z):
    node = 0
    while not tree["leaf"][node]:
        feature = tree["feature"][node]
        node = tree["left"][node] if z[feature] <= tree["threshold"][node] else tree["right"][node]
    return tree["value"][node]


def predict_features(model, feature, anchor, state=None):
    kind = model["kind"]
    if kind in {"coolant", "constant_gap"}:
        return anchor + model["gap"], None
    if kind == "constant_RC":
        if state is None:
            initial = model["initial"]
            z = [
                (v - m) / s
                for v, m, s in zip(
                    feature[:15], initial["mean"][:15], initial["scale"][:15], strict=True
                )
            ]
            state = anchor + dot(z, initial["coefficient"]) + initial["intercept"]
        prediction = state
        c = model["coefficient"]
        conductance = (c[0] + c[1]) / 100
        forcing = (
            c[0] * feature[1]
            + c[1] * feature[0]
            + c[2] * feature[7]
            + c[3] * feature[10]
            + c[4] * feature[11]
        )
        if conductance > 1e-12:
            decay = math.exp(-0.5 * conductance)
            state = decay * state + (1 - decay) * forcing / conductance
        else:
            state += 0.5 * forcing
        return prediction, state
    dimension = model["dimension"]
    z = [
        (v - m) / s
        for v, m, s in zip(
            feature[:dimension], model["mean"][:dimension], model["scale"][:dimension], strict=True
        )
    ]
    if kind.endswith("ridge"):
        return anchor + dot(z, model["coefficient"]) + model["intercept"], None
    if kind == "network":
        outputs = []
        for member in model["members"]:
            hidden = z
            for layer, bias in zip(member["weights"], member["biases"], strict=True):
                hidden = [dot(hidden, column) + b for column, b in zip(layer, bias, strict=True)]
                if len(bias) != 1:
                    hidden = [max(0.0, value) for value in hidden]
            outputs.append(hidden[0])
        standardized = sum(outputs) / len(outputs)
    elif kind == "hgb":
        standardized = model["initial_prediction"] + sum(tree_value(t, z) for t in model["trees"])
    elif kind == "extra_trees":
        cast = [float32(v) for v in z]
        standardized = sum(tree_value(t, cast) for t in model["trees"]) / len(model["trees"])
    elif kind == "svr":
        standardized = model["svm_intercept"] + sum(
            weight
            * math.exp(-model["gamma"] * sum((a - b) ** 2 for a, b in zip(z, support, strict=True)))
            for support, weight in zip(model["support"], model["dual"], strict=True)
        )
    else:
        raise ValueError(kind)
    gap = model["target_mean"] + model["target_scale"] * standardized
    if model.get("base") is not None:
        gap += dot(z, model["base"]["coefficient"]) + model["base"]["intercept"]
    return anchor + gap, None


class Thermometer:
    def __init__(self, model):
        self.model = model
        self.history = OnlineHistory()
        self.state = None

    def step(self, values):
        feature, anchor = self.history.step(values)
        prediction, self.state = predict_features(self.model, feature, anchor, self.state)
        return prediction
