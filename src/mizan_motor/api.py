"""Validated, transactional access to the frozen 2 Hz inference algorithm."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from os import PathLike
from pathlib import Path

from .portable import INPUT_NAMES, OnlineHistory, float32, predict_features

FEATURE_DIMENSION = 105
MODEL_KINDS = {
    "coolant", "constant_gap", "constant_RC", "instant_ridge", "ewma_ridge",
    "network", "hgb", "extra_trees", "svr",
}


class ModelValidationError(ValueError):
    """A model is incompatible with the portable inference schema."""


class SampleValidationError(ValueError):
    """A sample cannot produce a finite estimate; session state is unchanged."""


def _object(value, name):
    if not isinstance(value, dict):
        raise ModelValidationError(f"{name} must be a JSON object")
    return value


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelValidationError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (positive and value <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ModelValidationError(f"{name} must be a {qualifier} number")
    return value


def _array(value, name, length=None, *, positive=False):
    if not isinstance(value, list) or (length is not None and len(value) != length):
        suffix = f" of length {length}" if length is not None else ""
        raise ModelValidationError(f"{name} must be an array{suffix}")
    for index, number in enumerate(value):
        _number(number, f"{name}[{index}]", positive=positive)
    return value


def _normalization(model, dimension, name):
    mean = _array(model.get("mean"), f"{name}.mean")
    scale = _array(model.get("scale"), f"{name}.scale", positive=True)
    if len(mean) != len(scale) or len(mean) not in {dimension, FEATURE_DIMENSION}:
        raise ModelValidationError(
            f"{name}.mean and {name}.scale must both have length "
            f"{dimension} or {FEATURE_DIMENSION}"
        )


def _ridge(model, dimension, name, *, normalization=False):
    _object(model, name)
    _array(model.get("coefficient"), f"{name}.coefficient", dimension)
    _number(model.get("intercept"), f"{name}.intercept")
    if normalization or "mean" in model or "scale" in model:
        _normalization(model, dimension, name)


def _network(model, dimension):
    members = model.get("members")
    if not isinstance(members, list) or not members:
        raise ModelValidationError("model.members must be a nonempty array")
    for member_index, member in enumerate(members):
        name = f"model.members[{member_index}]"
        _object(member, name)
        weights, biases = member.get("weights"), member.get("biases")
        if not isinstance(weights, list) or not weights:
            raise ModelValidationError(f"{name}.weights must contain at least one layer")
        if not isinstance(biases, list) or len(biases) != len(weights):
            raise ModelValidationError(f"{name}.biases must match the number of weight layers")
        width = dimension
        for layer_index, (layer, bias) in enumerate(zip(weights, biases, strict=True)):
            layer_name = f"{name}.weights[{layer_index}]"
            if not isinstance(layer, list) or not layer:
                raise ModelValidationError(f"{layer_name} must be a nonempty array")
            _array(bias, f"{name}.biases[{layer_index}]", len(layer))
            for output_index, column in enumerate(layer):
                _array(column, f"{layer_name}[{output_index}]", width)
            width = len(layer)
        if width != 1:
            raise ModelValidationError(f"{name} must have exactly one final output")


def _trees(model, dimension):
    trees = model.get("trees")
    if not isinstance(trees, list) or not trees:
        raise ModelValidationError("model.trees must be a nonempty array")
    for tree_index, tree in enumerate(trees):
        name = f"model.trees[{tree_index}]"
        _object(tree, name)
        leaf = tree.get("leaf")
        if not isinstance(leaf, list) or not leaf:
            raise ModelValidationError(f"{name}.leaf must be a nonempty array")
        count = len(leaf)
        for key in ("feature", "left", "right", "threshold", "value"):
            array = tree.get(key)
            if not isinstance(array, list) or len(array) != count:
                raise ModelValidationError(f"{name}.{key} must have length {count}")
        _array(tree["threshold"], f"{name}.threshold", count)
        _array(tree["value"], f"{name}.value", count)
        for node, is_leaf in enumerate(leaf):
            if type(is_leaf) not in (bool, int) or is_leaf not in (False, True):
                raise ModelValidationError(f"{name}.leaf[{node}] must be boolean or 0/1")
            for key in ("feature", "left", "right"):
                if type(tree[key][node]) is not int:
                    raise ModelValidationError(f"{name}.{key}[{node}] must be an integer")
            if not is_leaf:
                if not 0 <= tree["feature"][node] < dimension:
                    raise ModelValidationError(f"{name}.feature[{node}] is outside model.dimension")
                for key in ("left", "right"):
                    if not 0 <= tree[key][node] < count:
                        raise ModelValidationError(f"{name}.{key}[{node}] is outside the tree")
        # A malformed child cycle would otherwise make the original traversal loop forever.
        colors = [0] * count
        for start in range(count):
            stack = [(start, False)]
            while stack:
                node, leaving = stack.pop()
                if leaving:
                    colors[node] = 2
                elif colors[node] == 1:
                    raise ModelValidationError(f"{name} contains a child cycle")
                elif colors[node] == 0:
                    colors[node] = 1
                    stack.append((node, True))
                    if not leaf[node]:
                        stack.extend([(tree["left"][node], False), (tree["right"][node], False)])


def _validate_model(model):
    _object(model, "model")
    kind = model.get("kind")
    if not isinstance(kind, str) or kind not in MODEL_KINDS:
        raise ModelValidationError(f"Unsupported model.kind: {kind!r}")
    if model.get("inputs") != list(INPUT_NAMES):
        raise ModelValidationError(f"model.inputs must be {list(INPUT_NAMES)!r} in this order")
    period = _number(model.get("sample_period_seconds"), "model.sample_period_seconds")
    if period != 0.5:
        raise ModelValidationError("model.sample_period_seconds must be 0.5 (2 Hz)")
    if "feature_names" in model:
        names = model["feature_names"]
        if not isinstance(names, list) or len(names) != FEATURE_DIMENSION or not all(
            isinstance(name, str) for name in names
        ):
            raise ModelValidationError("model.feature_names must contain 105 strings")
    if kind in {"coolant", "constant_gap"}:
        _number(model.get("gap"), "model.gap")
        return
    if kind == "constant_RC":
        _array(model.get("coefficient"), "model.coefficient", 5)
        _ridge(model.get("initial"), 15, "model.initial", normalization=True)
        return
    dimension = model.get("dimension")
    if type(dimension) is not int or not 1 <= dimension <= FEATURE_DIMENSION:
        raise ModelValidationError("model.dimension must be an integer between 1 and 105")
    _normalization(model, dimension, "model")
    if kind.endswith("ridge"):
        _ridge(model, dimension, "model")
        return
    _number(model.get("target_mean"), "model.target_mean")
    _number(model.get("target_scale"), "model.target_scale", positive=True)
    if model.get("base") is not None:
        _ridge(model["base"], dimension, "model.base")
    if kind == "network":
        _network(model, dimension)
    elif kind in {"hgb", "extra_trees"}:
        if kind == "hgb":
            _number(model.get("initial_prediction"), "model.initial_prediction")
        _trees(model, dimension)
    elif kind == "svr":
        _number(model.get("gamma"), "model.gamma", positive=True)
        _number(model.get("svm_intercept"), "model.svm_intercept")
        support = model.get("support")
        if not isinstance(support, list) or not support:
            raise ModelValidationError("model.support must be a nonempty array")
        _array(model.get("dual"), "model.dual", len(support))
        for index, vector in enumerate(support):
            _array(vector, f"model.support[{index}]", dimension)


def load_model(path: str | PathLike) -> dict:
    """Read and validate a JSON model; no downloads, training, or inference."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            model = json.load(stream)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ModelValidationError(f"Cannot read JSON model {path}: {error}") from error
    _validate_model(model)
    return model


def _sample_values(sample):
    if isinstance(sample, Mapping):
        missing = [name for name in INPUT_NAMES if name not in sample]
        if missing:
            raise SampleValidationError(f"Missing input channels: {', '.join(missing)}")
        values = [sample[name] for name in INPUT_NAMES]
    elif isinstance(sample, Sequence) and not isinstance(sample, (str, bytes, bytearray)):
        if len(sample) != len(INPUT_NAMES):
            raise SampleValidationError("A sample sequence must have exactly seven values")
        values = list(sample)
    else:
        raise SampleValidationError("A sample must be a channel mapping or seven-value sequence")
    converted = []
    for name, value in zip(INPUT_NAMES, values, strict=True):
        try:
            if isinstance(value, bool) or value is None:
                raise ValueError
            value = float(value)
            if not math.isfinite(value) or not math.isfinite(float32(value)):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as error:
            raise SampleValidationError(
                f"Input channel {name!r} must be finite and representable as float32"
            ) from error
        converted.append(value)
    return converted


class MotorTemperatureEstimator:
    """One ordered 2 Hz session estimating the published ``pm`` reference channel.

    Call ``reset()`` before a new session. The model and filter state are private
    copies; a rejected step leaves both history and RC state unchanged.
    """

    def __init__(self, model_or_path):
        if isinstance(model_or_path, (str, PathLike)):
            self._model = load_model(model_or_path)
        elif isinstance(model_or_path, Mapping):
            self._model = deepcopy(dict(model_or_path))
            _validate_model(self._model)
        else:
            raise ModelValidationError("Expected a model mapping or JSON model path")
        self.reset()

    def reset(self) -> None:
        """Discard all causal history and state before the next sample."""
        self._history = OnlineHistory()
        self._state = None

    def new_session(self) -> None:
        """Start a fresh session; equivalent to ``reset()``."""
        self.reset()

    def step(self, sample) -> float:
        """Consume one sample, in INPUT_NAMES order or as named channels."""
        values = _sample_values(sample)
        candidate_history = deepcopy(self._history)
        try:
            feature, anchor = candidate_history.step(values)
            if not all(math.isfinite(value) for value in feature):
                raise ValueError("derived features are nonfinite")
            prediction, candidate_state = predict_features(
                self._model, feature, anchor, self._state
            )
            if not math.isfinite(prediction) or (
                candidate_state is not None and not math.isfinite(candidate_state)
            ):
                raise ValueError("prediction or state is nonfinite")
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            raise SampleValidationError(f"Sample cannot be evaluated finitely: {error}") from error
        self._history, self._state = candidate_history, candidate_state
        return prediction
