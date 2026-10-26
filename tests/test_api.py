"""Public contracts and original-algorithm agreement on algebra fixtures only."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from mizan_motor import (
    INPUT_NAMES, ModelValidationError, MotorTemperatureEstimator,
    SampleValidationError, load_model,
)
from mizan_motor.portable import Thermometer as OriginalAlgorithm

from fixtures import SAMPLES, common, network, rc, ridge


class EstimatorTests(unittest.TestCase):
    def test_api_preserves_original_algorithm_exactly(self):
        # portable.py is the unchanged research algorithm. This checks that input
        # mapping, validation, and transactional state do not alter its results.
        for make_model in (network, ridge, rc):
            model = make_model()
            original = OriginalAlgorithm(deepcopy(model))
            estimator = MotorTemperatureEstimator(model)
            for index, values in enumerate(SAMPLES):
                sample = dict(zip(INPUT_NAMES, values)) if index % 2 else tuple(values)
                self.assertEqual(estimator.step(sample), original.step(values))

    def test_known_network_and_ensemble_algebra(self):
        model = network()
        model["members"][0]["weights"][0][0][15] = 0.0
        model["members"][0]["weights"][0][0][90] = 0.0
        model["members"][0]["weights"][0][1][15] = 0.0
        second = deepcopy(model["members"][0])
        second["biases"][-1][0] = 2.5
        model["members"].append(second)
        # z[0]=1: members output 2.5 and 4.5; coolant anchor is exactly 20.
        self.assertEqual(MotorTemperatureEstimator(model).step([100, 20, 0, 0, 0, 0, 0]), 23.5)

    def test_order_matters_and_reset_discards_history(self):
        model = ridge()
        a, b = [0, 20, 0, 0, 0, 0, 0], [100, 20, 0, 0, 0, 0, 0]
        forward, reverse = MotorTemperatureEstimator(model), MotorTemperatureEstimator(model)
        forward.step(a)
        reverse.step(b)
        self.assertNotEqual(forward.step(b), reverse.step(a))
        forward.reset()
        self.assertEqual(forward.step(b), MotorTemperatureEstimator(model).step(b))
        forward.step(a)
        forward.new_session()
        self.assertEqual(forward.step(b), MotorTemperatureEstimator(model).step(b))

    def test_rejected_samples_leave_history_and_rc_state_unchanged(self):
        invalid = [
            [], SAMPLES[0] + [1], "1234567", {},
            {**dict(zip(INPUT_NAMES, SAMPLES[0])), "ambient": None},
        ]
        for value in (math.nan, math.inf, -math.inf, True, "invalid", 1e40):
            invalid.append([value, *SAMPLES[0][1:]])
        derived_overflow = SAMPLES[0].copy()
        derived_overflow[4] = 1e30  # Finite float32 input; squared feature cannot fit float32.
        invalid.append(derived_overflow)
        for make_model in (network, ridge, rc):
            for bad in invalid:
                with self.subTest(model=make_model.__name__, bad=repr(bad)):
                    estimator = MotorTemperatureEstimator(make_model())
                    control = MotorTemperatureEstimator(make_model())
                    estimator.step(SAMPLES[0])
                    control.step(SAMPLES[0])
                    with self.assertRaises(SampleValidationError):
                        estimator.step(bad)
                    self.assertEqual(estimator.step(SAMPLES[1]), control.step(SAMPLES[1]))

    def test_extra_target_is_ignored_and_caller_model_is_not_mutated(self):
        model = network()
        original = deepcopy(model)
        estimator = MotorTemperatureEstimator(model)
        control = MotorTemperatureEstimator(original)
        model["scale"][0] = 1000
        model["members"][0]["biases"][-1][0] = 999
        sample = dict(zip(INPUT_NAMES, SAMPLES[0]))
        sample["pm"] = object()  # The hidden target is never inspected or converted.
        self.assertEqual(estimator.step(sample), control.step(SAMPLES[0]))
        self.assertEqual(sample["ambient"], SAMPLES[0][0])

    def test_json_loading_and_path_constructor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(network()), encoding="utf-8")
            loaded = load_model(path)
            self.assertEqual(loaded, network())
            self.assertEqual(
                MotorTemperatureEstimator(path).step(SAMPLES[0]),
                MotorTemperatureEstimator(loaded).step(SAMPLES[0]),
            )
            path.write_text('{"kind":', encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "Cannot read JSON model"):
                load_model(path)


class ModelValidationTests(unittest.TestCase):
    def test_malformed_networks_fail_before_session_creation(self):
        cases = []
        for key, value in [
            ("kind", "unknown"), ("inputs", list(reversed(INPUT_NAMES))),
            ("sample_period_seconds", 1.0), ("dimension", 0),
            ("dimension", 106), ("dimension", True), ("mean", [0.0] * 104),
            ("target_scale", 0), ("members", []),
        ]:
            model = network()
            model[key] = value
            cases.append(model)
        for value in (0, -1, math.nan, math.inf):
            model = network()
            model["scale"][0] = value
            cases.append(model)
        model = network()
        model["members"][0]["biases"].pop()
        cases.append(model)
        model = network()
        model["members"][0]["weights"][0][0].pop()
        cases.append(model)
        model = network()
        model["members"][0]["biases"][0][0] = math.nan
        cases.append(model)
        model = network()
        model["members"][0]["weights"][-1].append([1.0, 1.0])
        model["members"][0]["biases"][-1].append(0.0)
        cases.append(model)
        for model in cases:
            with self.subTest(model=repr(model)[:120]):
                with self.assertRaises(ModelValidationError):
                    MotorTemperatureEstimator(model)

    def test_ridge_rc_svr_and_tree_dimensions_are_validated(self):
        short_ridge = ridge()
        short_ridge["coefficient"].pop()
        short_rc = rc()
        short_rc["coefficient"].pop()
        bad_svr = {
            **common("svr"), "target_mean": 0, "target_scale": 1,
            "gamma": 1, "svm_intercept": 0, "support": [[0] * 104], "dual": [1],
        }
        cyclic_tree = {
            **common("hgb"), "target_mean": 0, "target_scale": 1,
            "initial_prediction": 0,
            "trees": [{
                "leaf": [False], "feature": [0], "threshold": [0],
                "left": [0], "right": [0], "value": [0],
            }],
        }
        for model in (short_ridge, short_rc, bad_svr, cyclic_tree):
            with self.assertRaises(ModelValidationError):
                MotorTemperatureEstimator(model)


if __name__ == "__main__":
    unittest.main()
