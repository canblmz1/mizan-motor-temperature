"""Small parameter/algebra fixtures; no measured data or physical simulation."""

from mizan_motor import INPUT_NAMES


def common(kind="network"):
    return {
        "kind": kind,
        "inputs": list(INPUT_NAMES),
        "sample_period_seconds": 0.5,
        "dimension": 105,
        "mean": [0.0] * 105,
        "scale": [1.0] * 105,
    }


def network():
    positive, negative = [0.0] * 105, [0.0] * 105
    positive[0], positive[15], positive[90] = 1.0, 0.5, 0.25
    negative[0], negative[15] = -1.0, -0.5
    member = {
        "weights": [[positive, negative], [[2.0, 3.0]]],
        "biases": [[0.0, 0.0], [0.5]],
    }
    return {
        **common(),
        "target_mean": 0.0,
        "target_scale": 1.0,
        "base": None,
        "members": [member],
    }


def ridge():
    coefficient = [0.0] * 105
    coefficient[15] = 1.0  # The 10 s causal ambient history, rather than elapsed time.
    return {**common("ewma_ridge"), "coefficient": coefficient, "intercept": 0.0}


def rc():
    return {
        "kind": "constant_RC",
        "inputs": list(INPUT_NAMES),
        "sample_period_seconds": 0.5,
        "coefficient": [1.0, 0.5, 0.25, 0.0, 0.0],
        "initial": {
            "mean": [0.0] * 105,
            "scale": [1.0] * 105,
            "coefficient": [0.0] * 15,
            "intercept": 0.0,
        },
    }


SAMPLES = [
    [25.0, 50.0, 250.0, 250.0, 100.0, 100.0, 3000.0],
    [100.0, 25.0, -250.0, 0.0, 100.0, -100.0, 6000.0],
    [0.0, 20.0, 0.0, 10.0, 0.0, 0.0, 99.0],
    [0.0, 20.0, 0.0, 10.0, 0.0, 0.0, 100.0],
]
