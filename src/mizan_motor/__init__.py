"""Causal estimates of the published motor-temperature reference channel."""

from .api import (
    ModelValidationError,
    MotorTemperatureEstimator,
    SampleValidationError,
    load_model,
)
from .portable import INPUT_NAMES

__all__ = [
    "INPUT_NAMES",
    "ModelValidationError",
    "MotorTemperatureEstimator",
    "SampleValidationError",
    "load_model",
]
__version__ = "0.1.0"
