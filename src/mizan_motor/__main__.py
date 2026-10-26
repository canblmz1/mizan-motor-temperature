"""CSV prediction with explicit model selection and session boundaries."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

from .api import ModelValidationError, MotorTemperatureEstimator, SampleValidationError
from .portable import INPUT_NAMES


def _csv_rows(reader, path):
    try:
        yield from reader
    except csv.Error as error:
        raise ValueError(f"{path}: line {reader.reader.line_num}: {error}") from error


def _predict_csv(model_path: Path, input_path: Path, output_path: Path) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if model_path.resolve() == output_path.resolve():
        raise ValueError("Model and output paths must be different")
    estimator = MotorTemperatureEstimator(model_path)
    temporary = None
    rows = 0
    try:
        with input_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            try:
                names = reader.fieldnames
            except csv.Error as error:
                raise ValueError(f"{input_path}: line {reader.reader.line_num}: {error}") from error
            if names is None:
                raise ValueError(f"{input_path}: CSV header is missing")
            if len(names) != len(set(names)):
                raise ValueError(f"{input_path}: duplicate CSV column names")
            missing = [name for name in INPUT_NAMES if name not in names]
            if missing:
                raise ValueError(f"{input_path}: missing input columns: {', '.join(missing)}")
            with_session = "session_id" in names
            fields = ["row"] + (["session_id"] if with_session else []) + ["pm_estimate_c"]
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=output_path.parent,
                prefix=output_path.name + ".", suffix=".tmp", delete=False,
            ) as destination:
                temporary = Path(destination.name)
                writer = csv.DictWriter(destination, fieldnames=fields)
                writer.writeheader()
                previous_session = None
                for row in _csv_rows(reader, input_path):
                    try:
                        if None in row:
                            raise ValueError("row has more fields than the header")
                        session = row.get("session_id") if with_session else None
                        if with_session and (session is None or not session.strip()):
                            raise ValueError("session_id must be nonempty when its column is present")
                        if with_session and session != previous_session:
                            estimator.new_session()
                        prediction = estimator.step(row)
                    except (SampleValidationError, ValueError) as error:
                        raise ValueError(f"{input_path}: line {reader.line_num}: {error}") from error
                    rows += 1
                    output = {"row": rows, "pm_estimate_c": repr(prediction)}
                    if with_session:
                        output["session_id"] = session
                    writer.writerow(output)
                    previous_session = session
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mizan-motor")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Estimate the pm reference channel from 2 Hz CSV")
    predict.add_argument("--model", required=True, type=Path, help="Path to an explicit JSON model")
    predict.add_argument("--input", required=True, type=Path, help="CSV with all seven named inputs")
    predict.add_argument("--output", required=True, type=Path, help="Prediction CSV path")
    arguments = parser.parse_args(argv)
    try:
        _predict_csv(arguments.model, arguments.input, arguments.output)
    except (OSError, UnicodeError, csv.Error, ModelValidationError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
