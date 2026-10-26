"""CLI contracts using tiny algebra fixtures, without a fitted model or target."""

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mizan_motor import INPUT_NAMES, MotorTemperatureEstimator

from fixtures import SAMPLES, ridge


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "model.json"
        self.input = self.directory / "telemetry.csv"
        self.output = self.directory / "predictions.csv"
        self.model.write_text(json.dumps(ridge()), encoding="utf-8")
        self.environment = os.environ.copy()
        source = str(Path(__file__).resolve().parents[1] / "src")
        previous = self.environment.get("PYTHONPATH")
        self.environment["PYTHONPATH"] = source + (os.pathsep + previous if previous else "")

    def invoke(self, text, *, include_model=True):
        self.input.write_text(text, encoding="utf-8", newline="")
        command = [sys.executable, "-m", "mizan_motor", "predict"]
        if include_model:
            command += ["--model", str(self.model)]
        command += ["--input", str(self.input), "--output", str(self.output)]
        return subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", env=self.environment,
            cwd=self.directory, check=False,
        )

    def csv_text(self, rows, *, sessions=None, target=False):
        stream = io.StringIO(newline="")
        names = list(INPUT_NAMES) + (["session_id"] if sessions is not None else [])
        names += ["pm"] if target else []
        writer = csv.writer(stream)
        writer.writerow(names)
        for index, row in enumerate(rows):
            values = list(row) + ([sessions[index]] if sessions is not None else [])
            writer.writerow(values + (["not-a-numeric-target"] if target else []))
        return stream.getvalue()

    def read_output(self):
        with self.output.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_named_channels_work_without_target_and_extra_target_is_ignored(self):
        for with_target in (False, True):
            result = self.invoke(self.csv_text(SAMPLES, target=with_target))
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = self.read_output()
            control = MotorTemperatureEstimator(ridge())
            self.assertEqual(len(rows), len(SAMPLES))
            for index, (row, sample) in enumerate(zip(rows, SAMPLES), 1):
                self.assertEqual(int(row["row"]), index)
                self.assertEqual(float(row["pm_estimate_c"]), control.step(sample))

    def test_every_session_boundary_resets_including_return_to_old_id(self):
        samples = [SAMPLES[0], SAMPLES[1], SAMPLES[1], SAMPLES[0]]
        sessions = ["A", "A", "B", "A"]
        result = self.invoke(self.csv_text(samples, sessions=sessions))
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.read_output()
        control = MotorTemperatureEstimator(ridge())
        previous = None
        for row, sample, session in zip(rows, samples, sessions):
            if session != previous:
                control.reset()
            self.assertEqual(row["session_id"], session)
            self.assertEqual(float(row["pm_estimate_c"]), control.step(sample))
            previous = session

    def test_invalid_row_reports_line_and_preserves_existing_output(self):
        self.output.write_text("existing output\n", encoding="utf-8")
        bad = SAMPLES[1].copy()
        bad[2] = "inf"
        result = self.invoke(self.csv_text([SAMPLES[0], bad]))
        self.assertEqual(result.returncode, 2)
        self.assertIn("line 3", result.stderr)
        self.assertIn("u_d", result.stderr)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing output\n")
        self.assertEqual(list(self.directory.glob("predictions.csv.*.tmp")), [])

    def test_missing_or_duplicate_headers_fail_explicitly(self):
        for text in ("ambient,coolant\n20,21\n", "ambient,ambient\n20,21\n", ""):
            result = self.invoke(text)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(self.output.exists())
            self.assertIn("error:", result.stderr)

    def test_missing_value_extra_fields_and_blank_session_fail_explicitly(self):
        header = ",".join(INPUT_NAMES)
        malformed = [
            header + "\n1,2,3\n",
            header + "\n1,2,3,4,5,6,7,8\n",
            self.csv_text([SAMPLES[0]], sessions=[""]),
        ]
        for text in malformed:
            result = self.invoke(text)
            self.assertEqual(result.returncode, 2)
            self.assertIn("line 2", result.stderr)
            self.assertFalse(self.output.exists())

    def test_model_path_is_required(self):
        result = self.invoke(self.csv_text([SAMPLES[0]]), include_model=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--model", result.stderr)
        self.assertFalse(self.output.exists())

    def test_output_cannot_replace_model_even_with_equivalent_path(self):
        original = self.model.read_bytes()
        self.output = self.directory / "unused" / ".." / self.model.name
        result = self.invoke(self.csv_text([SAMPLES[0]]))
        self.assertEqual(result.returncode, 2)
        self.assertIn("Model and output paths must be different", result.stderr)
        self.assertEqual(self.model.read_bytes(), original)
        self.assertEqual(list(self.directory.glob("model.json.*.tmp")), [])

    def test_csv_syntax_error_has_file_and_line_context(self):
        text = ",".join(INPUT_NAMES) + '\n"unterminated\n'
        result = self.invoke(text)
        self.assertEqual(result.returncode, 2)
        self.assertIn(str(self.input), result.stderr)
        self.assertIn("line 2", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
