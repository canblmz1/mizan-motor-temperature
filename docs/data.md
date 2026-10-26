# Data and reproducibility

**Dataset:** Electric Motor Temperature, version 3. Creators: Wilhelm Kirchgässner, Oliver Wallscheid, Joachim Böcker. [Publisher](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature); [DOI 10.34740/KAGGLE/DSV/2161054](https://doi.org/10.34740/KAGGLE/DSV/2161054). License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The source has 1,330,816 rows, 69 session IDs, and one motor/test bench. `measures_v2.csv` is 300,061,411 bytes. The downloaded source ZIP was 122,212,134 bytes with locally recorded SHA-256 `10e49e6220fa3c8a9d22c8bab8280ffe2d5e1e3380a6a18bcfc64ea2e8b56b92`. This is a local reproducibility fingerprint, not an independently supplied publisher checksum.

The full dataset and training feature arrays are excluded from this repository. Obtain the dataset from its publisher if you need the optional reference recipe. Installation and inference do not download anything.

Included data-derived artifacts:

- `examples/telemetry.csv`: first 12 consecutive samples from each test session, in fixed session order 60/62/74; seven predictor channels and a reset identifier only.
- `examples/expected.csv`: corresponding recorded references and frozen `p17`/`p06` predictions. The original parser stored references in float32; saved prediction columns are rounded to six decimals.
- `results/predictions.csv.gz`: the previously recorded 63,904 test reference and selected/neural/classical prediction rows.
- `results/all-methods.csv` and `results/benchmark.json`: recorded metrics for all separately frozen controls.
- `assets/heldout-temperatures.png`: complete-session plots from the original evaluation.

The samples were selected by row position, not error. They demonstrate numerical agreement, including startup behavior, and do not estimate population accuracy. The reference/prediction file is already evaluated test output, not a fresh blind test set. Reusing it for tuning cannot establish independent improvement.

Training and validation sessions were parsed separately from test numerical values. The original selection and checkpoint freeze preceded test parsing. The reference recipe retains this order, but its portable packaging has not been retrained or validated end to end in this release.

Attribution and share-alike terms for these artifacts are retained in [DATA_LICENSE.md](../DATA_LICENSE.md). Exact frozen artifact digests and changes made for distribution are recorded in [provenance.json](provenance.json).
