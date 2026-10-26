# Recorded regression example

`telemetry.csv` contains the first 12 consecutive 2 Hz samples of sessions 60, 62, and 74, grouped in that fixed order. `session_id` instructs the CLI to reset history; it is not a model feature. `expected.csv` carries corresponding previously recorded predictions.

The source is Kirchgässner, Wallscheid and Böcker, Electric Motor Temperature v3, [DOI](https://doi.org/10.34740/KAGGLE/DSV/2161054), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Changes: selected fixed startup rows, retained only seven input channels, added a reset identifier, and paired them with the original frozen output. References use the original parser's float32 representation; frozen predictions have six decimal places.

Use `python scripts/verify_release.py` after installation. Agreement tolerance is 0.000001 °C because of the saved CSV precision. This is a software regression check, not model accuracy or a new test result. The startup slice intentionally retains the unknown-initial-state limitation.
