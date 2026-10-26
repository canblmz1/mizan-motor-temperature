# Optional reference training recipe

This directory preserves the recorded experiment's feature and learning mathematics in a standalone CPU reference. It is separate from the dependency-free inference package. **The portable training/evaluation wrapper has not been retrained or validated end to end for this release.** The published weights and benchmark numbers come from the original completed run, not from executing this wrapper.

## Environment and data

Use Python 3.11 or newer and a separate environment. `requirements.txt` records the versions observed in the successful original run; `REFERENCE_ENVIRONMENT.json` records Python 3.13.15 and the known package versions. The original receipt did not record `threadpoolctl`'s version, so that dependency is not presented as an exact historical pin. These are observed versions, not a claim of validation across operating systems.

Obtain Electric Motor Temperature v3 from [its publisher](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature). Supply a local ZIP containing the exact `measures_v2.csv`. The runner verifies the uncompressed CSV size and SHA-256 from the original content-identity receipt; it does not download data. The ZIP fingerprint is retained in the local run lock, allowing equivalent compression while preventing source/test input changes.

```bash
python -m venv .venv-research
# Activate that environment using your shell's usual command.
python -m pip install -r research/requirements.txt
python research/run.py --help
```

## Explicit stages

The source command requires a new, nonexistent work directory. Choose a directory outside `research/` and keep the same directory and data file for all stages.

```bash
python research/run.py source --data-zip /path/to/electric_motor_v3.zip --work-dir .work/reference
python research/run.py test --data-zip /path/to/electric_motor_v3.zip --work-dir .work/reference
python research/export_models.py --data-zip /path/to/electric_motor_v3.zip --work-dir .work/reference
```

`source` runs the fixed 21-candidate comparison on the original 63 training and three validation sessions, selects by validation score, and freezes source/data/environment/selection/checkpoint digests. `test` verifies that freeze, reads only the three registered test sessions for evaluation, performs no fitting, and refuses repeated/partial test attempts. `export_models.py` exports the locally trained frozen checkpoints to JSON without fitting. This does not replace the original released weights automatically.

The code retains 105 causal features, the same candidate grid, equal-profile fitting weights, fixed seeds, common validation checkpoint selection, and full-session evaluation without warmup removal. `learn.py` and `fast_svr.py`, the feature/history mathematics, and the source-stage selection body were statically compared with the original source. The recorded cloud run's uniform-weight compatibility adapter for older scikit-learn MLP APIs and its fast-SVR inference check are retained. Nonuniform sample weights cannot be silently discarded.

## Packaging changes and limits

Hardcoded machine paths and unrelated monorepo preservation checks were replaced with explicit data/work arguments and local freeze verification. Cache/export paths are configurable. Publication removes account credentials, cloud dispatch machinery, and automatic execution. The code in this directory is reference software; installation of the inference package never imports or runs it.

The wrapper requests two numerical threads. Its sampled, cooperative guard stops at 2,048 MiB process RSS, less than 1,024 MiB available system memory, 2,400 seconds for `source`, or 300 seconds for `test`. These are **portable-wrapper limits**, not an operating-system sandbox or a restatement of the original run's measured resources. Interrupted stages are not silently resumed or extended.

The original test has already been opened and published. Running it again checks reproduction; it does not create a new independent evaluation or justify tuning to those outcomes. Native library/platform differences and the adapted wrapper may affect a fresh run. Use the shipped weights and `scripts/verify_release.py` for the verified inference path.
