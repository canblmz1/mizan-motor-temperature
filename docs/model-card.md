# Model card

## Intended use

Recorded-data research and a reproducible streaming-inference demonstration for the `pm` channel in Electric Motor Temperature v3. The model estimates a measured magnet reference from seven observed channels and their causal histories.

## Frozen model

`p17`: mean of two fixed-seed ReLU MLPs with hidden widths 128 and 64, operating on 105 features. Fifteen instantaneous quantities, five causal EWMA histories, and selected history standard deviations represent thermal inertia. Only training rows determine normalization. Temperature is represented relative to the observed coolant anchor.

The common epoch was selected using all three validation sessions; the recorded run used epoch 100. Training choices, feature definitions, and split are retained in the reference recipe. The released JSON weights are unchanged.

The package also supplies `p04` instantaneous Ridge, `p06` EWMA Ridge, and `p20` constant-coefficient RC. The separately evaluated SVR results remain in the benchmark table; SVR weights are not part of this release. All 13 separately frozen family/view controls are reported.

## Information and population

- Inputs: ambient, coolant, d/q voltages, d/q currents, motor speed, and their causal histories.
- Excluded: hidden temperature at all times, stator temperatures, torque, future values, identity/time as predictors.
- Required sampling: ordered 0.5-second steps; the deposited CSV has no independent hardware timestamp vector.
- Population: one PMSM/test bench, 69 recorded sessions. A held-out session is not a new motor.
- Split: 63 train; validation 4/11/32; test 60/62/74.

The source paper describes `pm` as an average of four magnet-surface sensors. It is not a maximum-temperature measurement. Original metrology and missing-sample behavior were not independently measured for this software release.

## Recorded performance

| Session | Rows | MAE, °C | 95th absolute error, °C | Maximum absolute error, °C |
|---|---:|---:|---:|---:|
| 60 | 14,543 | 0.903 | 2.325 | 6.211 |
| 62 | 25,600 | 2.010 | 4.810 | 5.687 |
| 74 | 23,761 | 2.010 | 4.497 | 17.536 |

Session-macro MAE: 1.641 °C. Pooled MAE: 1.758 °C; RMSE: 2.370 °C. Coverage: all 63,904 test rows, including startup. The 17.536 °C underestimation occurs at session 74 startup. Unknown prior thermal state is an important limitation.

Validation-selected Ridge test MAE: 2.298 °C. A separately frozen SVR + Ridge scored 1.609 °C. The neural model passed the predeclared demonstration accuracy gate but failed the full neural-advantage gate because worst underestimation was not improved. Published TNN/LPTN results use different information/initialization recipes; direct state-of-the-art score comparisons are not claimed.

## Limits

Cross-motor transfer, uncertainty calibration, arbitrary sample spacing, hardware runtime, and physical safety thresholds have not been established. No maximum-temperature bound or control safety guarantee follows from these errors. This release reports established-method engineering results; it makes no algorithmic novelty claim.

Source experiment: 2026-10-20, source commit `0571110bb20fcfbdad88d91a04b04428a48536c3`. The publication package adds API validation and portable entry points without retraining. Numerical regression checks validate software equivalence, not new generalization performance.
