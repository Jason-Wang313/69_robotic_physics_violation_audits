# Paper 69 Terminal Evidence

Decision: `KILL_ARCHIVE`

## Real-Evidence Rebuild
The v4 rebuild replaces the synthetic scaffold with a MuJoCo contact-manipulation benchmark. It generates real valid rollouts, injects controlled physics violations, trains learned baselines, evaluates audit methods, and reports main, ablation, and stress results.

Run command:

```powershell
python src\run_experiment.py
```

Generated evidence:
- 420 main MuJoCo rollout summaries.
- 3,780 main method-evaluation rows.
- 480 ablation rows.
- 1,200 stress-sweep rows.
- 5 seeds, 12 main rollouts per seed, 7 splits, 9 main methods.
- CSVs: training rollouts, raw rollouts, metrics, seed metrics, pairwise comparisons, ablations, stress sweep, negative cases.
- Figures: F1 by split, false positives by split, ablation F1, stress sweep.

## Combined Violation-Shift Results

| Method | F1 | Precision | Recall | False Positive |
|---|---:|---:|---:|---:|
| `random_flagger` | 0.667 | 1.000 | 0.500 | 0.000 |
| `kinematic_residual_threshold` | 1.000 | 1.000 | 1.000 | 0.000 |
| `energy_residual_threshold` | 1.000 | 1.000 | 1.000 | 0.000 |
| `contact_impulse_threshold` | 0.125 | 1.000 | 0.067 | 0.000 |
| `ensemble_dynamics_uncertainty` | 1.000 | 1.000 | 1.000 | 0.000 |
| `autoencoder_reconstruction_audit` | 1.000 | 1.000 | 1.000 | 0.000 |
| `supervised_failure_classifier` | 1.000 | 1.000 | 1.000 | 0.000 |
| `physics_violation_audit` | 1.000 | 1.000 | 1.000 | 0.000 |
| `oracle_violation_labels` | 1.000 | 1.000 | 1.000 | 0.000 |

## Nominal Valid False Positives

On nominal valid MuJoCo rollouts, `physics_violation_audit` false-flags 23.3% of traces. Autoencoder reconstruction, ensemble uncertainty, and the supervised classifier false-flag all nominal traces under the current calibration, which underscores that this benchmark is not yet a robust deployable audit.

## Ablation Results

| Ablation | F1 | Precision | Recall |
|---|---:|---:|---:|
| `full_physics_violation_audit` | 1.000 | 1.000 | 1.000 |
| `no_actuator_check` | 1.000 | 1.000 | 1.000 |
| `no_causality_check` | 1.000 | 1.000 | 1.000 |
| `no_contact_check` | 1.000 | 1.000 | 1.000 |
| `no_energy_check` | 1.000 | 1.000 | 1.000 |
| `no_support_check` | 1.000 | 1.000 | 1.000 |
| `no_friction_slip_check` | 0.000 | 0.000 | 0.000 |
| `scalar_residual_only` | 0.000 | 0.000 | 0.000 |

## Terminal Rationale
The central claim requires explicit physics checks to beat residual and learned baselines while keeping false positives low. They do not. Several baselines match the proposed audit on combined violations, and the audit has substantial false positives on valid traces. The honest action is `KILL_ARCHIVE`.
