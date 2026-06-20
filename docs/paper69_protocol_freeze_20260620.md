# Paper 69 Protocol Freeze

Date: 2026-06-20

Freeze status: frozen before full run.

## Frozen Implementation

The frozen implementation is `src/run_experiment.py` after the Dev1 stacker bootstrap repair and the Dev2 paired-stat/ablation-gate repairs recorded in `docs/paper69_development_log_20260620.md`.

No further method tuning is allowed after this document. Recoverable failures may be fixed only if they are execution, serialization, plotting, validation, or PDF-generation defects that do not alter the method or decision gates.

## Frozen Full Command

```powershell
python src\run_experiment.py --seeds 8 --episodes 14 --ablation-episodes 10 --stress-episodes 8 --train-scenes 192 --workers 4 --splits nominal_valid rare_valid_bounce valid_clock_skew valid_low_friction_slip contact_corruption energy_work_corruption support_levitation actuator_saturation noncausal_teleport combined_violation_shift subtle_contact_corruption subtle_energy_corruption timestamp_noncausal_corruption adversarial_compensated_violation mixed_near_threshold --ablation-splits combined_violation_shift adversarial_compensated_violation mixed_near_threshold rare_valid_bounce --stress-splits combined_violation_shift adversarial_compensated_violation rare_valid_bounce --stress-levels 0.0 0.2 0.4 0.6 0.8 1.0 --results-dir results --figures-dir figures
```

## Frozen Evidence Scale

- Training rows: 192.
- Main rollout rows: 1,680.
- Main method-evaluation rows: 25,200.
- Main methods: 15.
- Main splits: 15.
- Seed metric rows: 1,800.
- Pairwise comparison rows: 210.
- Ablation rows: 3,840.
- Ablation methods: 12.
- Ablation splits: 4.
- Stress rows: 9,216.
- Stress methods: 8.
- Stress splits: 3.
- Stress levels: 6.

## Frozen Decision Gates

- KILL_ARCHIVE if v5 fails to beat the strongest non-oracle hard-corruption aggregate baseline by at least 0.030 F1.
- KILL_ARCHIVE if v5 fails to beat the strongest non-oracle combined/adversarial aggregate baseline by at least 0.030 F1.
- KILL_ARCHIVE if v5 exceeds 0.100 false-positive rate on any rare-valid split.
- KILL_ARCHIVE if v5 recall at 5% false-positive budget trails the best non-oracle baseline by more than 0.030.
- KILL_ARCHIVE if any removed-component ablation matches or beats full v5 within 0.020 F1 on a corruption split.
- KILL_ARCHIVE if v5 loses the maximum-stress gate to any non-oracle baseline.
- STRONG_REVISE only if all local gates pass; ICLR-main readiness still requires real robot or public benchmark validation.

## Artifact Rules

- Write the final numbered PDF only to `C:/Users/wangz/Downloads/69.pdf`.
- Do not copy any PDF to the visible Desktop.
- Preserve frozen CSVs, figures, generated tables, validation script, and public GitHub repository state.
