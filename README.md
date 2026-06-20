# 69 Robotic Physics Violation Audits

Submission-hardening version: v5 expanded hostile-review archive

Terminal decision: KILL_ARCHIVE for ICLR main conference.

The repository is retained as an archive of a falsified physics-audit mechanism. The v5 rebuild expands the MuJoCo contact benchmark, adds rare-valid and adversarial corruption regimes, calibrates explicit physics audits, evaluates strong residual/learned/reconstruction/conformal baselines, freezes the protocol before the final run, and reports the negative result without post-hoc filtering.

The proposed `physics_violation_audit_v5` does not survive the ICLR-main gate. Strong non-oracle baselines, especially `calibrated_learned_stack`, `hgb_failure_classifier`, `random_forest_ensemble`, and `pca_reconstruction_audit`, match or beat the explicit audit on hard corruption regimes. The explicit audit also exceeds the rare-valid false-positive gate and trails the best learned baselines at a fixed 5 percent false-positive budget.

Final PDF: `C:/Users/wangz/Downloads/69.pdf`

Final PDF pages: 46

Final PDF SHA256: `395FE00C4D5222E0BC0A4B58E434EAE11C6F7D7079B29E11FE0383BC07666546`

Desktop PDF copy: absent by design.

## Reproduce Real Evidence

```powershell
python src\run_experiment.py --seeds 8 --episodes 14 --ablation-episodes 10 --stress-episodes 8 --train-scenes 192 --workers 4 --splits nominal_valid rare_valid_bounce valid_clock_skew valid_low_friction_slip contact_corruption energy_work_corruption support_levitation actuator_saturation noncausal_teleport combined_violation_shift subtle_contact_corruption subtle_energy_corruption timestamp_noncausal_corruption adversarial_compensated_violation mixed_near_threshold --ablation-splits combined_violation_shift adversarial_compensated_violation mixed_near_threshold rare_valid_bounce --stress-splits combined_violation_shift adversarial_compensated_violation rare_valid_bounce --stress-levels 0.0 0.2 0.4 0.6 0.8 1.0 --results-dir results --figures-dir figures
```

The frozen run writes MuJoCo rollout summaries, raw method evaluations, metrics, seed metrics, pairwise comparisons, ablations, stress sweeps, fixed-FPR results, negative cases, learned-baseline summaries, and figures into `results/` and `figures/`.

## Rebuild Archive PDF

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_submission_pdf.ps1
```

Canonical local PDF: `C:/Users/wangz/Downloads/69.pdf`

## Validation

```powershell
python scripts\validate_submission_artifacts.py
```

Expected result: validation passes for row counts, figures, bright boxed citation settings, 25+ page PDF, Downloads-only output, and Desktop hygiene.
