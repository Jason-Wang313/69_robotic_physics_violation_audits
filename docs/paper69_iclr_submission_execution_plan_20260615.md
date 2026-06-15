# Paper 69 ICLR-Main Execution Plan

Date: 2026-06-15

Paper: `69_robotic_physics_violation_audits`

Goal: verify whether the current real MuJoCo physics-violation audit evidence can honestly support an ICLR-main-target submission, or whether the paper must remain `KILL_ARCHIVE` as a falsified benchmark/mechanism claim.

## Execution Gates

1. Reproducibility gate:
   - Compile `src/run_experiment.py`.
   - Confirm training, rollout-summary, main, seed, pairwise, ablation, stress-sweep, negative-case, and compatibility CSV outputs exist.
   - Confirm all CSV outputs are non-empty and finite.
   - Rebuild the PDF from `paper/main.tex` with BibTeX.

2. Evidence gate:
   - Confirm the benchmark uses real MuJoCo contact rollouts with controlled violation injections rather than synthetic probability tables.
   - Confirm five seeds, seven splits, nine main audit methods, confidence intervals, pairwise comparisons, ablations, stress sweeps, and negative cases.
   - Confirm baselines include random flagging, kinematic residual thresholding, energy residual thresholding, contact impulse thresholding, ensemble dynamics uncertainty, autoencoder reconstruction, supervised failure classification, and oracle labels.

3. Negative-claim gate:
   - Compare `physics_violation_audit` against residual and learned baselines under the combined violation shift.
   - Check whether explicit audit checks are necessary under ablations.
   - Check false-positive rate on nominal valid MuJoCo traces.
   - Fix stale documentation that still presents the archive reason as synthetic-only evidence rather than the current real MuJoCo falsification.

4. Artifact gate:
   - Rebuild `paper/main.pdf`.
   - Copy only `C:/Users/wangz/Downloads/69.pdf`.
   - Confirm `C:/Users/wangz/Desktop/69.pdf` is absent.
   - Confirm the GitHub repository is public, clean, and pushed.

## Decision Rule

Upgrade only if explicit physics audits beat residual and learned baselines while keeping false positives low on valid traces. If residual/learned baselines match the audit, ablations show redundant checks, or nominal false positives remain too high, keep the terminal decision as `KILL_ARCHIVE`.
