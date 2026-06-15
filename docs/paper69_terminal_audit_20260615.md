# Paper 69 Terminal Audit

Date: 2026-06-15

Paper: `69_robotic_physics_violation_audits`

Decision: `KILL_ARCHIVE`

ICLR-main ready: no

## Commands Executed

- `python -m py_compile src\run_experiment.py`
- CSV finite/schema audit over `results/training_audit_rollouts.csv`, `results/training_summary.csv`, `results/physics_audit_rollouts.csv`, `results/physics_audit_raw.csv`, `results/physics_audit_metrics.csv`, `results/physics_audit_pairwise.csv`, `results/physics_audit_ablation.csv`, `results/physics_audit_ablation_raw.csv`, `results/raw_seed_metrics.csv`, `results/negative_cases.csv`, compatibility CSVs, and `results/stress_sweep.csv`.
- `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` in `paper`
- `Copy-Item paper\main.pdf C:\Users\wangz\Downloads\69.pdf -Force`

## Verified Evidence

- Real MuJoCo contact rollouts with controlled physics-violation corruptions are implemented in `src/run_experiment.py`.
- Training evidence contains 90 valid audit rollouts.
- Main rollout evidence contains 420 MuJoCo rollout summaries.
- Main evaluation evidence contains 3,780 method-evaluation rows: 7 splits, 5 seeds, 12 episodes per seed/split/method, and 9 methods.
- Ablation evidence contains 480 rows.
- Stress-sweep evidence contains 1,200 rows.
- Baselines include random flagging, kinematic residual thresholding, energy residual thresholding, contact impulse thresholding, ensemble dynamics uncertainty, autoencoder reconstruction, supervised failure classification, and oracle labels.
- CSV outputs are present, non-empty, and finite.
- BibTeX warnings from missing prior-work sort keys were fixed without inventing authors.
- The rebuilt PDF is `C:/Users/wangz/Downloads/69.pdf`.
- `C:/Users/wangz/Desktop/69.pdf` is absent.

## Fatal Results

The explicit physics-violation audit does not support an ICLR-main claim:

- Combined violation shift: `physics_violation_audit` reaches F1 `1.000`.
- Combined violation shift: `kinematic_residual_threshold`, `energy_residual_threshold`, `ensemble_dynamics_uncertainty`, `autoencoder_reconstruction_audit`, `supervised_failure_classifier`, and `oracle_violation_labels` also reach F1 `1.000`.
- Pairwise comparisons show zero F1 difference versus the residual, learned, and oracle baselines on this split.
- On nominal valid MuJoCo traces, the explicit audit false-flags `23.3%` of rollouts.
- Combined-shift ablations show several checks are redundant: removing actuator, causality, contact, energy, or support checks leaves F1 at `1.000`.

## Gate Decision

This paper satisfies the local evidence-package requirements for a real negative result: high-fidelity simulator evidence, controlled corruptions, residual and learned baselines, ablations, stress tests, uncertainty, negative cases, rebuilt PDF, corrected BibTeX metadata, corrected hostile-review documentation, and public repository.

It does not satisfy `STRONG_REVISE` because the explicit audit is not externally superior to simple residual and learned baselines and still has too many nominal false positives. The correct terminal state remains `KILL_ARCHIVE`.

Required revival work:

- use harder hardware logs or public embodied-agent traces where residual thresholds do not solve the task;
- reduce nominal false-positive rates on valid rare contact dynamics;
- prove each audit family is necessary through harder ablations;
- show the audit predicts real policy failures rather than only injected corruption artifacts;
- perform a manual full-paper related-work synthesis.
