# Paper 69 Rebuild Plan: Robotic Physics Violation Audits

## Terminal Objective
Rebuild `69_robotic_physics_violation_audits` into a real evidence package. The paper may be submission-ready only if physics-violation audits detect and predict unsafe or impossible robot policy behavior better than generic uncertainty, reconstruction error, residual thresholds, and learned failure classifiers. If those baselines match the audit mechanism, archive it.

## Central Claim Under Test
Robot policies and embodied foundation-model planners may appear successful while silently tolerating impossible or unsafe physics: object interpenetration, contact impulses inconsistent with motion, energy creation, friction-cone violations, actuator saturation, non-causal state jumps, and unsupported levitation. A useful audit should expose these violations from rollouts and predict downstream policy failure or unsafe execution.

## High-Fidelity Benchmark
- Build a MuJoCo contact-manipulation benchmark with executed trajectories from push, lift, slide, and obstacle-avoidance primitives.
- Generate two rollout classes:
  - physically valid rollouts from MuJoCo policies/baselines
  - corrupted or policy-induced violation rollouts with controlled physics edits or bad control actions
- Violation families:
  - object/table or object/finger interpenetration
  - contact impulse without plausible acceleration
  - motion without contact or support
  - energy increase inconsistent with actuator work
  - friction-cone/slip inconsistency
  - actuator saturation hidden by smooth state traces
  - non-causal teleport/state jump
  - impossible stable levitation

## Methods And Baselines
- `random_flagger`: lower bound.
- `kinematic_residual_threshold`: flags large pose/velocity residuals.
- `energy_residual_threshold`: flags work-energy mismatch.
- `contact_impulse_threshold`: flags impulse/contact outliers.
- `ensemble_dynamics_uncertainty`: learned one-step dynamics ensemble with variance threshold.
- `autoencoder_reconstruction_audit`: learned trajectory reconstruction residual baseline.
- `supervised_failure_classifier`: trained classifier over rollout statistics.
- `physics_violation_audit`: proposed method with explicit contact, support, energy, friction, actuator, and causality checks.
- `oracle_violation_labels`: upper bound with access to injected corruption labels.

## Required Experiments
- Main benchmark: at least 5 seeds, 10-12 rollouts per seed per split and method, real MuJoCo rollouts for uncorrupted behavior.
- Splits:
  - nominal valid rollouts
  - contact corruption
  - energy/work corruption
  - support/levitation corruption
  - actuator saturation
  - non-causal teleport
  - combined violation shift
- Metrics:
  - violation detection F1/AUROC proxy
  - false-positive rate on valid rollouts
  - unsafe-policy prediction accuracy
  - calibration by violation severity
  - runtime/feature cost
- Ablations:
  - no contact check
  - no support check
  - no energy check
  - no friction/slip check
  - no actuator check
  - no causality check
  - scalar residual only
- Pairwise seed comparisons against uncertainty, autoencoder, supervised classifier, energy residual, and kinematic residual baselines.
- Stress sweep over noise, corruption severity, contact stiffness, friction, and actuator limits.
- Negative cases: sensor timestamp skew, legal but rare bouncing contact, out-of-distribution deformable object, and semantic task violation with valid physics.

## Submission-Readiness Gate
To be ICLR-main ready, the proposed audit must:
- beat every non-oracle baseline on combined violation shift and at least four of six non-nominal splits
- keep false positives on valid MuJoCo rollouts low enough to be useful
- show that explicit physics families matter through ablations
- predict unsafe execution, not merely detect injected artifacts
- include honest hostile prior-work discussion and limitations

## Terminal Decision Rules
- `SUBMISSION_READY_CANDIDATE`: only if the audit clears all empirical gates and supports a strong contribution.
- `STRONG_REVISE`: if explicit physics checks help but lack hardware/public benchmark breadth or manual related-work depth.
- `KILL_ARCHIVE`: if learned uncertainty, autoencoder residuals, supervised classifiers, scalar residuals, or simple energy/contact thresholds match the proposed audit.

## Resource Discipline
Keep RAM light with compact MuJoCo rollouts, trajectory summaries instead of full videos, small learned baselines, compact CSVs, and at most four workers. Do not reduce rigor: preserve real rollouts, seeds, baselines, ablations, stress tests, uncertainty, and terminal-failure analysis.

## Deliverables
- Rewritten `src/run_experiment.py` with MuJoCo rollout generation, violation injection, audit methods, learned baselines, and evaluation metrics.
- Updated requirements, README, child status, claims, gate, readiness, audit, and terminal evidence docs.
- CSV results, pairwise comparisons, ablations, stress sweep, negative cases, figures, and learned-baseline summaries.
- Rewritten paper and compiled `C:/Users/wangz/Downloads/69.pdf` only.
- Public GitHub repo pushed with final commit.
- Root reports updated before Paper 70 starts.
